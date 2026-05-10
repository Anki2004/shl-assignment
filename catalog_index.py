

import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

import faiss

# ---------------------------------------------------------------------------
# Embedder — try sentence-transformers first, fall back to TF-IDF
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

CATALOG_PATH = Path("data/catalog.json")
INDEX_PATH   = Path("data/catalog.faiss")
META_PATH    = Path("data/catalog_meta.pkl")
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """
    Wraps sentence-transformers OR TF-IDF depending on what's available.
    Interface is the same either way: .encode(texts) → np.ndarray
    """

    def __init__(self):
        self._st_model = None
        self._tfidf    = None
        self._mode     = None

    def fit_or_load(self, docs: Optional[list[str]] = None, *, force_tfidf=False):
        """
        If sentence-transformers is available and internet worked, use it.
        Otherwise fall back to TF-IDF (requires docs for fitting).
        """
        if _HAS_ST and not force_tfidf:
            try:
                self._st_model = SentenceTransformer(EMBED_MODEL)
                self._mode = "st"
                print(f"Embedder: using sentence-transformers ({EMBED_MODEL})")
                return
            except Exception as e:
                print(f"sentence-transformers unavailable ({e}), falling back to TF-IDF")

        # TF-IDF fallback
        assert docs is not None, "docs required for TF-IDF fitting"
        self._tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8192,
            sublinear_tf=True,
        )
        self._tfidf.fit(docs)
        self._mode = "tfidf"
        print(f"Embedder: using TF-IDF (vocab size {len(self._tfidf.vocabulary_)})")

    def load_tfidf(self, tfidf):
        """Load a pre-fitted TF-IDF model (from pickle)."""
        self._tfidf = tfidf
        self._mode  = "tfidf"

    def load_st(self):
        """Load sentence-transformers model at runtime."""
        if _HAS_ST:
            try:
                self._st_model = SentenceTransformer(EMBED_MODEL)
                self._mode = "st"
                return
            except Exception:
                pass
        raise RuntimeError("sentence-transformers not available")

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._mode == "st":
            vecs = self._st_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return np.array(vecs, dtype="float32")
        elif self._mode == "tfidf":
            mat = self._tfidf.transform(texts).toarray().astype("float32")
            mat = sk_normalize(mat, norm="l2")
            return mat
        else:
            raise RuntimeError("Embedder not initialised. Call fit_or_load() first.")

    @property
    def mode(self):
        return self._mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_doc_string(item: dict) -> str:
    """
    Create a rich text representation of an assessment for embedding.
    """
    name        = item.get("name", "")
    description = item.get("description", "")
    keys        = ", ".join(item.get("keys", []))
    job_levels  = ", ".join(item.get("job_levels", []))
    duration    = item.get("duration", "")
    remote      = "remote" if item.get("remote") == "yes" else ""
    adaptive    = "adaptive" if item.get("adaptive") == "yes" else ""

    parts = [
        f"Assessment: {name}.",
        f"Description: {description}",
        f"Test type: {keys}.",
        f"Suitable for: {job_levels}." if job_levels else "",
        f"Duration: {duration}."       if duration    else "",
        remote,
        adaptive,
    ]
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# CatalogIndex
# ---------------------------------------------------------------------------

class CatalogIndex:
    """
    Holds the FAISS index and catalog metadata.
    Use .search() to retrieve the top-k most relevant assessments.
    """

    def __init__(self):
        self.embedder  = Embedder()
        self.index:    Optional[faiss.Index] = None
        self.metadata: list[dict]            = []
        self.catalog:  list[dict]            = []
        self._url_set: set[str]              = set()

    # ------------------------------------------------------------------
    # Build (offline, once)
    # ------------------------------------------------------------------

    def build(self, catalog_path: Path = CATALOG_PATH):
        print(f"Loading catalog from {catalog_path} ...")
        raw = json.loads(catalog_path.read_text(encoding = "utf-8"))
        self.catalog  = raw
        self._url_set = {item["link"] for item in raw}
        print(f"  {len(raw)} items loaded.")

        docs = [_build_doc_string(item) for item in raw]

        # Try sentence-transformers, fall back to TF-IDF
        self.embedder.fit_or_load(docs)
        embeddings = self.embedder.encode(docs)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.metadata = [
            {
                "entity_id":   item["entity_id"],
                "name":        item["name"],
                "link":        item["link"],
                "keys":        item.get("keys", []),
                "job_levels":  item.get("job_levels", []),
                "duration":    item.get("duration", ""),
                "remote":      item.get("remote", ""),
                "adaptive":    item.get("adaptive", ""),
                "description": item.get("description", ""),
            }
            for item in raw
        ]

        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "catalog":  self.catalog,
                "embedder_mode": self.embedder.mode,
                "tfidf": self.embedder._tfidf,   # None if using sentence-transformers
            }, f)

        print(f"Index saved  → {INDEX_PATH}")
        print(f"Meta  saved  → {META_PATH}")
        print(f"Mode         : {self.embedder.mode}")

    # ------------------------------------------------------------------
    # Load (runtime)
    # ------------------------------------------------------------------

    def load(self):
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run: python catalog_index.py --build"
            )

        self.index = faiss.read_index(str(INDEX_PATH))

        with open(META_PATH, "rb") as f:
            saved = pickle.load(f)

        self.metadata = saved["metadata"]
        self.catalog  = saved["catalog"]
        self._url_set = {item["link"] for item in self.catalog}

        mode = saved.get("embedder_mode", "tfidf")
        if mode == "tfidf":
            self.embedder.load_tfidf(saved["tfidf"])
        else:
            self.embedder.load_st()

        print(f"CatalogIndex loaded: {len(self.metadata)} assessments (mode: {mode}).")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 15,
        filter_job_levels: Optional[list[str]] = None,
        filter_keys: Optional[list[str]] = None,
    ) -> list[dict]:
        q_vec = self.embedder.encode([query])

        fetch_k = min(top_k * 4, self.index.ntotal)
        scores, indices = self.index.search(q_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = dict(self.metadata[idx])
            meta["_score"] = float(score)

            if filter_job_levels:
                item_levels = set(meta.get("job_levels", []))
                if item_levels and not item_levels.intersection(set(filter_job_levels)):
                    continue

            if filter_keys:
                item_keys = set(meta.get("keys", []))
                if not item_keys.intersection(set(filter_keys)):
                    continue

            results.append(meta)
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_by_name(self, name: str) -> Optional[dict]:
        name_lower = name.lower()
        for item in self.catalog:
            if name_lower in item["name"].lower():
                return item
        return None

    def is_valid_url(self, url: str) -> bool:
        return url in self._url_set

    def get_all_urls(self) -> set[str]:
        return self._url_set


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--test",  action="store_true")
    args = parser.parse_args()

    idx = CatalogIndex()

    if args.build:
        idx.build()

    if args.test:
        idx.load()
        results = idx.search("Java developer stakeholder communication mid level", top_k=5)
        for r in results:
            print(f"  [{r['_score']:.3f}] {r['name']}  —  {r['keys']}")