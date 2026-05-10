
import json
from pathlib import Path
from collections import Counter

CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"


def validate_and_analyze():
    if not CATALOG_PATH.exists():
        print(f"ERROR: {CATALOG_PATH} not found.")
        print("Please create data/catalog.json and paste the catalog JSON array into it.")
        return None

    raw = CATALOG_PATH.read_text(encoding="utf-8").strip()
    if not raw or raw in ("", "[]", "null"):
        print(f"ERROR: {CATALOG_PATH} is empty.")
        print("Please paste the full catalog JSON array into data/catalog.json")
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in catalog.json: {e}")
        return None

    if not isinstance(data, list) or len(data) == 0:
        print("ERROR: catalog.json must be a non-empty JSON array.")
        return None

    # ── Validate required fields ──────────────────────────────────────────────
    required_fields = {"entity_id", "name", "link", "keys", "job_levels", "description"}
    bad = []
    for i, item in enumerate(data):
        missing = required_fields - set(item.keys())
        if missing:
            bad.append((i, item.get("name", "?"), missing))

    if bad:
        print(f"\nWARNING: {len(bad)} items missing required fields:")
        for i, name, missing in bad[:5]:
            print(f"  [{i}] {name}: missing {missing}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  SHL Catalog Analysis")
    print(f"{'='*50}")
    print(f"  Total assessments : {len(data)}")

    key_counts = Counter(k for item in data for k in item.get("keys", []))
    print(f"\n  Test types (keys):")
    for k, v in key_counts.most_common():
        print(f"    {k:<40} {v}")

    level_counts = Counter(l for item in data for l in item.get("job_levels", []))
    print(f"\n  Job levels:")
    for l, v in level_counts.most_common():
        print(f"    {l:<40} {v}")

    no_desc  = sum(1 for item in data if not item.get("description", "").strip())
    no_level = sum(1 for item in data if not item.get("job_levels"))
    print(f"\n  Quality: {no_desc} missing descriptions, {no_level} missing job levels")
    print(f"\n  catalog.json is valid ✓")
    print(f"  Next step: python catalog_index.py --build")
    print(f"{'='*50}")

    return data


if __name__ == "__main__":
    validate_and_analyze()