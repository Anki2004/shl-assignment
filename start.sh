set -e

if [ ! -f "data/catalog.faiss" ] || [ ! -f "data/catalog_meta.pkl" ]; then
    echo "Building FAISS index..."
    python catalog_index.py --build
fi

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"