from pathlib import Path
import csv
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "dataset" / "documents"
DB_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "ai_education_documents"
MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids, texts, metadatas = [], [], []
    for file in sorted(DOCS.glob("*.txt")):
        text = file.read_text(encoding="utf-8").strip()
        title = text.splitlines()[0]
        ids.append(file.stem)
        texts.append(text)
        metadatas.append({"title": title, "category": "Artificial Intelligence", "source": file.name})

    vectors = model.encode(texts, normalize_embeddings=True)
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors.tolist())
    np.save(ROOT / "dataset" / "embeddings.npy", vectors)

    query_file = ROOT / "dataset" / "queries.csv"
    results = []
    with query_file.open(encoding="utf-8-sig", newline="") as f:
        queries = list(csv.DictReader(f))
    for q in queries:
        qv = model.encode([q["query"]], normalize_embeddings=True)[0]
        found = collection.query(query_embeddings=[qv.tolist()], n_results=5)
        for rank, (doc, meta, distance) in enumerate(zip(
            found["documents"][0], found["metadatas"][0], found["distances"][0]
        ), start=1):
            results.append({
                "query_id": q["query_id"], "query": q["query"], "rank": rank,
                "document_title": meta.get("title", ""), "source": meta.get("source", ""),
                "cosine_distance": round(float(distance), 4), "retrieved_text": doc
            })

    out = ROOT / "dataset" / "retrieved_results.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Stored {len(texts)} documents in ChromaDB: {DB_DIR}")
    print(f"Saved embeddings: {ROOT / 'dataset' / 'embeddings.npy'}")
    print(f"Saved Top-5 results for {len(queries)} queries: {out}")
    print("\nSample results:")
    for row in results[:5]:
        print(f'{row["rank"]}. {row["document_title"]} | distance={row["cosine_distance"]}')

if __name__ == "__main__":
    main()
