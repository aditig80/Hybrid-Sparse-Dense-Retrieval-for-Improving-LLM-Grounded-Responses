import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from time import time

print("Loading data...")

documents = pd.read_csv("data/documents.csv")
queries = pd.read_csv("data/queries.csv")

# Keep alignment safe
df = pd.concat([documents, queries["query"]], axis=1)
df = df.dropna(subset=["document", "query"])

docs = df["document"].astype(str).tolist()
query_texts = df["query"].astype(str).tolist()

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding documents... (this may take time)")

doc_embeddings = model.encode(docs, show_progress_bar=True)

dimension = doc_embeddings.shape[1]

print("Building FAISS index...")

index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

print("Index ready.")

def search(query, top_k=10):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return indices[0]

def recall_at_k(k=5, max_queries=1000):
    correct = 0
    start = time()

    for i, query in enumerate(query_texts[:max_queries]):
        results = search(query, top_k=k)
        if i in results:
            correct += 1

    end = time()

    print(f"Tested on {max_queries} queries")
    print(f"Recall@{k}: {correct / max_queries:.4f}")
    print(f"Avg Query Time: {(end-start)/max_queries:.6f} sec")

recall_at_k(5, 1000)
recall_at_k(10, 1000)