import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from time import time

print("Loading data...")

documents = pd.read_csv("data/documents.csv")
queries = pd.read_csv("data/queries.csv")

df = pd.concat([documents, queries["query"]], axis=1)
df = df.dropna(subset=["document", "query"])

docs = df["document"].astype(str).tolist()
query_texts = df["query"].astype(str).tolist()

print("Building Sparse TF-IDF index...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=100000,
    ngram_range=(1,2),
    min_df=2
)

doc_tfidf = vectorizer.fit_transform(docs)

print("Building Dense index...")

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(docs, show_progress_bar=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

print("Hybrid system ready.")

def normalize(scores):
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

def hybrid_search(query, alpha=0.5, top_k=10):
    
    # Sparse scores
    query_vec = vectorizer.transform([query])
    sparse_scores = cosine_similarity(query_vec, doc_tfidf)[0]
    
    # Dense scores
    query_embedding = model.encode([query])[0]
    
    dense_scores = np.dot(doc_embeddings, query_embedding)
    
    # Normalize both
    sparse_scores = normalize(sparse_scores)
    dense_scores = normalize(dense_scores)
    
    # Combine
    combined_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
    
    top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    
    return top_indices

def evaluate(alpha, k=5, max_queries=1000):
    correct = 0
    start = time()
    
    for i, query in enumerate(query_texts[:max_queries]):
        results = hybrid_search(query, alpha=alpha, top_k=k)
        if i in results:
            correct += 1
    
    end = time()
    
    print(f"\nAlpha: {alpha}")
    print(f"Recall@{k}: {correct / max_queries:.4f}")
    print(f"Avg Query Time: {(end-start)/max_queries:.6f} sec")


def compute_mrr(alpha=0.8, max_queries=1000, top_k=50):
    total = 0
    
    for i, query in enumerate(query_texts[:max_queries]):
        results = hybrid_search(query, alpha=alpha, top_k=top_k)
        
        if i in results:
            rank = list(results).index(i) + 1
            total += 1 / rank
    
    print(f"\nAlpha: {alpha}")
    print(f"MRR: {total / max_queries:.4f}")

    print("\n--- MRR Evaluation ---")

# Pure Sparse
compute_mrr(alpha=0.0)

# Pure Dense
compute_mrr(alpha=1.0)

# Best Hybrid
compute_mrr(alpha=0.8)