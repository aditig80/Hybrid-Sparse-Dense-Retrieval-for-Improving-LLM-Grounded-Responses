import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from time import time

print("Loading data...")

documents = pd.read_csv("data/documents.csv")
queries = pd.read_csv("data/queries.csv")

# Remove NaN safely
documents = documents.dropna(subset=["document"])
queries = queries.dropna(subset=["query"])

# Convert explicitly to string (extra safety)
docs = documents["document"].astype(str).tolist()
query_texts = queries["query"].astype(str).tolist()
print("Building TF-IDF index...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=100000,
    ngram_range=(1,2),
    min_df=2
)
doc_vectors = vectorizer.fit_transform(docs)

print("Index built.")

def search(query, top_k=10):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_vectors)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return top_indices

print("Running evaluation...")

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