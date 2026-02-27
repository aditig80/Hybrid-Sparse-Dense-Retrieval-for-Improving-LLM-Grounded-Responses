# 🔍 Sparse vs Dense vs Hybrid Retrieval on StackOverflow QA

> Investigating retrieval effectiveness across TF-IDF, MiniLM embeddings, and hybrid methods on a 20K-document StackOverflow corpus.

---

## 📌 Overview

This project benchmarks three retrieval paradigms on a large-scale StackOverflow Question-Answer corpus:

| Method | Approach |
|--------|----------|
| **Sparse** | TF-IDF with bigrams + cosine similarity |
| **Dense** | SentenceTransformer embeddings + FAISS |
| **Hybrid** | Weighted combination of sparse & dense scores |

Retrieval quality is measured using **Recall@K** and **Mean Reciprocal Rank (MRR)** across 1,000 queries.

---

## 🎯 Research Question

> *Does hybrid sparse-dense retrieval improve ranking performance over standalone dense retrieval in domain-specific QA systems?*

---

## 📊 Dataset

- **~19,965 documents** — StackOverflow Q&A pairs
- **Ground truth** — Highest-scored answer per question
- **Evaluation set** — 1,000 queries

---

## 🧠 Methods

### Sparse Retrieval
- TF-IDF vectorizer with **bigrams**
- Cosine similarity for ranking

### Dense Retrieval
- Model: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (SentenceTransformers)
- Index: **FAISS L2**

### Hybrid Retrieval

Scores are combined via a weighted sum:

$$S = \alpha \cdot S_d + (1 - \alpha) \cdot S_s$$

Where:
- $S_d$ = dense retrieval score
- $S_s$ = sparse retrieval score
- $\alpha \in [0.0, 1.0]$ — swept to find optimal weight

---

## 📈 Results

| Method | Recall@5 | Recall@10 | MRR |
|---|---|---|---|
| Sparse | 0.394 | — | 0.292 |
| Dense | 0.779 | 0.840 | **0.670** |
| **Hybrid (α=0.8)** | **0.789** | **0.843** | 0.661 |

---

## 🔎 Key Findings

1. **Dense retrieval significantly outperforms sparse** — nearly 2× improvement in Recall@5 (0.394 → 0.779).
2. **Hybrid retrieval marginally improves Recall@5** (+1%) at optimal α=0.8, suggesting dense embeddings already capture most semantic signal.
3. **Dense achieves highest MRR (0.670)**, indicating stronger top-ranked result precision.
4. The optimal α=0.8 weighting (80% dense, 20% sparse) confirms dense features dominate in this domain.

---

## 🗂️ Project Structure

```
.
├── data/
│   └── stackoverflow_qa.jsonl      # Raw Q&A corpus
├── embeddings/
│   └── faiss_index.bin             # Precomputed FAISS index
├── retrieval/
│   ├── sparse.py                   # TF-IDF retrieval
│   ├── dense.py                    # MiniLM + FAISS retrieval
│   └── hybrid.py                   # Hybrid scoring
├── evaluation/
│   ├── metrics.py                  # Recall@K, MRR computation
│   └── alpha_sweep.py              # Alpha hyperparameter search
├── notebooks/
│   └── analysis.ipynb              # Results & visualizations
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run retrieval experiments

```bash
# Sparse baseline
python retrieval/sparse.py

# Dense baseline
python retrieval/dense.py

# Hybrid with alpha sweep
python evaluation/alpha_sweep.py --alpha_range 0.0 1.0 0.1
```

### Evaluate

```bash
python evaluation/metrics.py --method hybrid --alpha 0.8
```

---

## 📦 Requirements

```
sentence-transformers
faiss-cpu
scikit-learn
numpy
pandas
```

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{hybrid-retrieval-stackoverflow,
  title   = {Sparse vs Dense vs Hybrid Retrieval on StackOverflow QA},
  year    = {2025},
  note    = {Evaluated on ~20K StackOverflow Q\&A pairs using TF-IDF, MiniLM, and hybrid scoring}
}
```

---

## 📄 License

MIT License
