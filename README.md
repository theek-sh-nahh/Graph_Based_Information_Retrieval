# 🔎 Graph-Based Text Information Retrieval

A graph-based document retrieval system that models the relationship between terms and documents as a weighted bipartite graph, then uses relevance propagation (inspired by PageRank) to rank documents in response to a user query. Built with a Streamlit interface for interactive search and visualization.

---

## 📌 How It Works

Traditional IR systems like TF-IDF rank documents by direct term overlap with a query. This system goes further — it builds a **Term–Document Graph** where every term and document is a node, connected by TF-IDF-weighted edges. When a query is submitted, the matching term nodes are activated and relevance **propagates** through the graph to surface semantically connected documents, even those without an exact term match.

### Pipeline Overview

```
Raw Documents (.txt)
        │
        ▼
  Text Preprocessing        → Lowercasing, tokenization, stopword removal (NLTK)
        │
        ▼
  TF-IDF Computation        → sklearn TfidfVectorizer over the corpus
        │
        ▼
  Graph Construction        → Bipartite graph: term nodes ↔ document nodes (weighted by TF-IDF)
        │
        ▼
  Query Processing          → Preprocess query → activate matching term nodes (relevance = 1.0)
        │
        ▼
  Relevance Propagation     → Iterative random-walk with damping factor α = 0.85
        │
        ▼
  Ranked Results            → Top-K documents by propagated relevance score
```

---

## ⚙️ Core Components

### `ir_core.py`

| Function | Description |
|---|---|
| `preprocess(text)` | Lowercases, tokenizes, and removes NLTK stopwords |
| `load_documents(path)` | Reads all `.txt` files from `data/` and preprocesses them |
| `build_graph(documents)` | Constructs a weighted bipartite TF-IDF term–document graph using NetworkX |
| `search(query, top_k=5)` | Runs the full retrieval pipeline and returns top-K ranked documents |
| `build_query_subgraph(...)` | Extracts a focused subgraph around query terms and top results |
| `plot_query_graph(...)` | Renders the query activation graph using Matplotlib |
| `add_document_to_corpus(text)` | Saves a new `.txt` document to `data/` and makes it immediately searchable |

### `app.py`

A Streamlit app with three view modes:

- **Search Results** — Shows document snippets for top results
- **Relevance Ranking** — Displays documents ranked by propagation score
- **Graph Visualization** — Renders the query-activated subgraph interactively

Users can also add new documents to the corpus live via the sidebar.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/graph-based-text-ir.git
cd graph-based-text-ir
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the corpus (if `data/` is empty)

Open and run `generate_docs.ipynb`, or run:

```bash
python -c "
import os, random
os.makedirs('data', exist_ok=True)
# ... (see generate_docs.ipynb for the full script)
"
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web application interface |
| `networkx` | Graph construction and traversal |
| `scikit-learn` | TF-IDF vectorization |
| `nltk` | Stopword corpus |
| `matplotlib` | Graph visualization |

> **Note:** On first run, NLTK stopwords must be downloaded:
> ```python
> import nltk
> nltk.download('stopwords')
> ```

---

## 🧠 Relevance Propagation Algorithm

The search function implements a normalized random-walk with restart, similar in spirit to PageRank:

1. **Initialization** — Query terms found in the graph are assigned relevance `1.0`; all other nodes start at `0.0`
2. **Normalization** — Each node's outgoing edge weights are normalized by total degree weight
3. **Iteration** — At each step, relevance spreads from every node to its neighbors proportionally to edge weight, with a damping factor `α = 0.85`
4. **Convergence** — Iteration stops when the total change in relevance falls below `ε = 1e-6` (max 100 iterations)
5. **Ranking** — Document nodes are sorted by their final relevance scores

---

## 📓 Notebooks

- **`Graph_Based_Text_IR.ipynb`** — A pedagogical walkthrough of the entire pipeline: preprocessing → TF-IDF → graph construction → query activation → relevance propagation → result ranking. Useful for understanding the theory.
- **`generate_docs.ipynb`** — Generates a synthetic 100-document corpus covering topics in *Information Retrieval*, *Machine Learning*, and *Data Analytics*.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
