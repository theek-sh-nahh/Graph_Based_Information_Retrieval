import os
import re
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

STOPWORDS = set(stopwords.words("english"))

def add_document_to_corpus(text, path="data"):
    os.makedirs(path, exist_ok=True)
    doc_id = len([f for f in os.listdir(path) if f.endswith(".txt")]) + 1
    filename = f"doc{doc_id}.txt"

    with open(os.path.join(path, filename), "w", encoding="utf-8") as f:
        f.write(text)

    return filename

def preprocess(text):
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return [t for t in tokens if t not in STOPWORDS]

def load_documents(path="data"):
    docs = {}
    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                docs[file] = preprocess(f.read())
    return docs

def build_graph(documents):
    corpus = [" ".join(tokens) for tokens in documents.values()]
    doc_names = list(documents.keys())

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()

    G = nx.Graph()

    for doc in doc_names:
        G.add_node(doc, type="document")

    for term in terms:
        G.add_node(term, type="term")

    for i, doc in enumerate(doc_names):
        for j, term in enumerate(terms):
            weight = tfidf[i, j]
            if weight > 0:
                G.add_edge(doc, term, weight=float(weight))

    return G, doc_names

def search(query, top_k=5):
    documents = load_documents()
    G, doc_names = build_graph(documents)

    query_terms = preprocess(query)
    relevance = {node: 0.0 for node in G.nodes()}

    for term in query_terms:
        if term in G:
            relevance[term] = 1.0

    alpha = 0.85
    epsilon = 1e-6
    max_iter = 100

    degree_weights = {}
    for node in G.nodes():
        total_weight = sum(data["weight"] for _, data in G[node].items())
        degree_weights[node] = total_weight if total_weight > 0 else 1.0

    for _ in range(max_iter):
        new_rel = {node: (1 - alpha) * relevance.get(node, 0.0) for node in G.nodes()}

        for node in G.nodes():
            for nbr, data in G[node].items():
                normalized_weight = data["weight"] / degree_weights[node]
                new_rel[nbr] += alpha * relevance[node] * normalized_weight

        diff = sum(abs(new_rel[n] - relevance[n]) for n in relevance)
        relevance = new_rel

        if diff < epsilon:
            break

    scores = {d: relevance[d] for d in doc_names}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[:top_k]

def build_query_subgraph(G, query_terms, doc_names, top_docs, max_terms=5):

    nodes = set()
    for term in query_terms:
        if term in G:
            nodes.add(term)
    nodes.update(top_docs)
    for doc in top_docs:
        if doc in G:
            neighbors = sorted(
                [(nbr, data["weight"]) for nbr, data in G[doc].items()],
                key=lambda x: x[1],
                reverse=True
            )

            for term, _ in neighbors[:max_terms]:
                nodes.add(term)

    return G.subgraph(nodes)

def plot_query_graph(subG, query_terms, doc_names):
    pos = nx.spring_layout(subG, seed=42)

    node_colors = []
    for node in subG.nodes():
        if node in query_terms:
            node_colors.append("red")         
        elif node in doc_names:
            node_colors.append("skyblue")      
        else:
            node_colors.append("lightgreen")   

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=800)
    nx.draw_networkx_edges(subG, pos, alpha=0.5)
    nx.draw_networkx_labels(subG, pos, font_size=8)
    plt.title("Query Activation on Term–Document Graph")
    plt.axis("off")
    return plt
