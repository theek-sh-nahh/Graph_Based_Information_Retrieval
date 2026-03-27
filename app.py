import streamlit as st
from ir_core import (
    search,
    load_documents,
    build_graph,
    preprocess,
    build_query_subgraph,
    plot_query_graph,
    add_document_to_corpus
)

st.set_page_config(
    page_title="Graph-Based Text Information Retrieval",
    layout="wide"
)

def clear_new_doc():
    st.session_state.new_doc_text = ""

def handle_add_doc():
    text = st.session_state.new_doc_text.strip()
    if not text:
        st.warning("Document content cannot be empty")
        return

    filename = add_document_to_corpus(text)
    st.success(f"{filename} added successfully!")
    clear_new_doc()

with st.sidebar:
    st.markdown("## ➕ Document Management")
    st.markdown("---")

    if "new_doc_text" not in st.session_state:
        st.session_state.new_doc_text = ""

    st.text_area(
        "Paste document content",
        key="new_doc_text",
        height=250
    )

    st.button(
        "Add to Corpus",
        use_container_width=True,
        on_click=handle_add_doc
    )

    st.markdown("---")
    st.caption("Documents added here are immediately available for search.")

st.title("🔎 Graph-Based Text Information Retrieval")

query = st.text_input("Enter your search query")
if query:
    documents = load_documents()
    results = search(query)

    has_results = any(score > 0 for _, score in results)

    if not has_results:
        st.markdown("---")
        st.warning("❌ Can't find what you're looking for.")
        st.caption("Try using different keywords or add relevant documents to the corpus.")

    else:
        view_mode = st.radio(
            "View Mode",
            ["Search Results", "Relevance Ranking", "Graph Visualization"],
            horizontal=True
        )

        st.markdown("---")

        if view_mode == "Search Results":
            st.subheader("📄 Retrieved Documents")

            for doc, _ in results:
                with open(f"data/{doc}", "r", encoding="utf-8") as f:
                    content = f.read()
                    snippet = content[:300] + "..." if len(content) > 300 else content

                st.markdown(f"### {doc}")
                st.write(snippet)
                st.markdown("---")

        elif view_mode == "Relevance Ranking":
            st.subheader("📊 Relevance-Based Ranking")

            for rank, (doc, score) in enumerate(results, start=1):
                st.write(f"{rank}. **{doc}** — {round(score, 4)}")

        elif view_mode == "Graph Visualization":
            st.subheader("🕸️ Query Activation on Term–Document Graph")

            G, doc_names = build_graph(documents)
            query_terms = preprocess(query)
            top_docs = [doc for doc, _ in results]
            subG = build_query_subgraph(G, query_terms, doc_names, top_docs)


            fig = plot_query_graph(subG, query_terms, doc_names)
            st.pyplot(fig)

