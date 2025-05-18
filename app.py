import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
import os
import tempfile

st.set_page_config(page_title="RAG Chat App", layout="wide")

st.title("📄 Chat with Your Documents (RAG App)")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=["pdf", "txt", "docx"], accept_multiple_files=True)

file_docs = []
file_names = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.type == "text/plain":
            loader = TextLoader(tmp_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(tmp_path)
        else:
            continue

        docs = loader.load()
        file_docs.extend(docs)
        file_names.append(uploaded_file.name)

    st.success("✅ Documents processed and indexed!")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(file_docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    generator = pipeline("text-generation", model="gpt2")

    selected_file = st.selectbox("Select a file to ask questions about:", file_names)

    query = st.text_input("Ask a question about the selected file:")

    if query and selected_file:
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs if selected_file in doc.metadata.get("source", "")])

        if not context:
            context = "\n\n".join([doc.page_content for doc in docs])  # fallback

        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        response = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"]
        answer = response.split("Answer:")[-1].strip()

        st.session_state.history.append((query, answer))

    for q, a in st.session_state.history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
