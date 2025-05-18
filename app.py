import streamlit as st 
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import os

# Extract text from uploaded files
def extract_text(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        import docx
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

# Streamlit UI
st.set_page_config(page_title="RAG Chat App", layout="wide")
st.title("🧠 Chat with Your Documents (Local RAG, No API Key)")

uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    docs = []
    for file in uploaded_files:
        text = extract_text(file)
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": file.name}))
        else:
            st.warning(f"Could not extract text from {file.name}")

    if docs:
        st.info("Generating embeddings locally (first time may take a few seconds)...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        selected_file = st.selectbox("Select a document to chat with:", [doc.metadata["source"] for doc in docs])

        retriever = vectorstore.as_retriever(search_kwargs={"filter": {"source": selected_file}})

        st.info("Loading tiny local model for response generation...")
        pipe = pipeline("text-generation", model="sshleifer/tiny-gpt2", max_new_tokens=100)
        llm = HuggingFacePipeline(pipeline=pipe)

        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        question = st.text_input("Ask a question about the document:")

        if question:
            answer = qa.run(question)
            st.session_state.chat_history.append((question, answer))

        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")
            st.markdown("---")

    else:
        st.error("No valid text found in any of the uploaded files.")
else:
    st.info("Please upload at least one document to get started.")
