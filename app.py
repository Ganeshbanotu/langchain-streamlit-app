import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Multi-File RAG Chat", layout="wide")
st.title("📄 Ask Questions from Your Documents")

openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
uploaded_files = st.sidebar.file_uploader("📁 Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files and openai_api_key:
    documents = []
    file_map = {}

    # Load and parse each file
    for file in uploaded_files:
        file_name = file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if file_name.endswith(".pdf"):
            loader = PyMuPDFLoader(tmp_path)
        elif file_name.endswith(".txt"):
            loader = TextLoader(tmp_path)
        else:
            continue

        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = file_name
        documents.extend(loaded_docs)
        file_map[file_name] = loaded_docs

    # Split documents into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks = splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(all_chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Setup QA chain
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # User input
    selected_file = st.selectbox("🔍 Focus your question on a specific file (optional)", ["All Files"] + list(file_map.keys()))
    user_question = st.text_input("💬 Ask a question:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_question:
        if selected_file == "All Files":
            results = qa(user_question)
        else:
            selected_chunks = splitter.split_documents(file_map[selected_file])
            local_db = FAISS.from_documents(selected_chunks, embeddings)
            local_qa = RetrievalQA.from_chain_type(llm=llm, retriever=local_db.as_retriever(), return_source_documents=True)
            results = local_qa(user_question)

        answer = results["result"]
        sources = list({doc.metadata.get("source", "Unknown") for doc in results["source_documents"]})

        st.session_state.chat_history.append((user_question, answer, sources))

    # Display chat history
    for i, (q, a, s) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}: {q}**")
        st.markdown(f"🧠 {a}")
        st.markdown(f"📚 Sources: {', '.join(s)}")
        st.markdown("---")
else:
    st.info("⬅️ Upload PDF or TXT files and enter your OpenAI API Key to get started.")
