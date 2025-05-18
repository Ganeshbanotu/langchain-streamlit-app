import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
import tempfile

st.set_page_config(page_title="Chat with Your Documents", layout="wide")
st.title("📄 Chat with Your Documents (RAG App)")
st.subheader("Upload documents (PDF, DOCX, TXT)")

uploaded_files = st.file_uploader("Upload your files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

documents = []
file_names = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_names.append(file_name)

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file_name.endswith(".txt"):
            loader = TextLoader(tmp_path)
        elif file_name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            st.warning(f"Unsupported file type: {file_name}")
            continue

        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_name
        documents.extend(docs)

    st.success("✅ Documents processed and indexed!")

    # Select which file to chat with
    selected_file = st.selectbox("Select a file to chat with:", file_names)

    # Filter documents based on selection
    selected_docs = [doc for doc in documents if doc.metadata.get("source") == selected_file]

    # Embed selected docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(selected_docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # QA chain
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    # Chat loop
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about the documents:")
    if user_question:
        docs = vectorstore.similarity_search(user_question)
        answer = chain.run(input_documents=docs, question=user_question)

        # Save and show the chat history
        st.session_state.chat_history.append((user_question, answer))
    
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Answer:** {a}")
