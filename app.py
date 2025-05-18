import streamlit as st 
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# Text extraction function
def extract_text(uploaded_file):
    file_type = uploaded_file.type

    if file_type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    
    elif file_type == "application/pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        import docx
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    
    return ""

# Streamlit UI
st.set_page_config(page_title="RAG Chat App", layout="wide")
st.title("📄 Chat with Your Documents")

uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    docs = []
    for file in uploaded_files:
        text = extract_text(file)
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": file.name}))
        else:
            st.warning(f"Could not extract text from {file.name}")

    if docs:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        file_names = [doc.metadata["source"] for doc in docs]
        selected_file = st.selectbox("Select a document:", file_names)

        retriever = vectorstore.as_retriever(search_kwargs={"filter": {"source": selected_file}})
        qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=retriever)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        question = st.text_input("Ask something:")

        if question:
            response = qa.run(question)
            st.session_state.chat_history.append((question, response))

        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")
            st.markdown("---")
    else:
        st.error("No extractable text found.")
else:
    st.info("Upload PDF, DOCX, or TXT files to start.")
