import streamlit as st
import os
import tempfile
import PyPDF2
import docx
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Must be first
st.set_page_config(page_title="RAG Chat App", layout="wide")

st.title("📄 Chat with Your Documents (RAG App)")

st.markdown("#### Upload documents (PDF, DOCX, TXT)")
uploaded_files = st.file_uploader("Upload your files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

@st.cache_data(show_spinner=False)
def load_document(file):
    file_type = file.name.split(".")[-1]
    if file_type == "pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file_type == "docx":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file_type == "txt":
        text = file.read().decode("utf-8")
    else:
        text = ""
    return text

@st.cache_resource
def get_generator():
    return pipeline("text-generation", model="gpt2")

generator = get_generator()

# Load and index documents
if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        all_text += load_document(file) + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([all_text])
    chunks = [doc.page_content for doc in docs]
    st.success("✅ Documents processed and indexed!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("#### Ask a question about the documents:")
    user_question = st.text_input("Your question:", key="input_question")

    if user_question:
        # Combine the context for basic RAG-like generation
        context = "\n".join(chunks[:5])  # limit to 5 chunks to reduce input size
        prompt = f"Context:\n{context}\n\nQuestion: {user_question}\nAnswer:"
        response = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"]

        # Extract just the generated answer (optional cleanup)
        answer = response.split("Answer:")[-1].strip()

        # Add to history
        st.session_state.chat_history.append((user_question, answer))

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
            st.markdown(f"**Q{i+1}: {q}**")
            st.markdown(f"🟢 {a}")
            st.markdown("---")
