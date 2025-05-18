import streamlit as st
from PyPDF2 import PdfReader
import docx
import os
from transformers import pipeline, GPT2Tokenizer

# --- Setup Streamlit ---
st.set_page_config(page_title="RAG Chat App", layout="wide")
st.title("📄 Chat with Your Documents (RAG App)")

# --- File uploader ---
uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Limit 200MB per file • PDF, DOCX, TXT"
)

# --- Extract text from uploaded documents ---
def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.name.endswith(".txt"):
        text += file.read().decode("utf-8")
    return text

document_texts = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        text = extract_text(uploaded_file)
        document_texts.append(text)
    st.success("✅ Documents processed and indexed!")

# --- Prompt Input ---
query = st.text_input("Ask a question about the documents:")

# --- Answer generation ---
if query and document_texts:
    full_context = "\n".join(document_texts)
    prompt = f"Context: {full_context}\n\nQuestion: {query}\nAnswer:"

    # --- Load model and tokenizer ---
    generator = pipeline("text-generation", model="gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # --- Truncate prompt if too long ---
    max_input_tokens = 1024
    inputs = tokenizer(prompt, truncation=True, max_length=max_input_tokens, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # --- Generate output ---
    outputs = generator.model.generate(
        input_ids=input_ids,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- Display output ---
    st.markdown("### 💬 Response:")
    st.write(response)
