import streamlit as st
from PyPDF2 import PdfReader
import docx
st.set_page_config(page_title="RAG Chat App", layout="wide")
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline('text-generation', model='gpt2')
    return embedder, generator

embedder, generator = load_models()


st.title("📄 Chat with Your Documents (RAG App)")

# Upload files
uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

def parse_documents(files):
    texts = []
    for file in files:
        if file.type == "application/pdf":
            reader = PdfReader(file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            texts.append(text)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            text = "\n".join(para.text for para in doc.paragraphs)
            texts.append(text)
        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
            texts.append(text)
    return "\n".join(texts)

if uploaded_files:
    with st.spinner("Processing documents..."):
        full_text = parse_documents(uploaded_files)
        chunks = full_text.split("\n\n")
        embeddings = embedder.encode(chunks, convert_to_tensor=True)
        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embeddings
        st.success("✅ Documents processed and indexed!")

user_question = st.text_input("Ask a question about the documents:")

if user_question and "embeddings" in st.session_state:
    query_embedding = embedder.encode(user_question, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, st.session_state["embeddings"], top_k=3)[0]
    top_chunks = "\n\n".join([st.session_state["chunks"][hit["corpus_id"]] for hit in hits])

    prompt = f"Context:\n{top_chunks}\n\nQuestion: {user_question}\nAnswer:"

    response = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"]
    answer = response.split("Answer:")[-1].strip()

    st.markdown(f"**Answer:** {answer}")
