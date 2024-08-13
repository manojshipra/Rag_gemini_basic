import streamlit as st
import google.generativeai as genai
import pandas as pd
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="AI Document Assistant", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for improved appearance
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; padding: 0.75rem 0; }
    .upload-section, .query-section, .history-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .query-section { background-color: #e6f3ff; }
    .history-section { background-color: #f6f6f6; }
    .stExpander { border: none; }
    .stTextInput>div>input { border-radius: 5px; padding: 0.75rem; }
</style>
""", unsafe_allow_html=True)

# Configure Google Generative AI API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    return "".join(page.extract_text() or "" for page in PdfReader(pdf_file).pages)

# Function to extract text from CSV
def extract_text_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df.to_string(index=False)

# Initialize session state
if 'queries_responses' not in st.session_state:
    st.session_state.queries_responses = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False

# Function to process uploaded document
def process_document(file):
    with st.spinner("Processing document..."):
        file_text = extract_text_from_pdf(file) if file.type == "application/pdf" else extract_text_from_csv(file)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = [Document(page_content=doc) for doc in text_splitter.split_text(file_text)]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(documents, embeddings)
        st.session_state.file_processed = True
    st.success("Document processed successfully!")

# Function to generate response
def generate_response(query):
    if not st.session_state.vectorstore:
        st.warning("Please upload and process a document first.")
        return None

    retrieved_docs = st.session_state.vectorstore.similarity_search(query)
    context = " ".join(doc.page_content for doc in retrieved_docs)
    prompt = f"Based on the following context, please answer the question:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = model.generate_content(prompt).text
    st.session_state.queries_responses.append({"query": query, "response": response})
    return response

# Main layout
st.title("🤖 AI Document Assistant")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or CSV file", type=["pdf", "csv"])
    if uploaded_file and not st.session_state.file_processed:
        process_document(uploaded_file)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.header("💬 Ask Questions")
    with st.container():
        st.markdown('<div class="query-section">', unsafe_allow_html=True)
        query = st.text_input("Enter your question:", placeholder="What would you like to know about the document?")
        if st.button("Submit Question"):
            if st.session_state.vectorstore:
                with st.spinner("Generating answer..."):
                    response = generate_response(query)
                    if response:
                        st.success("Answer generated!")
                        st.write("**Answer:**", response)
            else:
                st.warning("Please upload a document first.")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.header("📜 Question History")
    with st.container():
        st.markdown('<div class="history-section">', unsafe_allow_html=True)
        if st.session_state.queries_responses:
            for i, item in enumerate(reversed(st.session_state.queries_responses)):
                with st.expander(f"Q{len(st.session_state.queries_responses)-i}: {item['query'][:50]}..."):
                    st.write("**Full Question:**", item['query'])
                    st.write("**Answer:**", item['response'])
        else:
            st.info("No questions asked yet.")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your AI Assistant Team")
