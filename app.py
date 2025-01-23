import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import google.generativeai as genai
import pandas as pd
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv
import tempfile

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
    .context-link { color: #0066cc; cursor: pointer; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# Configure Google Generative AI API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Function to extract text from PDF with page tracking
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text_with_pages = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text:
            text_with_pages.append((text, page_num))
    return text_with_pages

# Initialize session state
if 'queries_responses' not in st.session_state:
    st.session_state.queries_responses = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'document_content' not in st.session_state:
    st.session_state.document_content = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'total_pages' not in st.session_state:
    st.session_state.total_pages = 0
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

# Function to process uploaded document
def process_document(file):
    with st.spinner("Processing document..."):
        file_type = file.type
        st.session_state.file_type = file_type

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        st.session_state.uploaded_file = temp_file_path

        if file_type == "application/pdf":
            with open(temp_file_path, 'rb') as pdf_file:
                text_with_pages = extract_text_from_pdf(pdf_file)
            st.session_state.document_content = text_with_pages
            st.session_state.total_pages = len(text_with_pages)
            documents = []
            for text, page_num in text_with_pages:
                documents.extend([Document(page_content=chunk, metadata={"page": page_num})
                                  for chunk in CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)])
        elif file_type == "text/csv":
            df = pd.read_csv(temp_file_path)
            st.session_state.document_content = df
            documents = [Document(page_content=chunk) for chunk in CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(df.to_string())]
        else:
            st.error("Unsupported file type. Please upload a PDF or CSV file.")
            return

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(documents, embeddings)
        st.session_state.file_processed = True

    st.success("Document processed successfully!")

# Function to generate response with context tracking
def generate_response(query):
    if not st.session_state.vectorstore:
        st.warning("Please upload and process a document first.")
        return None

    retrieved_docs = st.session_state.vectorstore.similarity_search(query)
    context = ""
    context_sources = []
    for doc in retrieved_docs:
        context += doc.page_content + " "
        context_sources.append({"content": doc.page_content, "page": doc.metadata.get("page", "N/A")})

    prompt = f"""Based on the following context, please answer the question:
Context: {context}
Question: {query}
Please provide your answer in the same language as the question. If the context is in a different language, translate the relevant information as needed.
Answer:"""
    response = model.generate_content(prompt).text
    st.session_state.queries_responses.append({
        "query": query,
        "response": response,
        "context_sources": context_sources
    })
    return response, context_sources

# Function to display context and navigate to source
def display_context(context_sources):
    for i, source in enumerate(context_sources):
        page_num = source['page']
        context = source['content']
        st.markdown(f"""
        <details>
            <summary>Context {i+1} (Page {page_num})</summary>
            <p><span class="context-link" data-page="{page_num}">{context}</span></p>
        </details>
        """, unsafe_allow_html=True)

# Function to navigate to a specific page in the PDF
def navigate_to_page(page_num):
    st.session_state.current_page = page_num
    st.experimental_rerun()

# Main layout
st.title("🤖 AI Document Assistant")

# File upload in the main area
st.subheader("📁 Upload Document")
uploaded_file = st.file_uploader("Choose a PDF or CSV file", type=["pdf", "csv"])
if uploaded_file and not st.session_state.file_processed:
    process_document(uploaded_file)

# Main content area
if st.session_state.file_processed:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        with st.container():
            st.markdown('<div class="query-section">', unsafe_allow_html=True)
            query = st.text_input("Enter your question:", placeholder="What would you like to know about the document?")
            if st.button("Submit Question"):
                with st.spinner("Generating answer..."):
                    response, context_sources = generate_response(query)
                    if response:
                        st.success("Answer generated!")
                        st.write("**Answer:**", response)
                        st.subheader("Context Sources:")
                        display_context(context_sources)
            st.markdown('</div>', unsafe_allow_html=True)

        st.header("📜 Question History")
        with st.container():
            st.markdown('<div class="history-section">', unsafe_allow_html=True)
            if st.session_state.queries_responses:
                for i, item in enumerate(reversed(st.session_state.queries_responses)):
                    with st.expander(f"Q{len(st.session_state.queries_responses)-i}: {item['query'][:50]}..."):
                        st.write("**Full Question:**", item['query'])
                        st.write("**Answer:**", item['response'])
                        st.subheader("Context Sources:")
                        display_context(item['context_sources'])
            else:
                st.info("No questions asked yet.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.header("Document Viewer")
        if st.session_state.uploaded_file:
            if st.session_state.file_type == "application/pdf":
                with open(st.session_state.uploaded_file, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                    pdf_viewer(pdf_bytes, height=600)
                
                # Navigation buttons
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("Previous"):
                        if st.session_state.current_page > 1:
                            st.session_state.current_page -= 1
                            st.experimental_rerun()
                with col2:
                    st.write(f"Page {st.session_state.current_page} of {st.session_state.total_pages}")
                with col3:
                    if st.button("Next"):
                        if st.session_state.current_page < st.session_state.total_pages:
                            st.session_state.current_page += 1
                            st.experimental_rerun()
                
            elif st.session_state.file_type == "text/csv":
                st.dataframe(st.session_state.document_content)
        else:
            st.info("No document loaded. Please upload a document.")

# JavaScript code to handle context link clicks
st.markdown("""
<script>
document.addEventListener('click', function(event) {
    if (event.target.classList.contains('context-link')) {
        var pageNum = event.target.getAttribute('data-page');
        window.parent.postMessage({type: 'navigation', page: pageNum}, '*');
    }
});

window.addEventListener('message', function(event) {
    if (event.data.type === 'navigation') {
        var pageNum = parseInt(event.data.page);
        window.scrollTo(0, 0);
        var contextElement = document.querySelector(`[data-page="${pageNum}"]`);
        if (contextElement) {
            contextElement.style.backgroundColor = 'yellow';
            contextElement.scrollIntoView({behavior: 'smooth', block: 'center'});
        }
    }
});
</script>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your AI Assistant Team")
