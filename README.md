# AI Document Assistant 🤖

## Overview
AI Document Assistant is an interactive Streamlit application that allows users to upload PDF or CSV files and ask questions about their contents using advanced AI-powered retrieval and generation.

## Features
- PDF and CSV document support
- AI-powered document question answering
- Context-aware response generation
- Document viewer with page navigation
- Query history tracking
- Intuitive user interface

## Prerequisites
- Python 3.8+
- Google API Key for Gemini AI
- Internet connection

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-document-assistant.git
cd ai-document-assistant
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root and add:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Dependencies
- Streamlit
- Google GenerativeAI
- Langchain
- HuggingFace Embeddings
- PyPDF2
- pandas

## Running the Application
```bash
streamlit run app.py
```

## How to Use
1. Upload a PDF or CSV file
2. Wait for document processing
3. Ask questions about the document
4. View answers with contextual sources
5. Navigate through document and query history

## Configuration
- Customize embedding model in `process_document()` function
- Modify Gemini AI model in the configuration section
- Adjust text splitting parameters as needed

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request
