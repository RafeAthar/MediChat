import streamlit as st
import logging
import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
import docx2txt
import re
import openai

# Streamlit App and Page Configuration
st.set_page_config(
    page_title="Chat with Medical Knowledge", 
    page_icon="📜", 
    layout="centered", 
    initial_sidebar_state="auto"
)
st.title("📜 Chat with Medical Knowledge")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Output to a file
        logging.StreamHandler()          # Output to console
    ]
)

# Define the directory where documents are stored
DOCUMENTS_DIR = "./documents"  # Medical documents folder
EMBEDDINGS_FILE = "document_embeddings.faiss"

# Helper functions to extract text from documents
def process_pdf(pdf_path):
    file = PdfReader(pdf_path)
    text = ""
    for page in file.pages:
        text += str(page.extract_text())
    return text

def extract_text_from_docx_with_tables(file_path):
    try:
        text = docx2txt.process(file_path)
        text = re.sub(r'\n+', '\n', text)  # Replace multiple line breaks with one
        parsed_text = ' '.join(text.split())
        return parsed_text
    except Exception as e:
        logging.error(f'An error occurred: {str(e)}', exc_info=True)
        return str(e)

def chunk_text(text, max_length=500):
    """Chunk text into smaller segments."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunk = ' '.join(words[i:i + max_length])
        chunks.append(chunk)
    return chunks

def load_medical_documents():
    documents = []
    for filename in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        if filename.endswith(".pdf"):
            doc_text = process_pdf(file_path)
        elif filename.endswith(".docx"):
            doc_text = extract_text_from_docx_with_tables(file_path)
        else:
            logging.info(f"{filename} File type not supported!")
            continue
        
        # Chunk the document text
        chunks = chunk_text(doc_text)
        for i, chunk in enumerate(chunks):
            documents.append({
                'text': chunk,
                'metadata': {
                    'filename': filename,
                    'chunk_number': i
                }
            })
    return documents

def generate_embeddings(documents):
    embeddings = []
    for doc in documents:
        embedding_response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=doc['text']
        )
        embeddings.append(embedding_response['data'][0]['embedding'])
    return np.array(embeddings).astype('float32')

def save_embeddings(embeddings):
    st.write("Saving embeddings...")  # Inform the user
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, EMBEDDINGS_FILE)

def load_embeddings():
    """Load the FAISS index from file."""
    if os.path.exists(EMBEDDINGS_FILE):
        return faiss.read_index(EMBEDDINGS_FILE)
    else:
        return None

# Check if the document embeddings already exist
if os.path.exists(EMBEDDINGS_FILE):
    st.session_state.embeddings_index = load_embeddings()
    st.write("Embeddings loaded!")
else:
    if os.path.exists(DOCUMENTS_DIR):
        medical_documents = load_medical_documents()
        document_embeddings = generate_embeddings(medical_documents)
        save_embeddings(document_embeddings)
        st.session_state.embeddings_index = faiss.IndexFlatL2(document_embeddings.shape[1])
        st.write("Document embeddings generated and saved!")
    else:
        st.write("Medical documents folder not found.")
        st.stop()

# Store original documents separately
st.session_state.documents = load_medical_documents()

# Chat Interaction
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about medical topics based on the stored documents!"}
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Generate embedding for the user's query
            query_embedding_response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=prompt
            )
            query_embedding = np.array(query_embedding_response['data'][0]['embedding']).astype('float32').reshape(1, -1)

            # Find the closest document embeddings
            distances, indices = st.session_state.embeddings_index.search(query_embedding, k=8)  # Top 8 results

            # Retrieve the corresponding documents for the top 8 indices
            relevant_docs = []
            for idx in indices[0]:
                if idx < len(st.session_state.documents):
                    relevant_docs.append(st.session_state.documents[idx]['text'])

            context = " ".join(relevant_docs)

            # Send the context to OpenAI API for response generation
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"{context}\n\n{prompt}"}
                ],
                api_key=os.getenv("OPENAI_API_KEY")
            )
            answer = response.choices[0].message['content']
            st.write(answer)
            logging.info(f"question: {prompt}, response: {answer}")
            message = {"role": "assistant", "content": answer}
            st.session_state.messages.append(message)

# Docker commands
# docker build -t medi_chat .
# docker run -p 8503:8501 -e OPENAI_API_KEY="" medi_chat
