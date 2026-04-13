import streamlit as st
import logging
import os
import json
import numpy as np
import faiss
from PyPDF2 import PdfReader
import docx2txt
import re
import anthropic
from sentence_transformers import SentenceTransformer

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
DATA_DIR = "./data"  # Persistent data directory
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "document_embeddings.faiss")
CHAT_HISTORY_FILE = os.path.join(DATA_DIR, "chat_history.json")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize the local embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Initialize Anthropic client (deferred — only needed when answering questions)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def get_anthropic_client():
    if not ANTHROPIC_API_KEY:
        st.error("ANTHROPIC_API_KEY environment variable is not set. Please set it to use the chat feature.")
        st.stop()
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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

def chunk_text(text, max_length=200):  # Changed max_length to 200
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
            # Include filename and chunk number in the chunk text
            chunk_with_metadata = f"[{filename} - Chunk {i}] {chunk}"
            documents.append({
                'text': chunk_with_metadata,
                'metadata': {
                    'filename': filename,
                    'chunk_number': i
                }
            })
    return documents

def generate_embeddings(documents):
    texts = [doc['text'] for doc in documents]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    return np.array(embeddings).astype('float32')

def save_embeddings(embeddings):
    st.write("Saving embeddings... Please wait.")  # Inform the user
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
        st.write("Generating embeddings... Please wait.")  # Inform the user
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

# Initialize context if it doesn't exist
if "context" not in st.session_state:
    st.session_state.context = ""

# --- Chat History Persistence ---
def load_chat_history():
    """Load chat history from a JSON file on disk."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Failed to load chat history: {e}")
    return None

def save_chat_history(messages):
    """Save chat history to a JSON file on disk."""
    try:
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump(messages, f, indent=2)
    except IOError as e:
        logging.error(f"Failed to save chat history: {e}")

# Chat Interaction
if "messages" not in st.session_state:
    saved_messages = load_chat_history()
    if saved_messages:
        st.session_state.messages = saved_messages
    else:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about medical topics based on the stored documents!"}
        ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.messages)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Generate embedding for the user's query using local model
            query_embedding = embedding_model.encode([prompt]).astype('float32')

            # Find the closest document embeddings
            distances, indices = st.session_state.embeddings_index.search(query_embedding, k=2)  # Top 2 results

            # Retrieve the corresponding documents for the top 2 indices
            relevant_docs = []
            for idx in indices[0]:
                if idx < len(st.session_state.documents):
                    relevant_docs.append(st.session_state.documents[idx]['text'])

            # Update the context with the relevant documents
            st.session_state.context = " ".join(relevant_docs)

            # Improved prompt for the LLM
            improved_prompt = f"Based on the following context, answer the question as accurately as possible. Context: {st.session_state.context}\n\nQuestion: {prompt}"

            # Send the context to Anthropic API for response generation
            anthropic_client = get_anthropic_client()
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": improved_prompt}
                ]
            )
            answer = response.content[0].text

            # Format the response to include relevant documents
            relevant_docs_str = "\n".join([f"- {doc}" for doc in relevant_docs])  # Bullet points for relevant docs
            full_response = f"{answer}\n\n---\n\n### Source:\n{relevant_docs_str}"


            # Display the formatted answer
            st.write(full_response)
            logging.info(f"question: {prompt}, response: {full_response}")
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)
            save_chat_history(st.session_state.messages)