# Testing MediChat Streamlit App

## Overview
MediChat is a Streamlit-based medical Q&A app that uses FAISS for document embeddings and Anthropic Claude for chat responses. Data (chat history + embeddings) persists to a `data/` directory.

## Devin Secrets Needed
- `ANTHROPIC_API_KEY` — Required to test chat response generation. Without it, you can still test startup, embedding loading, and chat history persistence.

## Setup

### Install system dependencies
```bash
sudo apt-get install -y libjpeg-dev zlib1g-dev libpng-dev
```

### Install Python dependencies
```bash
cd /home/ubuntu/repos/MediChat
pip install -r requirements.txt
```

**Known issue:** `faiss-cpu` versions before 1.8.0 may not support Python 3.10+. If installation fails, check the version pin in `requirements.txt`.

### Generate embeddings (first run)
The app auto-generates FAISS embeddings from documents in the repo on first startup. This takes a few minutes as it downloads the `all-MiniLM-L6-v2` model (~90MB) and processes documents.

On subsequent runs, embeddings load from `data/document_embeddings.faiss` and you'll see "Embeddings loaded!" instead of "Generating embeddings...".

## Running the App
```bash
cd /home/ubuntu/repos/MediChat
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```
Then open `http://localhost:8501` in the browser.

## Key Test Scenarios

### 1. Chat History Persistence
- Send a message in the chat
- Restart the Streamlit app (kill and re-run)
- Verify the previous messages appear in the chat UI
- Check `data/chat_history.json` contains the messages

### 2. Restart with User Message as Last Entry
This is a regression-prone scenario. If the last saved message in `chat_history.json` is from the user (not the assistant), the app must not crash.
- Create/edit `data/chat_history.json` with a user message as the last entry
- Start the app and verify it loads without a `ValueError`
- The fix for this is a `prompt and` guard before processing responses

### 3. Embeddings Persistence
- Run the app once to generate embeddings
- Restart and verify "Embeddings loaded!" appears (not "Generating embeddings...")
- Check that `data/document_embeddings.faiss` exists on disk

### 4. Graceful Handling Without API Key
- Start the app without `ANTHROPIC_API_KEY` set
- Verify the app starts and shows embeddings
- Verify that attempting to send a message shows a user-friendly error (not a crash)

### 5. Chat Response Generation (requires ANTHROPIC_API_KEY)
- Set `ANTHROPIC_API_KEY` in the environment
- Send a medical question and verify Claude responds with relevant context from documents

## Docker Testing
```bash
docker build -t medi_chat .
docker run -p 8501:8501 -v medi_chat_data:/app/data -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" medi_chat
```
The `-v medi_chat_data:/app/data` flag is critical for persistence across container restarts.

## Tips
- The app uses Streamlit's `st.session_state` for in-memory state and `data/chat_history.json` for persistence. Both must stay in sync.
- When testing persistence, kill the Streamlit process completely (not just refresh the browser) to simulate a real restart.
- The embedding model (`all-MiniLM-L6-v2`) runs locally — no API key needed for embeddings.
- If you see numpy/faiss version conflicts, check that `requirements.txt` uses flexible version pins (`>=` not `==`).
