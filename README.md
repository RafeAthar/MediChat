# MediChat


Docker commands
----
docker build -t medi_chat .

docker run -p 8501:8501 -v medi_chat_data:/app/data -e OPENAI_API_KEY="YOUR_OPENAI_API_KEY" medi_chat



Visit http://localhost:8501/


# MediChat Documentation

## Overview
MediChat is a Streamlit-based web application designed to facilitate conversations about medical topics using stored documents. The application leverages OpenAI's language models to provide accurate responses based on the context derived from medical documents.

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **Document Processing**: PyPDF2, docx2txt
- **Machine Learning**: OpenAI API for embeddings and responses
- **Data Storage**: FAISS for efficient similarity search
- **Containerization**: Docker

## Prerequisites
Before running the application, ensure you have the following:
- Docker installed on your machine.
- An OpenAI API key. You can obtain one by signing up at [OpenAI](https://openai.com/).

## Setup Instructions

### 1. Clone the Repository
Clone the repository containing the application code to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Requirements File
Ensure the `requirements.txt` file is present in the root directory of the project. It should contain the following dependencies:

```plaintext
streamlit==1.24.0
PyPDF2==3.0.1
docx2txt==0.8
openai==0.27.0
faiss-cpu==1.7.2
numpy==1.23.0
```

### 3. Create a Dockerfile
Ensure the `Dockerfile` is present in the root directory of the project. It should contain the following instructions:
```Dockerfile
# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port on which the app will run
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 4. Build the Docker Image
Run the following command in the terminal to build the Docker image:
```bash
docker build -t medi_chat .
```

### 5. Run the Docker Container
To run the application, execute the following command, replacing `YOUR_OPENAI_API_KEY` with your actual OpenAI API key:
```bash
docker run -p 8501:8501 -v medi_chat_data:/app/data -e OPENAI_API_KEY="YOUR_OPENAI_API_KEY" medi_chat
```

> **Note:** The `-v medi_chat_data:/app/data` flag creates a named Docker volume that persists your data (chat history and document embeddings) across container restarts. Without this flag, data will be lost when the container is stopped or removed.

### 6. Access the Application
Open your web browser and navigate to [http://localhost:8501](http://localhost:8501) to access the MediChat application.

## Usage Instructions

1. **Upload Medical Documents**: Place your medical documents (PDF and DOCX formats) in the `./documents` directory. The application will process these documents to generate embeddings.

2. **Ask Questions**: Once the application is running, you can type your questions in the chat input box. The assistant will respond based on the context derived from the uploaded documents.

3. **View Responses**: The responses will include relevant excerpts from the documents used to generate the answer, providing context for the information provided.

## Logging
The application logs interactions and errors to a file named `app.log`. You can check this file for any issues or to review the conversation history.

## Conclusion
MediChat is a powerful tool for accessing medical knowledge through conversational AI. By following the setup instructions, you can easily deploy the application and start interacting with it. For any issues or contributions, feel free to reach out or submit a pull request.

