# RAG Chatbot Project

## Overview

This project implements a simple Retrieval-Augmented Generation (RAG) chatbot using LangChain, Streamlit, and the open-source Llama model via Ollama. The chatbot allows users to upload a text file as a knowledge base and then ask questions based on that information.

## Features

- Upload custom knowledge base via text file
- Interactive chat interface
- Utilizes RAG for more accurate and context-aware responses
- Runs locally without need for API keys

## Prerequisites

- Python 3.7+
- Ollama (with Llama2 model installed)

## Installation

1. Clone the repository:
```bash 
git clone https://github.com/dangminh214/RAG-Ollama-simple.git
```

3. Create a virtual environment (optional but recommended):
```bash
py -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate
```

5. Install the required packages:
```bash
py -m pip install -U langchain langchain-community streamlit faiss-cpu
```

7. Install Ollama from [https://ollama.ai/](https://ollama.ai/)

8. Pull the Llama2 model:
```bash
ollama pull mistral
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run llma.py
```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`)

3. Upload a text file to serve as the knowledge base

4. Start chatting with the bot by typing questions in the input box

## Code Structure

The main components of the code are:

- `OllamaEmbeddings`: For creating embeddings of the text
- `FAISS`: Vector store for efficient similarity search
- `CharacterTextSplitter`: Splits the input text into manageable chunks
- `Ollama`: The language model used for generating responses
- `ConversationalRetrievalChain`: Combines the retrieval and generation process

## Customization

You can customize the chatbot by:

- Changing the Ollama model (e.g., from "llama2" to another available model)
- Adjusting the `chunk_size` and `chunk_overlap` in the `CharacterTextSplitter`
- Modifying the Streamlit UI for a different look and feel

## Limitations

- The quality of responses depends on the uploaded knowledge base
- The bot's knowledge is limited to the uploaded text and the pre-trained knowledge of the Llama2 model
- Running locally means performance depends on your hardware

## Future Improvements

- Add support for multiple file uploads
- Implement persistent storage for the vector store
- Add options for different embedding and LLM models
- Improve error handling and user feedback
