import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain


# Function to initialize a smaller model if available
def initialize_model():
    try:
        # Initialize an embedding model to convert text into numerical vectors
        # "mistral" is used here as a smaller model to reduce memory usage
        embeddings = OllamaEmbeddings(model="mistral")

        # Load the language model for generating responses
        llm = OllamaLLM(model="mistral")
    except ValueError as e:
        st.error("Memory error: Please try using a smaller model or reduce document size.")
        raise e
    return embeddings, llm


# Initialize embeddings and LLM
embeddings, llm = initialize_model()


# Function to process the document and create vector representations
def process_document(file):
    # Read file content and decode it to a string
    text = file.getvalue().decode("utf-8")

    # Split text into smaller chunks for better retrieval
    # chunk_size=512 ensures each piece isn't too large for embedding
    # chunk_overlap=50 helps retain context between chunks
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    texts = text_splitter.split_text(text)

    # Convert text chunks into vector embeddings and store in FAISS for retrieval
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore


# Streamlit UI
st.title("RAG Chatbot")

# File uploader to accept a text file as knowledge base
uploaded_file = st.file_uploader("Upload a text file for the knowledge base", type="txt")

if uploaded_file:
    # Process uploaded document to create a vector store
    vectorstore = process_document(uploaded_file)

    # Create a retrieval-augmented generation (RAG) chain
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

    # Initialize chat history in session state if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("What would you like to know?"):
        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response using the RAG chain
        with st.chat_message("assistant"):
            response = chain({"question": prompt, "chat_history": []})
            st.markdown(response['answer'])

        # Store assistant's response in chat history
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})

else:
    st.write("Please upload a text file to start chatting.")
