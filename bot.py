import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

# Function to initialize a smaller model if available
def initialize_model():
    try:
        # Use a smaller model like "mistral" to reduce memory usage
        embeddings = OllamaEmbeddings(model="mistral")
        llm = OllamaLLM(model="mistral")
    except ValueError as e:
        st.error("Memory error: Please try using a smaller model or reduce document size.")
        raise e
    return embeddings, llm

# Initialize embeddings and LLM
embeddings, llm = initialize_model()

# Function to process the document
def process_document(file):
    text = file.getvalue().decode("utf-8")
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)  # Adjusted chunk size
    texts = text_splitter.split_text(text)
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

# Streamlit UI
st.title("RAG Chatbot")

uploaded_file = st.file_uploader("Upload a text file for the knowledge base", type="txt")

if uploaded_file:
    vectorstore = process_document(uploaded_file)
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = chain({"question": prompt, "chat_history": []})
            st.markdown(response['answer'])
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})

else:
    st.write("Please upload a text file to start chatting.")
