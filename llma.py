import streamlit as st
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain

# Initialize embeddings and LLM
embeddings = OllamaEmbeddings(model="llama2")
llm = Ollama(model="llama2")

# Function to process the document
def process_document(file):
    text = file.getvalue().decode("utf-8")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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
