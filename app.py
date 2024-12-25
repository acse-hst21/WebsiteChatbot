import streamlit as st
import torch
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

huggingfacetoken = os.getenv('huggingfacetoken')

st.title("Website chatbot :robot_face:")
st.caption("This app allows you to chat with a webpage using RAG methods")

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'webpage_url' not in st.session_state:
    st.session_state.webpage_url = None
if 'query' not in st.session_state:
    st.session_state.query = None

def set_device() -> str:
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def rag_chain(prompt_template: str, query: str, huggingface_llm: HuggingFaceEndpoint,
              vectorstore: FAISS) -> dict:
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    retrievalQA = RetrievalQA.from_chain_type(
        llm=huggingface_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = retrievalQA.invoke({'query': query})

    return result

prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum.
3. If you are provided context that is not relevant to the question, ignore it.

{context}

Question: {question}

Helpful Answer:
"""

@st.cache_data(show_spinner=False)
def create_vectorstore(_splits: list, _embeddings: HuggingFaceBgeEmbeddings):
    return FAISS.from_documents(documents=_splits, embedding=_embeddings)

# Get the webpage URL from the user
webpage_url = st.text_input("Enter Webpage URL", type="default")

if webpage_url and st.session_state.webpage_url != webpage_url:
    st.session_state.clear()
    with st.spinner('Loading webpage...'):
        # 1. Load the data
        try:
            loader = WebBaseLoader(webpage_url)
            docs = loader.load()
        except Exception as e:
            st.error(f'Could not load webpage. Error: {e}')
            st.stop()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # 2. Create embeddings and vector store
        device = set_device()
        HF_embeddings = HuggingFaceBgeEmbeddings(
            model_name= 'sentence-transformers/all-MiniLM-l6-v2',
            model_kwargs= {'device': device},
            encode_kwargs= {'normalize_embeddings': True}
        )
        
        vectorstore = create_vectorstore(_splits=splits, _embeddings=HF_embeddings)

        st.session_state.webpage_url = webpage_url
        st.session_state.vectorstore = vectorstore

    st.success("Loaded webpage successfully!")

if st.session_state.vectorstore:
    # Ask a question about the webpage
    query = st.text_input("Ask any question about the webpage")

    # Chat with the webpage
    if query:
        huggingface_llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=huggingfacetoken,
            repo_id='mistralai/Mistral-7B-v0.1',
            temperature=0.05,
            max_new_tokens=200
        )

        with st.spinner('Processing your query. This might take a minute...'):
            rag_result = rag_chain(prompt_template=prompt_template,
                                   query=query,
                                   huggingface_llm=huggingface_llm,
                                   vectorstore=st.session_state.vectorstore)
            result = rag_result['result']

        if result.strip() == "":
            result = "I'm unable to generate a relevant answer."

        st.write(result)
        print(rag_result)
