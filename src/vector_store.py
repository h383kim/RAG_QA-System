import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import (OPENAI_API_KEY, PERSIST_DIRECTORY,
                        CHUNK_SIZE, CHUNK_OVERLAP)

def build_vectorstore(documents):
    """Build and persist a Chroma vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)

    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vectordb = Chroma.from_documents(
        documents=docs, 
        embedding=embedding,
        persist_directory=PERSIST_DIRECTORY
    )
    vectordb.persist()
    return vectordb

def load_vectorstore():
    """Load an existing Chroma vector store from disk."""
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding
    )
    return vectordb
