import os
import requests
import zipfile

from langchain.document_loaders import TextLoader, DirectoryLoader
from src.config import ARTICLES_DIRECTORY

def download_data(url: str, zip_path: str):
    """Download and unzip data if not already present."""
    if not os.path.exists(zip_path):
        response = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(response.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(ARTICLES_DIRECTORY)

def load_documents(directory: str):
    """Use LangChain's DirectoryLoader to load text documents."""
    loader = DirectoryLoader(directory, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents
