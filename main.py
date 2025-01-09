import os
from src.config import ARTICLES_DIRECTORY
from src.data_loader import download_data, load_documents
from src.vectorstore import build_vectorstore, load_vectorstore
from src.query_chain import create_qa_chain, process_llm_response

def main():
    # 1. Download data if needed
    zip_url = "https://github.com/kairess/toy-datasets/raw/master/techcrunch-articles.zip"
    zip_path = "techcrunch-articles.zip"
    if not os.path.exists(ARTICLES_DIRECTORY):
        os.makedirs(ARTICLES_DIRECTORY, exist_ok=True)
    download_data(zip_url, zip_path)

    # 2. Load documents
    documents = load_documents(ARTICLES_DIRECTORY)

    # 3. Build or load vectorstore
    # (If you want to always rebuild, uncomment below)
    vectordb = build_vectorstore(documents)

    # Or, if already persisted, you can just load:
    # vectordb = load_vectorstore()

    # 4. Create QA chain
    qa_chain = create_qa_chain(vectordb)

    # 5. Query
    query = "How much money did Pando raise?"
    print(f"Query: {query}")
    llm_response = qa_chain(query)

    # 6. Process response
    process_llm_response(llm_response)

if __name__ == "__main__":
    main()
