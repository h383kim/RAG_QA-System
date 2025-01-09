from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from src.config import OPENAI_API_KEY

def create_qa_chain(vectordb, k=3):
    """Create a RetrievalQA chain given a vector store."""
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def process_llm_response(llm_response):
    """Pretty-print the final result and the sources used."""
    print(llm_response['result'])
    print("\nSources:")
    for doc in llm_response['source_documents']:
        print("-", doc.metadata["source"])
