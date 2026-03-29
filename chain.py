# chain definitions — imported by llm.py
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from loader import load_documents
from ingest import ingest
from retriver import retrive

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        You are a helpful assistant.
        Use the following context to answer the question.
        If the answer is not in the context, say "I don't know".

        Context: {context}
        Question: {question}
    """
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retrieval_step = RunnableLambda(retrive)

def build_setup_chain(url: str):
    """Builds an indexing chain for a specific documentation URL."""
    load_step = RunnableLambda(lambda _: load_documents(url))
    ingest_step = RunnableLambda(lambda docs: ingest(url, documents=docs))
    return load_step | ingest_step

def build_rag_chain(llm):
    """Accepts the LLM and returns a ready RAG chain."""
    return (
        {"context": retrieval_step | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )