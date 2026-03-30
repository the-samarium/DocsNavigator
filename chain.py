# chain definitions — imported by llm.py
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from loader import load_documents
from ingest import ingest
from retriver import retrive

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        You are an expert technical documentation assistant.
        Use the provided context fragments to answer the user's question accurately.

        INSTRUCTIONS:
        1. Prioritize core, introductory, and setup information (e.g., getting started, installation, guide , etc) if the question is general.
        2. If the context contains instructions for multiple different modules or scenarios, differentiate between them clearly (e.g., "Core Installation" vs. "Database Integration").
        3. Use well-formatted Markdown: headers (###), bullet points, and code blocks (```).
        4. If the answer is not contained within the context provided, simply say: "I'm sorry, I couldn't find specific information about that in the current documentation index."
        5. Keep responses technical, concise, and professional.

        Context: 
        ----------------------
        {context}
        ----------------------

        Question: {question}

        Answer:
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