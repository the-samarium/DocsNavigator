# load FAISS index and retrieve via MMR
import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from sympy import true

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def retrive(query):
    if not os.path.exists("faiss_index"):
        raise FileNotFoundError("No FAISS index found. Run ingestion first.")

    # allow_unsafe = os.getenv("ALLOW_DANGEROUS_DESERIALIZATION", "false").lower() == "true"
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=true
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,          # fewer final chunks → faster LLM
            "fetch_k": 20,   # not too large → faster retrieval
            "lambda_mult": 0.6,
    }
    )

    return retriever.invoke(query)