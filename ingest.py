# chunk -> clean -> embed -> save to FAISS
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from loader import load_documents

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

console = Console()
ALLOW_UNSAFE_DESERIALIZATION = (
    os.getenv("ALLOW_DANGEROUS_DESERIALIZATION", "false").lower() == "true"
)


def clean_chunks(chunks):
    cleaned = []
    for chunk in chunks:
        text = chunk.page_content.strip()
        if len(text) < 20:
            continue
        text = text.replace("\x00", "").encode("utf-8", "ignore").decode("utf-8")
        chunk.page_content = text
        cleaned.append(chunk)
    return cleaned

def build_batch_index(batch):
    """Embed one batch and return a FAISS shard."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.from_documents(batch, embeddings)


def ingest(url: str, documents=None):
    if documents is None:
        documents = load_documents(url)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    chunks = clean_chunks(chunks)

    if not chunks:
        raise ValueError("No valid chunks to embed. Check the URL or page content.")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    BATCH_SIZE = 65
    EMBED_WORKERS = 2
    total_batches = (len(chunks) - 1) // BATCH_SIZE + 1
    vectorstore = None
    batches = [chunks[i : i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    with Live(
        Spinner("dots", text=Text(f" Embedding chunks …  (0/{total_batches} batches)", style="dim")),
        console=console,
        refresh_per_second=12,
        transient=True,
    ) as live:
        completed = 0
        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
            futures = [executor.submit(build_batch_index, batch) for batch in batches]
            for future in as_completed(futures):
                batch_store = future.result()
                if vectorstore is None:
                    vectorstore = batch_store
                else:
                    vectorstore.merge_from(batch_store)

                completed += 1
                live.update(
                    Spinner(
                        "dots",
                        text=Text(
                            f" Embedding chunks …  ({completed}/{total_batches} batches, {EMBED_WORKERS} workers)",
                            style="dim",
                        ),
                    )
                )

    # merge with existing index if present
    try:
        existing = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=ALLOW_UNSAFE_DESERIALIZATION,
        )
        existing.merge_from(vectorstore)
        existing.save_local("faiss_index")
    except Exception:
        vectorstore.save_local("faiss_index")