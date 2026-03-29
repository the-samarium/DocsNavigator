# recursively scrapes all subpages from a given base URL
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup

def normalize_url(url: str) -> str:
    if not url.startswith("http"):
        url = "https://" + url
    return url.rstrip("/")

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # remove nav, footer, scripts — keep only main content
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def load_documents(url: str):
    base_url = normalize_url(url)
    # print(f"Loader: Crawling all subpages of {base_url} ...")

    loader = RecursiveUrlLoader(
        url=base_url,
        extractor=bs4_extractor,
        prevent_outside=True,  # stay within the same domain
        max_depth=8,  # reduce crawl scope for faster ingestion, # remove this to load all pages , but loading and injestion will take time
        timeout=10,
    )

    docs = loader.load()
    print(f"Loader: Loaded {len(docs)} page(s).")
    return docs