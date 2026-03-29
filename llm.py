# entrypoint — run with: python llm.py
import os
import re
import shutil
import time
from langchain_ollama import ChatOllama
from chain import build_rag_chain
from ingest import ingest

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text
from rich.rule import Rule
from rich.padding import Padding
from rich.markup import escape

os.environ["USER_AGENT"] = "ProjectRAG/1.0"

console = Console()

# ── LLM + Chain ────────────────────────────────────────────
llm = ChatOllama(model="glm-5:cloud", temperature=0.3)
rag_chain = build_rag_chain(llm)

def clean_response(content: str) -> str:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()

# ── UI ─────────────────────────────────────────────────────
def banner():
    os.system("cls" if os.name == "nt" else "clear")
    console.print()
    console.print(Panel.fit(
        "[bold cyan]  RAG  [/bold cyan][dim] · Retrieval-Augmented Generation[/dim]\n"
        "[dim]    LangChain  ·  FAISS  ·  Ollama[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()

def ask_website() -> str:
    console.print(Rule("[dim]Load new documentation[/dim]", style="dim"))
    console.print()
    console.print(Padding("[bold white]Which documentation do you want to load?[/bold white]", (0, 2)))
    console.print(Padding("[dim]All subpages under the URL will be crawled and indexed.[/dim]", (0, 2)))
    console.print()
    while True:
        url = Prompt.ask("  [cyan]›[/cyan] [dim]URL[/dim]").strip()
        if url:
            return url
        console.print(Padding("[red]✘  URL cannot be empty.[/red]", (0, 2)))

def do_ingest(url: str):
    console.print()
    console.print(Padding(f"[dim]Crawling [bold]{escape(url)}[/bold] …[/dim]", (0, 2)))
    console.print()

    start = time.time()

    with Live(
        Spinner("dots", text=Text(" Building knowledge base …", style="dim")),
        console=console,
        refresh_per_second=12,
        transient=True,
    ):
        try:
            ingest(url)
        except Exception as e:
            console.print(Padding(f"[red]✘  Ingestion failed: {escape(str(e))}[/red]", (0, 2)))
            return False

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    timer_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    console.print(Padding(f"[bold green]✔  Knowledge base ready.[/bold green]  [dim]({timer_str})[/dim]", (0, 2)))
    console.print()
    return True

def startup_flow():
    if not os.path.exists("faiss_index"):
        url = ask_website()
        if not do_ingest(url):
            return False
    else:
        console.print(Rule("[dim]Startup[/dim]", style="dim"))
        console.print()
        console.print(Padding("[green]●[/green]  [bold white]Existing knowledge base found.[/bold white]", (0, 2)))
        console.print()
        continue_chat = Confirm.ask(
            "  [cyan]›[/cyan] [dim]Continue chatting on existing knowledge base?[/dim]",
            default=True
        )
        console.print()

        if continue_chat:
            console.print(Padding("[dim]Continuing on existing knowledge base.[/dim]", (0, 2)))
            console.print()
        else:
            wipe = Confirm.ask(
                "  [cyan]›[/cyan] [dim]Wipe existing index and start fresh?[/dim]",
                default=False
            )
            console.print()
            if wipe:
                shutil.rmtree("faiss_index")
                console.print(Padding("[green]✔  Old index wiped.[/green]", (0, 2)))
            url = ask_website()
            if not do_ingest(url):
                return False

    console.print(Rule(style="dim"))
    console.print()
    return True

def switch_flow():
    console.print()
    wipe = Confirm.ask(
        "  [cyan]›[/cyan] [dim]Wipe existing index and start fresh?[/dim]",
        default=False
    )
    console.print()
    if wipe:
        shutil.rmtree("faiss_index")
        console.print(Padding("[green]✔  Old index wiped.[/green]", (0, 2)))
    url = ask_website()
    return do_ingest(url)

def print_response(text: str):
    console.print(f"  [bold orange1]Assistant[/bold orange1] [dim]›[/dim]  {escape(text)}")

def run_application():
    banner()
    if not startup_flow():
        console.print(Padding("[red]Cannot continue without a valid knowledge base. Restart and try again.[/red]", (0, 2)))
        return

    console.print(Padding(
        "[bold green]Chat started![/bold green]  [dim]'exit' to quit  ·  'switch' to load a new site[/dim]",
        (0, 2)
    ))
    console.print()

    while True:
        console.print()
        query = Prompt.ask("  [bold orange1]You[/bold orange1] [dim]›[/dim]")

        if not query.strip():
            continue

        if query.strip().lower() in ["exit", "quit", "q"]:
            console.print()
            console.print(Rule(style="dim"))
            console.print(Padding("[bold cyan]Exiting ...[/bold cyan]", (1, 2)))
            console.print()
            break

        if query.strip().lower() == "switch":
            if not switch_flow():
                console.print(Padding("[red]Switch failed. Keeping current knowledge base.[/red]", (0, 2)))
            continue

        console.print()
        with Live(
            Spinner("dots2", text=Text(" Retrieving context and generating answer …", style="dim")),
            console=console,
            refresh_per_second=12,
            transient=True,
        ):
            result = rag_chain.invoke(query.strip())
            response = clean_response(result.content)

        print_response(response)

if __name__ == "__main__":
    run_application()