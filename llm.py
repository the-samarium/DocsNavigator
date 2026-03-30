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
from rich.padding import Padding
from rich.markup import escape

os.environ["USER_AGENT"] = "ProjectRAG/1.0"

CHAT_WIDTH = 96
console = Console(width=CHAT_WIDTH)

# ── LLM + Chain ────────────────────────────────────────────
llm = ChatOllama(model="glm-5:cloud", temperature=0.3)
rag_chain = build_rag_chain(llm)

def clean_response(content: str) -> str:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()

# ── UI ─────────────────────────────────────────────────────
def section_divider(label: str | None = None):
    line = "─" * 86
    if label:
        console.print(Padding(f"[dim]{line}[/dim]", (0, 2)))
        console.print(Padding(f"[bold dark_orange3]{label}[/bold dark_orange3]", (0, 2)))
        console.print(Padding(f"[dim]{line}[/dim]", (0, 2)))
    else:
        console.print(Padding(f"[dim]{line}[/dim]", (0, 2)))


def banner():
    os.system("cls" if os.name == "nt" else "clear")
    console.print()
    logo = (
        "[bold dark_orange3]██████╗  ██████╗  ██████╗ ███████╗[/bold dark_orange3]\n"
        "[bold dark_orange3]██╔══██╗██╔═══██╗██╔════╝██╔════╝[/bold dark_orange3]\n"
        "[bold dark_orange3]██║  ██║██║   ██║██║     ███████╗[/bold dark_orange3]\n"
        "[bold dark_orange3]██║  ██║██║   ██║██║     ╚════██║[/bold dark_orange3]\n"
        "[bold dark_orange3]██████╔╝╚██████╔╝╚██████╗███████║[/bold dark_orange3]\n"
        "[bold dark_orange3]╚═════╝  ╚═════╝  ╚═════╝╚══════╝[/bold dark_orange3]\n"
        "[bold white]  navigator[/bold white]"
    )
    console.print(
        Padding(
            Panel(
                Padding(logo, (1, 2)),
                border_style="white",
                padding=(0, 1),
                title="[bold dark_orange3]HEADER[/bold dark_orange3]",
                title_align="left",
                expand=False,
            ),
            (0, 2),
        )
    )
    section_divider()
    console.print()

def ask_website() -> str:
    section_divider("URL")
    console.print()
    console.print(Padding("[bold dark_orange3]Which documentation do you want to load?[/bold dark_orange3]", (0, 2)))
    console.print(Padding("[dim]All subpages under the URL will be crawled and indexed.[/dim]", (0, 2)))
    console.print()
    while True:
        url = Prompt.ask("  [dark_orange3]›[/dark_orange3] [dim]URL[/dim]").strip()
        if url:
            console.print(
                Padding(
                    "[bold dark_orange3]Based on pages number, embedding model used, retriver , etc. Initial db formation can take upto 8-10 min max on 200+ pages , if dept not set[/bold dark_orange3]",
                    (0, 2),
                )
            )
            return url
        console.print(Padding("[dark_red]✘  URL cannot be empty.[/dark_red]", (0, 2)))

def do_ingest(url: str):
    console.print()
    console.print(Padding(f"[dim]Crawling [bold]{escape(url)}[/bold] …[/dim]", (0, 2)))
    console.print()

    start = time.time()

    try:
        with Live(
            Spinner("dots", text=Text(" Building knowledge base …", style="dim")),
            console=console,
            refresh_per_second=12,
            transient=True,
        ):
            ingest(url)
    except Exception as e:
        console.print(Padding(f"[dark_red]✘  Ingestion failed: {escape(str(e))}[/dark_red]", (0, 2)))
        return False

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    timer_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    console.print(Padding(f"[bold orange1]✔  Knowledge base ready.[/bold orange1]  [dim]({timer_str})[/dim]", (0, 2)))
    console.print()
    return True

def startup_flow():
    if not os.path.exists("faiss_index"):
        # section_divider("STARTUP")
        url = ask_website()
        if not do_ingest(url):
            return False
    else:
        # section_divider("STARTUP")
        console.print(Padding("[bold dark_orange3]Startup[/bold dark_orange3]", (0, 2)))
        console.print()
        console.print(Padding("[orange1]●[/orange1]  [bold dark_orange3]Existing knowledge base found.[/bold dark_orange3]", (0, 2)))
        console.print()
        continue_chat = Confirm.ask(
            "  [dark_orange3]›[/dark_orange3] [dim]Continue chatting on existing knowledge base?[/dim]",
            default=True,
        )
        console.print()

        if continue_chat:
            console.print(Padding("[dim]Continuing on existing knowledge base.[/dim]", (0, 2)))
            console.print()
        else:
            wipe = Confirm.ask(
                "  [dark_orange3]›[/dark_orange3] [dim]Wipe existing index and start fresh?[/dim]",
                default=False,
            )
            console.print()
            if wipe:
                shutil.rmtree("faiss_index")
                console.print(Padding("[orange1]✔  Old index wiped.[/orange1]", (0, 2)))
            url = ask_website()
            if not do_ingest(url):
                return False

    section_divider()
    console.print()
    return True

def switch_flow():
    console.print()
    wipe = Confirm.ask(
        "  [dark_orange3]›[/dark_orange3] [dim]Wipe existing index and start fresh?[/dim]",
        default=False,
    )
    console.print()
    if wipe:
        shutil.rmtree("faiss_index")
        console.print(Padding("[orange1]✔  Old index wiped.[/orange1]", (0, 2)))
    url = ask_website()
    return do_ingest(url)

def print_response(text: str):
    # Only agent/user labels use a different orange shade for quick visual distinction.
    console.print(f"  [bold orange2]Assistant[/bold orange2] [dim]›[/dim]  {escape(text)}")

def run_application():
    banner()
    if not startup_flow():
        console.print(Padding("[dark_red]Cannot continue without a valid knowledge base. Restart and try again.[/dark_red]", (0, 2)))
        return

    console.print(
        Padding(
            "[bold dark_orange3]Chat started![/bold dark_orange3]  [dim]'exit' to quit  ·  'switch' to load a new site[/dim]",
            (0, 2),
        )
    )
    console.print()

    while True:
        console.print()
        query = Prompt.ask("  [bold orange3]You[/bold orange3] [dim]›[/dim]").strip()

        if not query.strip():
            continue

        if query.strip().lower() in ["exit", "quit", "q"]:
            console.print()
            section_divider()
            console.print(Padding("[bold dark_orange3]Exiting ...[/bold dark_orange3]", (1, 2)))
            console.print()
            break

        if query.strip().lower() == "switch":
            if not switch_flow():
                console.print(Padding("[dark_red]Switch failed. Keeping current knowledge base.[/dark_red]", (0, 2)))
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