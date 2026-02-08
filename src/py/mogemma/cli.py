import sys
import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .hub import HubManager
from .config import GenerationConfig, EmbeddingConfig
from .model import GemmaModel, EmbeddingModel

console = Console()

@click.group()
def cli():
    """mogemma: High-performance Gemma 3 inference via Mojo."""
    pass

@cli.command()
@click.argument("model_id")
@click.option("--cache-dir", type=click.Path(), help="Custom cache directory.")
def pull(model_id: str, cache_dir: str | None):
    """Download a model from Hugging Face Hub."""
    hub = HubManager(cache_path=cache_dir)
    console.print(f"[bold blue]Pulling {model_id}...[/bold blue]")
    try:
        path = hub.download(model_id)
        console.print(f"[bold green]Successfully downloaded to {path}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error downloading model: {e}[/bold red]")
        sys.exit(1)

@cli.command()
def info():
    """Show system information and backend status."""
    table = Table(title="mogemma System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    
    # Check Mojo bridge
    from . import _core
    if _core is not None:
        table.add_row("Mojo Backend", "[green]Available (MAX optimized)[/green]")
    else:
        table.add_row("Mojo Backend", "[yellow]Not Available (Fallback to dummy mode)[/yellow]")
    
    # Check HF Cache
    hub = HubManager()
    table.add_row("Cache Path", str(hub.cache_path))
    
    console.print(table)

@cli.command()
@click.argument("model_id")
@click.option("--temp", default=1.0, help="Sampling temperature.")
@click.option("--max-tokens", default=512, help="Max tokens to generate.")
def chat(model_id: str, temp: float, max_tokens: int):
    """Start an interactive chat session with a Gemma model."""
    hub = HubManager()
    model_path = hub.resolve_model(model_id)
    
    if not model_path.exists():
        console.print(f"[bold red]Model not found: {model_id}[/bold red]")
        console.print(f"Run [bold cyan]mogemma pull {model_id}[/bold cyan] first.")
        sys.exit(1)
        
    config = GenerationConfig(model_path=model_path, temperature=temp, max_new_tokens=max_tokens)
    model = GemmaModel(config)
    
    console.print(Panel(f"Starting chat with [bold cyan]{model_id}[/bold cyan]\nType 'exit' or 'quit' to stop.", title="mogemma Chat"))
    
    while True:
        try:
            prompt = console.input("[bold green]User > [/bold green]")
            if prompt.lower() in ["exit", "quit"]:
                break
                
            console.print("[bold blue]Gemma > [/bold blue]", end="")
            for chunk in model.generate_stream(prompt):
                console.print(chunk, end="")
            console.print()
        except KeyboardInterrupt:
            break

@cli.command()
@click.argument("model_id")
@click.argument("text")
def embed(model_id: str, text: str):
    """Generate embeddings for the given text."""
    hub = HubManager()
    model_path = hub.resolve_model(model_id)
    
    if not model_path.exists():
        console.print(f"[bold red]Model not found: {model_id}[/bold red]")
        sys.exit(1)
        
    config = EmbeddingConfig(model_path=model_path)
    model = EmbeddingModel(config)
    
    vectors = model.embed(text)
    console.print(f"[bold green]Embedding shape: {vectors.shape}[/bold green]")
    console.print(vectors)

if __name__ == "__main__":
    cli()
