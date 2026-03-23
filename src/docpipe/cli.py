"""CLI interface for docpipe."""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

from docpipe.config import DocpipeConfig, load_config
from docpipe.graph import rebuild_graph
from docpipe.pipeline import Lockfile, cleanup_orphans, process_file
from docpipe.registry import RegistryEntry, update_registry
from docpipe.status import StatusTracker
from docpipe.watcher import start_watcher

console = Console()
DEFAULT_CONFIG = "config.yaml"


def _setup_logging(cfg: DocpipeConfig) -> None:
    log_dir = Path(cfg.logging.file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        cfg.logging.file,
        maxBytes=cfg.logging.max_size_mb * 1024 * 1024,
        backupCount=cfg.logging.backup_count,
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

    root = logging.getLogger("docpipe")
    root.setLevel(getattr(logging, cfg.logging.level.upper()))
    root.addHandler(handler)


def _build_status_table(tracker: StatusTracker, cfg: DocpipeConfig) -> Table:
    data = tracker.to_dict()
    table = Table(title="DocPipe Status", show_lines=True)
    table.add_column("File")
    table.add_column("Status")
    table.add_column("Pages")
    table.add_column("Images")
    table.add_column("Graph")
    table.add_column("Updated")

    for fname, info in data.get("files", {}).items():
        status = info.get("status", "unknown")
        style = {"done": "green", "failed": "red", "processing": "yellow"}.get(status, "")
        table.add_row(
            fname,
            status,
            str(info.get("pages", "-")),
            str(info.get("images", "-")),
            "yes" if info.get("graph_ingested") else "no",
            info.get("last_processed", "-"),
            style=style,
        )

    graph = data.get("graph", {})
    api = data.get("api_usage", {})
    table.caption = (
        f"Graph: {graph.get('entities', 0)} entities, {graph.get('relations', 0)} relations | "
        f"API: {api.get('tokens_today', 0)} tokens today (~${api.get('estimated_cost_usd', 0):.4f})"
    )
    return table


async def _run_ingest(
    cfg: DocpipeConfig,
    file_path: str | None = None,
    rebuild: bool = False,
) -> None:
    if rebuild:
        md_dir = cfg.output_dir / "markdown"
        console.print(f"Rebuilding graph from {md_dir}...")
        count = await rebuild_graph(md_dir, cfg.graph)
        console.print(f"Graph rebuilt: {count} documents ingested.")
        return

    input_dir = cfg.input_dir
    if file_path:
        target = input_dir / file_path
        if not target.exists():
            console.print(f"[red]File not found: {target}[/red]")
            return
        files = [target]
    else:
        supported = set(cfg.converter.supported_extensions) | {".pdf"}
        files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in supported]

    console.print(f"Processing {len(files)} file(s)...")
    for f in files:
        console.print(f"  Processing: {f.name}")
        success = await process_file(f, cfg)
        status = "[green]done[/green]" if success else "[red]failed[/red]"
        console.print(f"  {f.name}: {status}")


def _watcher_callback(cfg: DocpipeConfig) -> Any:
    """Create the callback for the watcher."""

    def callback(files: list[Path], deleted: list[str]) -> None:
        for name in deleted:
            stem = Path(name).stem
            cleanup_orphans(stem, cfg.output_dir)
            reg_path = cfg.output_dir / cfg.registry.filename
            entry = RegistryEntry.failed(name, "deleted")
            update_registry(reg_path, entry, remove=True)
            tracker = StatusTracker(cfg.output_dir)
            tracker.remove_file(name)
            tracker.save()
            console.print(f"  Removed: {name}")

        for f in files:
            console.print(f"  Processing: {f.name}")
            success = asyncio.run(process_file(f, cfg))
            status = "[green]done[/green]" if success else "[red]failed[/red]"
            console.print(f"  {f.name}: {status}")

    return callback


@click.group()
def main() -> None:
    """DocPipe — document ingestion pipeline."""
    pass


@main.command()
def init() -> None:
    """Create default config.yaml and folder structure."""
    cfg_path = Path(DEFAULT_CONFIG)
    if cfg_path.exists():
        console.print("[yellow]config.yaml already exists, skipping[/yellow]")
    else:
        cfg_path.write_text("# DocPipe Configuration\ninput_dir: ./input\noutput_dir: ./output\n")
        console.print("Created config.yaml")

    for d in ["input", "output", "output/markdown", "output/images", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text("OPENAI_API_KEY=your-key-here\n")
        console.print("Created .env (set your OPENAI_API_KEY)")

    console.print("[green]Project initialized.[/green]")


@main.command()
@click.option("--config", "config_path", default=DEFAULT_CONFIG, help="Path to config.yaml")
@click.option("--dashboard", is_flag=True, help="Show live dashboard")
def run(config_path: str, dashboard: bool) -> None:
    """Start the file watcher (foreground, blocking)."""
    cfg = load_config(Path(config_path))
    _setup_logging(cfg)

    lock = Lockfile(cfg.output_dir / ".docpipe.lock")
    if not lock.acquire():
        console.print(
            "[red]Error: docpipe is already running (lockfile held). "
            "Stop the watcher first or wait for it to finish.[/red]"
        )
        sys.exit(1)

    try:
        tracker = StatusTracker(cfg.output_dir)
        tracker.set_watcher_running()
        tracker.save()

        callback = _watcher_callback(cfg)
        observer = start_watcher(cfg.input_dir, cfg.watcher, cfg.converter, callback)

        console.print(f"[green]Watching {cfg.input_dir} (Ctrl+C to stop)[/green]")

        if dashboard:
            initial_table = _build_status_table(tracker, cfg)
            with Live(initial_table, refresh_per_second=0.5, console=console) as live:
                while observer.is_alive():
                    tracker.heartbeat()
                    tracker.save()
                    live.update(_build_status_table(tracker, cfg))
                    observer.join(timeout=2)
        else:
            while observer.is_alive():
                tracker.heartbeat()
                tracker.save()
                observer.join(timeout=5)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping watcher...[/yellow]")
        observer.stop()
        observer.join()
    finally:
        tracker = StatusTracker(cfg.output_dir)
        tracker.set_watcher_stopped()
        tracker.save()
        lock.release()

    console.print("[green]Watcher stopped.[/green]")


@main.command()
@click.option("--config", "config_path", default=DEFAULT_CONFIG, help="Path to config.yaml")
@click.option("--rebuild-graph", "rebuild", is_flag=True, help="Rebuild graph from scratch")
@click.argument("file", required=False)
def ingest(config_path: str, rebuild: bool, file: str | None) -> None:
    """One-shot: process files in input_dir."""
    cfg = load_config(Path(config_path))
    _setup_logging(cfg)

    lock = Lockfile(cfg.output_dir / ".docpipe.lock")
    if not lock.acquire():
        console.print(
            "[red]Error: docpipe is already running (lockfile held). "
            "Stop the watcher first or wait for it to finish.[/red]"
        )
        sys.exit(1)

    try:
        asyncio.run(_run_ingest(cfg, file, rebuild))
    finally:
        lock.release()


@main.command()
@click.option("--config", "config_path", default=DEFAULT_CONFIG, help="Path to config.yaml")
def status(config_path: str) -> None:
    """Show pipeline status."""
    cfg = load_config(Path(config_path))
    tracker = StatusTracker(cfg.output_dir)
    table = _build_status_table(tracker, cfg)
    console.print(table)
