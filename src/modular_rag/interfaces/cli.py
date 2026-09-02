"""Minimal command-line interface used to verify the project skeleton."""

import argparse
from collections.abc import Sequence

from modular_rag import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without performing application work."""
    parser = argparse.ArgumentParser(
        prog="modular-rag",
        description="Modular RAG development entry point.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the minimal A1 entry point."""
    build_parser().parse_args(argv)
    print("Modular RAG skeleton is ready. Next task: A2 test foundation.")
    return 0
