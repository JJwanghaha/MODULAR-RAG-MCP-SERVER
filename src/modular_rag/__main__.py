"""Allow the project to run with ``python -m modular_rag``."""

from modular_rag.interfaces.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
