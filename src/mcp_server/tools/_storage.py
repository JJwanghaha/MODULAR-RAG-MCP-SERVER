"""三个 MCP 工具共用路径与已有 Chroma 打开方式，不创建空知识库。"""

from contextlib import contextmanager
from dataclasses import replace

from src.core.settings import load_settings, resolve_path


def configured_storage(settings=None, data_dir=None):
    settings = settings if settings is not None else load_settings()
    root = resolve_path("data" if data_dir is None else data_dir)
    if data_dir is not None:
        settings = replace(settings, vector_store=replace(settings.vector_store,
                           persist_directory=str(root / "db" / "chroma")))
    return settings, root


def validate_collection(name: str) -> None:
    if not isinstance(name, str) or not name.strip() or name in {".", ".."} or any(char in name for char in "/\\"):
        raise ValueError("collection must be a single path component")


@contextmanager
def existing_chroma_client(settings):
    if settings.vector_store.provider != "chroma":
        raise ValueError("Collection metadata tools currently require Chroma")
    path = resolve_path(settings.vector_store.persist_directory)
    if not (path / "chroma.sqlite3").is_file():
        raise FileNotFoundError("Knowledge database does not exist")
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(
        _env_file=None, anonymized_telemetry=False, allow_reset=False))
    try:
        yield client
    finally:
        client.close()
