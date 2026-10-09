"""C13：图片文件留在磁盘，SQLite 只登记 ID、路径和来源。"""

import shutil
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.settings import resolve_path


class ImageStorage:
    """沿用上游 image_index 表；每次操作独立打开和关闭连接。"""

    def __init__(self, db_path: str = "data/db/image_index.db", images_root: str = "data/images"):
        self.db_path = str(resolve_path(db_path))
        self.images_root = resolve_path(images_root)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS image_index (
                image_id TEXT PRIMARY KEY, file_path TEXT NOT NULL,
                collection TEXT, doc_hash TEXT, page_num INTEGER, created_at TEXT NOT NULL
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)")
            connection.commit()

    def register_image(self, image_id: str, file_path: str | Path, collection: str | None = None,
                       doc_hash: str | None = None, page_num: int | None = None) -> str:
        """登记 Loader 已保存的文件，不复制或移动；Markdown 页码允许为空。"""
        if not image_id.strip():
            raise ValueError("image_id must be nonempty")
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError("Image file does not exist")
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("""INSERT OR REPLACE INTO image_index
                (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""", (image_id, str(path), collection, doc_hash, page_num,
                                               datetime.now(timezone.utc).isoformat()))
            connection.commit()
        return str(path)

    def save_image(self, image_id: str, image_data: bytes | Path | str, collection: str | None = None,
                   doc_hash: str | None = None, page_num: int | None = None, extension: str = "png") -> str:
        """按上游复制原始字节；extension 只是后缀，不执行格式转换。"""
        namespace = collection if collection is not None else "default"
        for component in (namespace, image_id, extension):
            if not component.strip() or component in {".", ".."} or any(char in component for char in "/\\"):
                raise ValueError("Image destination components must be single names")
        destination = self.images_root / namespace / f"{image_id}.{extension}"
        # 防止已有目录或文件符号链接把写入引向图片根目录之外。
        if not destination.resolve().is_relative_to(self.images_root.resolve()):
            raise ValueError("Image destination escapes images_root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(image_data, bytes):
            destination.write_bytes(image_data)
        elif isinstance(image_data, (str, Path)):
            source = Path(image_data)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        else:
            raise ValueError("image_data must be bytes or a file path")
        return self.register_image(image_id, destination, collection, doc_hash, page_num)

    def get_image_path(self, image_id: str) -> str | None:
        """返回登记的绝对路径；登记存在不代表文件尚在磁盘上。"""
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute("SELECT file_path FROM image_index WHERE image_id = ?", (image_id,)).fetchone()
        return row[0] if row else None

    def image_exists(self, image_id: str) -> bool:
        """与上游一致，只判断登记是否存在。"""
        return self.get_image_path(image_id) is not None

    def list_images(self, collection: str | None = None, doc_hash: str | None = None) -> list[dict[str, Any]]:
        """按集合及文档哈希筛选元数据，不返回图片字节。"""
        query, values = "SELECT * FROM image_index WHERE 1=1", []
        if collection is not None:
            query += " AND collection = ?"
            values.append(collection)
        if doc_hash is not None:
            query += " AND doc_hash = ?"
            values.append(doc_hash)
        query += " ORDER BY created_at, image_id"
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(query, values).fetchall()]

    def close(self) -> None:
        """兼容上游调用入口；本实现没有需要延后关闭的连接。"""
