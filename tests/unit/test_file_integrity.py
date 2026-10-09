"""C2 通过公开接口验证真实 SQLite 状态与文件指纹。"""

import pytest

pytestmark = pytest.mark.unit


def test_success_record_survives_checker_recreation(tmp_path):
    from src.libs.loader.file_integrity import SQLiteIntegrityChecker

    path = tmp_path / "a.md"
    path.write_bytes(b"abc")
    db = tmp_path / "history" / "ingestion.db"
    checker = SQLiteIntegrityChecker(str(db))
    file_hash = checker.compute_sha256(str(path))

    assert file_hash == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert checker.should_skip(file_hash) is False
    checker.mark_success(file_hash, str(path), "notes")
    checker.close()

    assert SQLiteIntegrityChecker(str(db)).should_skip(file_hash) is True


def test_failure_does_not_block_retry_and_success_can_be_replaced(tmp_path):
    from src.libs.loader.file_integrity import SQLiteIntegrityChecker

    checker = SQLiteIntegrityChecker(str(tmp_path / "history.db"))
    checker.mark_failed("hash", "a.md", "test-error")
    assert checker.should_skip("hash") is False
    checker.mark_success("hash", "a.md", "notes")
    assert checker.should_skip("hash") is True
    checker.mark_failed("hash", "a.md", "retry-error")
    assert checker.should_skip("hash") is False


def test_hash_tracks_bytes_not_filename_and_changed_file_is_not_skipped(tmp_path):
    from src.libs.loader.file_integrity import SQLiteIntegrityChecker

    first, second = tmp_path / "a.md", tmp_path / "renamed.md"
    first.write_bytes(b"abc")
    second.write_bytes(b"abc")
    checker = SQLiteIntegrityChecker(str(tmp_path / "history.db"))
    old_hash = checker.compute_sha256(str(first))
    checker.mark_success(old_hash, str(first), "first-collection")

    assert checker.compute_sha256(str(second)) == old_hash
    assert checker.should_skip(checker.compute_sha256(str(second))) is True
    first.write_bytes(b"abcd")
    assert checker.should_skip(checker.compute_sha256(str(first))) is False


def test_skip_scope_preserves_upstream_global_content_hash_rule(tmp_path):
    from src.libs.loader.file_integrity import SQLiteIntegrityChecker

    checker = SQLiteIntegrityChecker(str(tmp_path / "history.db"))
    checker.mark_success("same-hash", "a.md", "collection-a")
    assert checker.should_skip("same-hash") is True
    checker.mark_success("same-hash", "b.md", "collection-b")
    assert checker.should_skip("same-hash") is True


def test_empty_file_has_known_sha256(tmp_path):
    from src.libs.loader.file_integrity import SQLiteIntegrityChecker

    path = tmp_path / "empty.md"
    path.write_bytes(b"")
    checker = SQLiteIntegrityChecker(str(tmp_path / "history.db"))
    assert checker.compute_sha256(str(path)) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.mark.parametrize("kind,error", [("missing", FileNotFoundError), ("directory", OSError)])
def test_hash_rejects_missing_file_and_directory(tmp_path, kind, error):
    from src.libs.loader.file_integrity import SQLiteIntegrityChecker

    checker = SQLiteIntegrityChecker(str(tmp_path / "history.db"))
    path = tmp_path / "missing.md" if kind == "missing" else tmp_path
    with pytest.raises(error):
        checker.compute_sha256(str(path))


def test_record_is_readable_in_a_fresh_process(tmp_path):
    import subprocess
    import sys
    from src.core.settings import REPO_ROOT
    from src.libs.loader.file_integrity import SQLiteIntegrityChecker

    db = str(tmp_path / "history.db")
    SQLiteIntegrityChecker(db).mark_success("hash", "a.md")
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; from src.libs.loader.file_integrity import SQLiteIntegrityChecker; "
        "assert SQLiteIntegrityChecker(sys.argv[1]).should_skip('hash')",
        db,
    ], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_independent_connections_accept_concurrent_marking(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from src.libs.loader.file_integrity import SQLiteIntegrityChecker

    checker = SQLiteIntegrityChecker(str(tmp_path / "history.db"))
    def mark(index):
        checker.mark_success(f"hash-{index}", f"file-{index}.md", "notes")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(mark, range(12)))
    assert all(checker.should_skip(f"hash-{index}") for index in range(12))
