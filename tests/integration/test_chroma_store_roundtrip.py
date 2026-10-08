"""用真实临时 Chroma 验证公开存储接口，不调用任何模型。"""

from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

from src.core.settings import load_settings
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def database_root():
    """模块结束后清理本次测试专属的数据库目录。"""
    with TemporaryDirectory(prefix="rag-chroma-test-") as path:
        yield path


@pytest.fixture(autouse=True)
def offline_database_environment(monkeypatch, database_root):
    """SDK 全局配置不能读项目 .env，测试也不能连接外部网络。"""
    import socket

    monkeypatch.chdir(database_root)

    def no_network(*args, **kwargs):
        raise AssertionError("Chroma integration tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)


@pytest.fixture
def store(database_root):
    """共享本地引擎，但每条测试使用独立 collection。"""
    result = VectorStoreFactory.create(
        load_settings(), persist_directory=database_root,
        collection_name=f"test_{uuid4().hex}",
    )
    yield result
    result.client.close()


def test_chroma_factory_roundtrip_returns_text_metadata_and_cosine_score(store):
    """上游 metadata.text 路径写入后，查询返回对应文本和归一化分数。"""
    store.upsert([
        {"id": "a", "vector": [1.0, 0.0], "metadata": {"text": "甲内容", "source_path": "a.md"}},
        {"id": "b", "vector": [0.0, 1.0], "metadata": {"text": "乙内容", "source_path": "b.md"}},
    ])
    result = store.query([1.0, 0.0], top_k=2)
    assert [record["id"] for record in result] == ["a", "b"]
    assert result[0]["text"] == "甲内容"
    assert result[0]["metadata"] == {"text": "甲内容", "source_path": "a.md"}
    assert [record["score"] for record in result] == pytest.approx([1.0, 0.5])


def test_metadata_conversions_preserve_upstream_behavior_without_mutating_input(store):
    """保留标量、丢弃 None、将列表和字典转字符串，不悄悄改造协议。"""
    metadata = {
        "text": "结构示例", "tags": ["a", "b"], "pages": (1, 2),
        "position": {"page": 1}, "optional": None,
        "enabled": True, "count": 2, "weight": 0.25,
    }
    store.upsert([{"id": "rich", "vector": [1.0, 0.0], "metadata": metadata}])
    result = store.query([1.0, 0.0], top_k=1)[0]
    assert result["text"] == "结构示例"
    assert result["metadata"] == {
        "text": "结构示例", "tags": "a,b", "pages": "1,2",
        "position": "{'page': 1}", "enabled": True, "count": 2, "weight": 0.25,
    }
    assert metadata["tags"] == ["a", "b"]
    assert metadata["position"] == {"page": 1}
    assert metadata["optional"] is None


def test_missing_metadata_text_uses_id_even_when_top_level_text_is_present(store):
    """按上游契约忽略顶层 text，空 metadata 用占位字段保存。"""
    store.upsert([{"id": "fallback-id", "vector": [1.0, 0.0], "text": "顶层文本不参与映射"}])
    result = store.query([1.0, 0.0], top_k=1)[0]
    assert result["text"] == "fallback-id"
    assert result["metadata"] == {"_placeholder": "true"}


def test_repeated_upsert_updates_vector_text_and_metadata_without_duplicate(store):
    """相同 ID 更新后查询只得到一条新记录，向量与文本同时变化。"""
    store.upsert([{"id": "same", "vector": [1.0, 0.0], "metadata": {"text": "旧文本", "source_path": "old.md"}}])
    update = [{"id": "same", "vector": [0.0, 1.0], "metadata": {"text": "新文本", "source_path": "new.md"}}]
    store.upsert(update)
    store.upsert(update)
    result = store.query([1.0, 0.0], top_k=1)
    assert len(result) == 1
    assert result[0]["id"] == "same"
    assert result[0]["text"] == "新文本"
    assert result[0]["metadata"] == {"text": "新文本", "source_path": "new.md"}
    assert result[0]["score"] == pytest.approx(0.5)


@pytest.mark.parametrize("filters,expected", [
    ({"scope": "public"}, ["public-b", "public-a"]),
    ({"scope": "public", "source_path": "a.md"}, ["public-a", "public-a2"]),
])
def test_metadata_filters_apply_before_top_k_with_one_or_multiple_conditions(store, filters, expected):
    """最相似的记录不符合过滤条件时，不能占用 TopK 名额。"""
    store.upsert([
        {"id": "private-a", "vector": [1.0, 0.0], "metadata": {"text": "甲", "scope": "private", "source_path": "a.md"}},
        {"id": "public-b", "vector": [0.8, 0.6], "metadata": {"text": "乙", "scope": "public", "source_path": "b.md"}},
        {"id": "public-a", "vector": [0.0, 1.0], "metadata": {"text": "丙", "scope": "public", "source_path": "a.md"}},
        {"id": "public-a2", "vector": [-1.0, 0.0], "metadata": {"text": "丁", "scope": "public", "source_path": "a.md"}},
    ])
    assert [record["id"] for record in store.query([1.0, 0.0], top_k=2, filters=filters)] == expected


def test_data_survives_writer_exit_and_is_read_by_a_new_process(database_root):
    """写入进程退出后，新进程通过公开 query 读回真实持久化数据。"""
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from src.core.settings import REPO_ROOT

    path = str(Path(database_root) / "restart-roundtrip")
    setup = '''
import socket
import sys
from src.core.settings import load_settings
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

def no_network(*args, **kwargs):
    raise AssertionError("persistent test must stay offline")

socket.socket.connect = no_network
store = VectorStoreFactory.create(load_settings(), persist_directory=sys.argv[1], collection_name="restart_test")
'''
    writer = setup + '''
store.upsert([{"id": "persisted", "vector": [1.0, 0.0], "metadata": {"text": "持久化示例", "source_path": "sample.md"}}])
store.client.close()
print("writer finished")
'''
    reader = setup + '''
import json
result = store.query([1.0, 0.0], top_k=1)
store.client.close()
print(json.dumps(result, ensure_ascii=False))
'''
    environment = {key: value for key, value in os.environ.items() if key not in {
        "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    }}
    environment["PYTHONPATH"] = str(REPO_ROOT)
    written = subprocess.run(
        [sys.executable, "-c", writer, path], cwd=database_root, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert written.returncode == 0, written.stderr
    assert written.stdout.strip() == "writer finished"
    loaded = subprocess.run(
        [sys.executable, "-c", reader, path], cwd=database_root, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert loaded.returncode == 0, loaded.stderr
    result = json.loads(loaded.stdout)
    assert len(result) == 1
    assert result[0]["id"] == "persisted"
    assert result[0]["text"] == "持久化示例"
    assert result[0]["metadata"] == {"text": "持久化示例", "source_path": "sample.md"}
    assert result[0]["score"] == pytest.approx(1.0)


@pytest.mark.parametrize("operation", ["upsert", "query"])
def test_dimension_mismatch_reports_storage_operation(store, operation):
    """集合已确定为二维后，不能混入其他维度，错误说明失败操作。"""
    store.upsert([{"id": "two-dim", "vector": [1.0, 0.0], "metadata": {"text": "二维"}}])
    with pytest.raises(RuntimeError, match=f"Chroma {operation} failed"):
        if operation == "upsert":
            store.upsert([{"id": "three-dim", "vector": [1.0, 0.0, 0.0]}])
        else:
            store.query([1.0, 0.0, 0.0], top_k=1)


def test_empty_collection_returns_no_results(store):
    """合法向量查询空库是正常结果，不是服务错误。"""
    assert store.query([1.0, 0.0], top_k=2) == []


def test_filter_with_no_matches_returns_no_results(store):
    """过滤条件没有匹配记录时不能返回不符合范围的数据。"""
    store.upsert([{"id": "only-private", "vector": [1.0, 0.0], "metadata": {"text": "示例", "scope": "private"}}])
    assert store.query([1.0, 0.0], top_k=1, filters={"scope": "public"}) == []


@pytest.mark.parametrize("records", [
    [], [{"id": "a"}], [{"vector": [1.0, 0.0]}],
    [{"id": "a", "vector": []}], ["not-a-record"],
])
def test_invalid_records_are_rejected_by_public_entry(store, records):
    """沿用 BaseVectorStore 的写入前置条件。"""
    with pytest.raises(ValueError):
        store.upsert(records)


@pytest.mark.parametrize("vector,top_k", [
    ([], 1), ("not-a-vector", 1), ([1.0, 0.0], 0),
    ([1.0, 0.0], -1), ([1.0, 0.0], 1.5),
])
def test_invalid_query_parameters_are_rejected_by_public_entry(store, vector, top_k):
    """空向量和非法 TopK 在数据库查询前被拒绝。"""
    with pytest.raises(ValueError):
        store.query(vector, top_k=top_k)
