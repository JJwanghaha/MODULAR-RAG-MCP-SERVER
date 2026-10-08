"""用内存策略验证上游向量存储接口的数据契约。"""

from copy import deepcopy
from dataclasses import replace

import pytest

from src.core.settings import load_settings
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

pytestmark = pytest.mark.unit


class FakeVectorStore(BaseVectorStore):
    """仅存在于测试中的内存存储，分数用点积作确定性示例。"""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.records = {}

    def upsert(self, records, trace=None, **kwargs):
        self.validate_records(records)
        for record in records:
            self.records[record["id"]] = deepcopy(record)

    def query(self, vector, top_k=10, filters=None, trace=None, **kwargs):
        self.validate_query_vector(vector, top_k)
        results = []
        for record in self.records.values():
            metadata = record.get("metadata", {})
            if filters and not all(metadata.get(key) == value for key, value in filters.items()):
                continue
            score = sum(left * right for left, right in zip(vector, record["vector"]))
            results.append({**deepcopy(record), "score": score, "metadata": dict(metadata)})
        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


@pytest.fixture
def factory():
    """每个测试使用独立注册表。"""
    class TestFactory(VectorStoreFactory):
        _PROVIDERS = {}

    return TestFactory


def settings_for(provider):
    """保持其他配置，仅替换存储提供者。"""
    settings = load_settings()
    return replace(settings, vector_store=replace(settings.vector_store, provider=provider))


def test_factory_store_roundtrip_preserves_record_contract(factory):
    """从工厂创建存储后，写入数据可通过查询接口取回。"""
    factory.register_provider("memory", FakeVectorStore)
    settings = settings_for("MEMORY")
    store = factory.create(settings)
    store.upsert([
        {"id": "a", "vector": [1.0, 0.0], "text": "甲", "metadata": {"source": "a.pdf"}},
        {"id": "b", "vector": [0.0, 1.0], "text": "乙", "metadata": {"source": "b.pdf"}},
    ])

    assert store.settings is settings
    assert store.query([1.0, 0.0], top_k=1) == [
        {"id": "a", "vector": [1.0, 0.0], "text": "甲", "metadata": {"source": "a.pdf"}, "score": 1.0}
    ]


@pytest.mark.parametrize("records", [
    [], ["bad"], [{"vector": [1.0]}], [{"id": "a"}],
    [{"id": "a", "vector": "bad"}], [{"id": "a", "vector": []}],
])
def test_upsert_rejects_invalid_records(records):
    """写入入口拒绝缺失字段和无效向量。"""
    with pytest.raises(ValueError):
        FakeVectorStore().upsert(records)


@pytest.mark.parametrize("vector,top_k", [([], 1), ("bad", 1), ([1.0], 0), ([1.0], -1), ([1.0], 1.5)])
def test_query_rejects_invalid_parameters(vector, top_k):
    """查询入口拒绝空向量、错误类型和非正整数 TopK。"""
    with pytest.raises(ValueError):
        FakeVectorStore().query(vector, top_k=top_k)


def test_unknown_store_is_reported(factory):
    """未接入的数据库不会被工厂静默选中。"""
    with pytest.raises(ValueError, match="Unsupported VectorStore provider: 'missing'"):
        factory.create(settings_for("missing"))


def test_store_registration_requires_base_interface(factory):
    """注册的类必须提供向量存储接口。"""
    with pytest.raises(ValueError, match="must inherit from BaseVectorStore"):
        factory.register_provider("bad", object)


def test_store_construction_failure_names_provider(factory):
    """存储初始化失败时定位具体提供者。"""
    class BrokenStore(FakeVectorStore):
        def __init__(self, **kwargs):
            raise RuntimeError("failed")

    factory.register_provider("broken", BrokenStore)
    with pytest.raises(RuntimeError, match="Failed to instantiate VectorStore provider 'broken'"):
        factory.create(settings_for("broken"))


def test_repeated_upsert_updates_record_instead_of_duplicating():
    """同一 ID 更新后，通过查询只得到一条最新记录。"""
    store = FakeVectorStore()
    store.upsert([{"id": "a", "vector": [1.0], "text": "旧内容"}])
    update = [{"id": "a", "vector": [2.0], "text": "新内容"}]
    store.upsert(update)
    store.upsert(update)
    assert store.query([1.0]) == [
        {"id": "a", "vector": [2.0], "text": "新内容", "score": 2.0, "metadata": {}}
    ]


def test_filtering_happens_before_top_k_limit():
    """不符合过滤条件的记录不能占用 TopK 名额。"""
    store = FakeVectorStore()
    store.upsert([
        {"id": "a", "vector": [3.0], "metadata": {"source": "a.pdf"}},
        {"id": "b", "vector": [2.0], "metadata": {"source": "b.pdf"}},
        {"id": "c", "vector": [1.0], "metadata": {"source": "b.pdf"}},
    ])
    results = store.query([1.0], top_k=2, filters={"source": "b.pdf"})
    assert [item["id"] for item in results] == ["b", "c"]


def test_empty_store_returns_no_results():
    """合法查询遇到空存储时正常返回空列表。"""
    assert FakeVectorStore().query([1.0]) == []
