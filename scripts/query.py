"""D7：查询命令入口，复用 HybridSearch/CoreReranker，只返回证据不生成回答。

退出码：0 查询成功（可以无命中）；1 运行失败；2 参数、配置或初始化错误。
--help 不初始化资源；没有数据库/集合时不创建空集合、不调用模型。
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.hybrid_search import create_hybrid_search
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.query_service import create_query_components
from src.core.query_engine.reranker import create_core_reranker
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.core.settings import DEFAULT_SETTINGS_PATH, load_settings, resolve_path
from src.core.trace import TraceCollector, TraceContext
from src.core.trace.snapshots import snapshot_results
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("top-k 必须是正整数")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """沿用上游查询参数；data-dir 与昨天摄取入口使用同一布局。"""
    parser = argparse.ArgumentParser(description="从知识库检索相关片段及来源，不生成最终回答。")
    parser.add_argument("--query", "-q", required=True, help="查询问题，可使用 D1 的 key:value 过滤语法")
    parser.add_argument("--collection", "-c", help="集合名；显式参数优先于问题里的 collection，默认 default")
    parser.add_argument("--top-k", type=_positive_int, help="最终返回条数；省略时使用检索/重排配置")
    parser.add_argument("--config", default=str(DEFAULT_SETTINGS_PATH), help="YAML 配置文件")
    parser.add_argument("--no-rerank", action="store_true", help="强制关闭重排，不初始化评分模型")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示关键词、两路召回、融合、重排及降级信息")
    parser.add_argument("--data-dir", help="与摄取相同的隔离数据根目录，同时定位 Chroma 和 BM25")
    return parser.parse_args(argv)


def _print_results(results, title: str = "检索结果") -> None:
    """有意输出命中的正文摘要及来源；不输出向量、Key 或原始服务错误。"""
    print(f"{title}：{len(results)} 条")
    for index, result in enumerate(results, 1):
        print(f"  [{index}] score={result.score:.6f} id={result.chunk_id}")
        print(f"      来源：{result.metadata.get('source_path', '')}")
        page = result.metadata.get("page_num")
        if page is not None:
            print(f"      页码提示：{page}")
        print(f"      正文：{result.text.replace(chr(10), ' ')[:200]}")


def _build_components(settings, collection: str, index_dir: str | Path = "data/db/bm25"):
    """CLI 保留原入口，实际构造复用与 MCP 相同的 query_service。"""
    components = create_query_components(settings, collection, index_dir)
    return components.hybrid_search, components.reranker, components.vector_store


def _run_query(hybrid, reranker, query: str, top_k: int | None, collection: str, verbose: bool, trace=None) -> int:
    """融合候选量和最终返回量分开，重排前保留至少配置的候选数。"""
    candidate_count = max(hybrid.config.fusion_top_k, top_k or 0)
    trace_kwargs = {"trace": trace} if trace is not None else {}
    details = hybrid.search(query, top_k=candidate_count, filters={"collection": collection}, return_details=True, **trace_kwargs)
    if verbose:
        print(f"关键词：{details.processed_query.keywords}；过滤条件：{details.processed_query.filters}")
        _print_results(details.dense_results or [], "Dense 召回")
        _print_results(details.sparse_results or [], "BM25 召回")
        _print_results(details.results, "融合/过滤结果")
    if details.used_fallback:
        print(f"检索已降级：{details.dense_error or details.sparse_error}", file=sys.stderr)
    if not details.results:
        if trace is not None:
            trace.record_stage("rerank", {"method": "none", "status": "skipped", "reason": "empty_candidates"}, elapsed_ms=0.0)
            trace.metadata.update({"status": "empty", "final_results": [], "used_fallback": details.used_fallback})
        print("未找到匹配片段；请检查语料、问题及过滤条件。")
        return 0
    final_count = top_k if top_k is not None else reranker.config.top_k if reranker.is_enabled else hybrid.config.fusion_top_k
    reranked = reranker.rerank(query, details.results, top_k=final_count, **trace_kwargs)
    if trace is not None:
        trace.metadata.update({"status": "success", "final_results": snapshot_results(reranked.results),
                               "used_fallback": details.used_fallback, "rerank_fallback": reranked.used_fallback})
    if reranked.used_fallback:
        print(f"重排已回退：{reranked.fallback_reason}", file=sys.stderr)
    if verbose:
        print(f"重排后端：{reranked.reranker_type}；回退：{reranked.used_fallback}")
    _print_results(reranked.results)
    return 0


def main(argv: list[str] | None = None) -> int:
    """没有数据是正常空结果；实际查询需要配置的 Embedding 服务。"""
    args = parse_args(argv)
    try:
        if not args.query.strip():
            raise ValueError("Query cannot be blank")
        settings = load_settings(Path(args.config).resolve())
        parsed = QueryProcessor().process(args.query)
        collection = args.collection if args.collection is not None else parsed.filters.get("collection", "default")
        if not collection.strip() or collection in {".", ".."} or any(c in collection for c in "/\\"):
            raise ValueError("collection must be a single path component")
        index_dir = resolve_path("data/db/bm25")
        if args.data_dir is not None:
            data_dir = resolve_path(args.data_dir)
            settings = replace(settings, vector_store=replace(settings.vector_store,
                               persist_directory=str(data_dir / "db" / "chroma")))
            index_dir = data_dir / "db" / "bm25"
        if args.no_rerank:
            settings = replace(settings, rerank=replace(settings.rerank, enabled=False, provider="none"))
    except Exception as exc:
        print(f"入口检查失败：{type(exc).__name__}。请检查问题、集合及配置。", file=sys.stderr)
        return 2
    trace = TraceContext(trace_type="query") if settings.observability.trace_enabled else None
    if trace is not None:
        trace.metadata.update({"query": args.query, "collection": collection, "top_k": args.top_k,
                               "source": "cli", "status": "running"})
    store = None
    started = time.monotonic()
    try:
        try:
            hybrid, reranker, store = _build_components(settings, collection, index_dir)
        except Exception as exc:
            missing = isinstance(exc, FileNotFoundError) or isinstance(exc.__cause__, FileNotFoundError)
            if trace is not None:
                trace.record_stage("initialization", {"method": "query_setup", "status": "skipped" if missing else "error",
                    "reason": "missing_collection" if missing else None, "error_type": None if missing else type(exc).__name__},
                    elapsed_ms=(time.monotonic() - started) * 1000)
                trace.metadata.update({"status": "empty" if missing else "error", "final_results": [],
                                       "error_type": None if missing else type(exc).__name__})
            if missing:
                print("数据库或集合不存在；请先用 ingest.py 摄取文档。未创建数据库或调用模型。")
                return 0
            print(f"查询初始化失败：{type(exc).__name__}", file=sys.stderr)
            return 2
        if trace is not None:
            trace.record_stage("initialization", {"method": "query_setup", "status": "success",
                "embedding_provider": settings.embedding.provider, "embedding_model": settings.embedding.model},
                elapsed_ms=(time.monotonic() - started) * 1000)
        started = time.monotonic()
        try:
            return _run_query(hybrid, reranker, args.query, args.top_k, collection, args.verbose, trace=trace)
        except Exception as exc:
            if trace is not None:
                trace.record_stage("error", {"method": "query", "status": "error", "error_type": type(exc).__name__},
                                   elapsed_ms=(time.monotonic() - started) * 1000)
                trace.metadata.update({"status": "error", "error_type": type(exc).__name__})
            print(f"查询失败：{type(exc).__name__}", file=sys.stderr)
            return 1
    finally:
        try:
            client = getattr(store, "client", None)
            if client is not None:
                client.close()
        finally:
            if trace is not None:
                trace.finish()
                TraceCollector.from_settings(settings).collect(trace)
                if args.verbose:
                    print(f"Trace：{trace.trace_id}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
