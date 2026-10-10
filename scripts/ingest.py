"""C15：摄取命令行入口，复用 IngestionPipeline，不实现第二套处理逻辑。

退出码：0 全部成功或跳过；1 部分失败；2 全部失败或入口参数/配置错误。
--dry-run 只检查配置和列文件，不读取正文、不初始化模型或存储。
"""

import argparse
import logging
import sys
from pathlib import Path

# 允许从任意工作目录用绝对脚本路径执行，用户文件路径仍按当前目录解析。
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.settings import DEFAULT_SETTINGS_PATH, load_settings
from src.core.trace import TraceCollector, TraceContext
from src.ingestion.pipeline import IngestionPipeline, PipelineResult, SUPPORTED_EXTENSIONS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """沿用上游 --path，并兼容之前讲解中的 --file 名称。"""
    parser = argparse.ArgumentParser(description="把 PDF/Markdown 文档摄取到知识库。")
    parser.add_argument("--path", "--file", "-p", required=True, help="文件或目录；目录递归查找 PDF、MD、Markdown")
    parser.add_argument("--collection", "-c", default="default", help="知识库集合名，默认 default")
    parser.add_argument("--config", default=str(DEFAULT_SETTINGS_PATH), help="YAML 配置文件路径")
    parser.add_argument("--force", "-f", action="store_true", help="忽略成功处理记录，强制重新摄取；不自动清理旧向量")
    parser.add_argument("--dry-run", action="store_true", help="只列出候选文件，不读取正文、不调用模型、不写入存储")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示阶段进度及结果统计，不显示私有正文")
    parser.add_argument("--data-dir", help="隔离摄取数据目录；显式提供时同时覆盖 Chroma、BM25、SQLite、图片位置")
    return parser.parse_args(argv)


def discover_files(path: str | Path) -> list[Path]:
    """按后缀大小写无关地发现本地文件，稳定排序；不扫描其他格式。"""
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError("Source path does not exist")
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Supported formats: PDF, MD, Markdown")
        return [source]
    if not source.is_dir():
        raise ValueError("Source must be a file or directory")
    return sorted({file.resolve() for file in source.rglob("*") if file.is_file()
                   and file.suffix.lower() in SUPPORTED_EXTENSIONS})


def print_summary(results: list[PipelineResult], verbose: bool = False) -> None:
    """跳过单独统计，不把失败途中产生的块当作成功入库数量。"""
    skipped = sum(result.success and result.stages.get("integrity", {}).get("skipped", False) for result in results)
    successful = sum(result.success for result in results) - skipped
    failed = sum(not result.success for result in results)
    print(f"汇总：总数 {len(results)}，成功 {successful}，跳过 {skipped}，失败 {failed}")
    if verbose:
        print(f"成功新增/重写：{sum(r.chunk_count for r in results if r.success)} 块，"
              f"{sum(r.image_count for r in results if r.success)} 张图片登记")


def main(argv: list[str] | None = None) -> int:
    """解析参数后才处理文档；候选列表为空按上游视为成功 no-op。"""
    args = parse_args(argv)
    try:
        settings = load_settings(Path(args.config).resolve())
        files = discover_files(args.path)
    except Exception as exc:
        print(f"入口检查失败：{type(exc).__name__}。请检查路径、格式及 YAML 配置。", file=sys.stderr)
        return 2
    print(f"找到 {len(files)} 个候选文件；集合：{args.collection}")
    if args.dry_run:
        for file in files:
            print(file)
        print("预览完成：未读取正文、调用模型或写入数据库。")
        return 0
    if not files:
        print("没有支持的文档，无需处理。")
        return 0
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    source = Path(args.path).resolve()
    try:
        pipeline = IngestionPipeline(settings, args.collection, args.force, data_dir=args.data_dir,
                                     source_root=source if source.is_dir() else source.parent)
    except Exception as exc:
        print(f"流水线初始化失败：{type(exc).__name__}", file=sys.stderr)
        return 2
    results = []
    collector = TraceCollector.from_settings(settings)
    try:
        for number, file in enumerate(files, 1):
            print(f"[{number}/{len(files)}] {file}")
            progress = (lambda stage, step, total: print(f"  阶段 {step}/{total}：{stage}")) if args.verbose else None
            trace = TraceContext(trace_type="ingestion") if settings.observability.trace_enabled else None
            if trace is not None:
                trace.metadata["source"] = "cli"
            try:
                result = pipeline.run(file, trace=trace, on_progress=progress)
            finally:
                if trace is not None:
                    trace.finish()
                    collector.collect(trace)
            if trace is not None and args.verbose:
                print(f"  Trace：{trace.trace_id}")
            results.append(result)
            if result.success:
                print("  跳过：已有成功记录" if result.stages["integrity"]["skipped"]
                      else f"  成功：{result.chunk_count} 块，{result.image_count} 张图片登记")
            else:
                print(f"  失败：{result.error}", file=sys.stderr)
    finally:
        pipeline.close()
    print_summary(results, args.verbose)
    successful = sum(result.success for result in results)
    return 0 if successful == len(results) else 1 if successful else 2


if __name__ == "__main__":
    raise SystemExit(main())
