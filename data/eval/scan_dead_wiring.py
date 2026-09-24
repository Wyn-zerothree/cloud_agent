# -*- coding: utf-8 -*-
"""扫「定义了但没接线 / 调了不存在的方法」这类死代码。

起因：MemoryManager 的 finalize_session / background_extract 定义了却无调用点，
而 agent/main.py 调了一个不存在的 extract_and_save_preferences。这类问题静态可查。

判据（都是启发式，输出要人过一眼，不要当结论）：
A. 定义了但全仓既没有被 `x.name()` 调用、也没有被 `name()` 直接调用的函数/方法
B. 调用了 `recv.name()` 但全仓找不到任何名为 name 的定义，且 recv 不是导入的模块
   （用于抓「方法名写错」，比如 extract_and_save_preferences）

用法: python data/eval/scan_dead_wiring.py [输出文件]
"""
import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {"__pycache__", "test", "tests", ".git", "node_modules",
             "venv", ".venv", "results", "data"}
# 三方/框架/内置对象的方法名，不在本仓定义
BLACKLIST = {
    "invoke", "ainvoke", "stream", "astream", "batch", "abatch",
    "similarity_search", "similarity_search_with_score", "add_texts",
    "aembed_query", "embed_query", "aembed_documents", "embed_documents",
    "compile", "add_node", "add_edge", "add_conditional_edges", "set_entry_point",
    "tool", "run", "arun", "ainvoke", "bind", "bind_tools", "with_config",
    # 常用容器/内置
    "append", "extend", "insert", "get", "set", "pop", "keys", "values", "items",
    "update", "clear", "copy", "sort", "join", "split", "strip", "lower", "upper",
    "replace", "format", "encode", "decode", "startswith", "endswith", "findall",
    "search", "match", "group", "sub", "dumps", "loads", "load", "dump", "read",
    "readline", "read_text", "write", "write_text", "mkdir", "exists", "resolve",
    "relative_to", "as_posix", "rglob", "glob", "iterdir", "walk",
    "getenv", "setdefault", "getLogger", "basicConfig", "add_argument",
    "parse_args", "add_middleware", "include_router", "add_field", "add_index",
    "create_schema", "create_collection", "has_collection", "prepare_index_params",
    "delete", "query", "flush", "close", "ping", "aclose", "values",
    "from_url", "from_llm", "fromkeys", "most_common", "count",
    "create_task", "gather", "sleep", "to_thread", "iscoroutinefunction",
    "session", "driver", "verify_connectivity", "data", "error", "warning",
    "info", "debug", "exception", "abspath", "dirname", "basename", "cursor",
    "execute", "fetchall", "fetchone", "commit", "rollback", "get_instance",
    "override", "values", "keys", "Semaphore", "StreamHandler", "ArgumentParser",
    "reconfigure", "uuid4", "list_collections", "utility", "time", "post",
}


def iter_py():
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        yield p


def imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add(a.asname or a.name)
    return names


def scan():
    # 必须两遍：第一遍收全部定义，第二遍才判调用。单遍会因为文件遍历顺序不同
    # 把「先扫到的文件里的调用」误判成未定义（defined 还在累积中）。
    trees = []
    for path in iter_py():
        try:
            trees.append((path.relative_to(ROOT).as_posix(),
                          ast.parse(path.read_text(encoding="utf-8"))))
        except Exception:
            continue

    defined: dict[str, list[str]] = defaultdict(list)
    for rel, tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("__"):
                    defined[node.name].append(f"{rel}:{node.lineno}")

    attr_calls: dict[str, list[str]] = defaultdict(list)
    name_calls: dict[str, list[str]] = defaultdict(list)
    unresolved: list[tuple[str, str, str]] = []
    for rel, tree in trees:
        imported = imported_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                # 裸引用也算被用到（例如把函数当参数传给 add_conditional_edges）
                name_calls[node.id].append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Attribute):
                # 属性引用（不只调用）也算用到，这样才能覆盖 @property
                attr_calls[node.attr].append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    name_calls[f.id].append(f"{rel}:{node.lineno}")
                elif isinstance(f, ast.Attribute):
                    recv = f.value.id if isinstance(f.value, ast.Name) else None
                    if (f.attr not in defined and f.attr not in BLACKLIST
                            and recv is not None and recv not in imported):
                        unresolved.append((rel, node.lineno, f"{recv}.{f.attr}"))
    return defined, attr_calls, name_calls, unresolved


def main() -> None:
    defined, attr_calls, name_calls, unresolved = scan()
    lines = []

    lines.append("=" * 72)
    lines.append("A. 定义了但全仓零调用点（既无 x.name() 也无 name()）")
    lines.append("=" * 72)
    orphan = sorted(n for n in defined if not attr_calls.get(n) and not name_calls.get(n))
    for name in orphan:
        lines.append(f"  {name:34s} def@ {defined[name]}")
    lines.append(f"  —— 共 {len(orphan)} 个")

    lines.append("")
    lines.append("=" * 72)
    lines.append("B. recv.name() 但全仓无 name 定义，且 recv 不是导入的模块")
    lines.append("=" * 72)
    for rel, lineno, expr in sorted(unresolved):
        lines.append(f"  {expr:44s} @ {rel}:{lineno}")
    if not unresolved:
        lines.append("  (无)")

    lines.append("")
    lines.append(f"统计: 定义 {len(defined)} / A类 {len(orphan)} / B类 {len(unresolved)}")

    OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results" / "scan_dead_wiring.txt"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written -> {OUT}  ({len(orphan)} orphan, {len(unresolved)} unresolved)")


if __name__ == "__main__":
    main()
