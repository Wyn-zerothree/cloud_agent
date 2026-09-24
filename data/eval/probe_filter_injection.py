# -*- coding: utf-8 -*-
"""零成本探测：Milvus 过滤表达式里字符串拼接 user_id 是否可被注入绕过。

要回答两件事：
1. 未转义拼接（long_term.py 的写法）能否用一个带引号的 user_id 读出别人的行？
2. 转义 `"` -> `\\"`（cache.py 的写法）到底有没有用？Milvus 的表达式解析器认不认
   这种转义。如果不认，那 cache.py 的"防护"是假的。

不调付费接口，只用 pymilvus 的 query。
用法: python data/eval/probe_filter_injection.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent"
sys.path.insert(0, str(AGENT_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(AGENT_DIR / ".env")

from pymilvus import MilvusClient  # noqa: E402

OUT = Path(__file__).parent / "results" / "probe_filter_injection.txt"

client = MilvusClient(
    uri=f"http://{os.getenv('MILVUS_HOST', 'localhost')}:{os.getenv('MILVUS_PORT', '19530')}"
)

PAYLOAD = 'x" or user_id != "x'
COLLECTIONS = ["qa_semantic_cache", "long_term_memory"]

lines = [f"payload = {PAYLOAD!r}", ""]
for coll in COLLECTIONS:
    lines.append(f"--- {coll} ---")
    baseline = client.query(collection_name=coll, filter='user_id == "user_1001"',
                            output_fields=["user_id"])
    lines.append(f"  基线 user_id == \"user_1001\"  -> {len(baseline)} 行")
    lines.append(f"  全表                               -> "
                 f"{len(client.query(collection_name=coll, filter='id >= 0', output_fields=['user_id']))} 行")

    # 1) long_term.py 的写法：原样拼进去
    raw = f'user_id == "{PAYLOAD}"'
    try:
        rows = client.query(collection_name=coll, filter=raw, output_fields=["user_id"])
        lines.append(f"  未转义 {raw}  -> {len(rows)} 行")
    except Exception as exc:
        lines.append(f"  未转义 {raw}  -> {type(exc).__name__}: {str(exc)[:100]}")

    # 2) cache.py 的写法：escape_double_quote 再拼
    escaped_payload = PAYLOAD.replace('"', '\\"')
    esc = f'user_id == "{escaped_payload}"'
    try:
        rows = client.query(collection_name=coll, filter=esc, output_fields=["user_id"])
        lines.append(f"  已转义 {esc}  -> {len(rows)} 行")
    except Exception as exc:
        lines.append(f"  已转义 {esc}  -> {type(exc).__name__}: {str(exc)[:100]}")
    lines.append("")

text = "\n".join(lines)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(text, encoding="utf-8")
print(f"written -> {OUT}")
