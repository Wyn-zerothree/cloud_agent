# -*- coding: utf-8 -*-
"""零成本探测：long_term_memory 集合是否存在、有多少行、schema 是否与代码一致。

不调任何付费接口（不 embedding、不 LLM），只用 pymilvus 的元信息查询接口。
用法: python data/eval/probe_ltmem.py [输出文件]
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

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results" / "probe_ltmem.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

lines = []
try:
    from pymilvus import Collection, connections
    connections.connect(alias="probe", host=os.getenv("MILVUS_HOST", "localhost"),
                        port=os.getenv("MILVUS_PORT", "19530"))
    from pymilvus import utility
    names = utility.list_collections(using="probe")
    lines.append(f"collections: {names}")
    for name in names:
        c = Collection(name, using="probe")
        lines.append(f"- {name}: num_entities={c.num_entities} desc={c.description!r}")
        if name == "long_term_memory":
            for f in c.schema.fields:
                lines.append(f"    field {f.name} type={f.dtype} params={f.params}")
except Exception as exc:
    lines.append(f"ERROR {type(exc).__name__}: {exc}")

text = "\n".join(lines)
OUT.write_text(text, encoding="utf-8")
print(f"written -> {OUT}")
