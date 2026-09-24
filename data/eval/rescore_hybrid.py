# -*- coding: utf-8 -*-
"""把两条路的**检索原文**落盘，供反复重算判分口径用（几乎零成本）。

为什么需要：
  第一版 run_hybrid_eval.py 只存了 Cypher，没存检索到的原文，导致判分口径
  （ground truth 写成了源文档的措辞，如 "3 个弹性网卡"，而图谱返回的是裸值 "3"）
  一旦发现有问题就得整批重跑付费的 LLM 部分。
  实际图谱侧的原文可以由存下来的 Cypher **免费重执行**得到；向量侧只需重新
  embedding（30 次 ≈ ¥0.0004）。所以把原文落盘后，之后调判分口径完全免费。

产出 data/eval/results/hybrid_contexts.jsonl：
  {"id","vector_text","graph_text","graph_cypher","graph_error","vector_error"}
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent"
sys.path.insert(0, str(AGENT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import core.workflow.state  # noqa: E402,F401  解开循环导入，必须先于 tools.*

BASE = Path(__file__).resolve().parent
RUNS = BASE / "results" / "hybrid_runs.jsonl"
OUT = BASE / "results" / "hybrid_contexts.jsonl"
TOP_K = 3


def main():
    runs = [json.loads(l) for l in open(RUNS, encoding="utf-8") if l.strip()]

    from tools.vector_tool import _get_milvus_store
    import tools.graph_tool as gt

    store = _get_milvus_store()
    gt._get_graph_chain()          # 触发连接 + schema 反射
    graph = gt._graph_instance

    records = []
    for i, r in enumerate(runs, 1):
        # 向量侧：只重做检索，不再调用任何 LLM
        try:
            hits = store.similarity_search(r["q"], k=TOP_K)
            v_text = "\n\n".join(d.page_content for d in hits)
            v_err = None
        except Exception as e:
            v_text, v_err = "", f"{type(e).__name__}: {e}"

        # 图谱侧：重执行存下来的 Cypher，完全免费
        g_text, g_err = "", r.get("graph_error")
        cypher = r.get("graph_cypher") or ""
        if cypher:
            try:
                g_text = json.dumps(graph.query(cypher), ensure_ascii=False, default=str)
                g_err = None
            except Exception as e:
                g_text, g_err = "", f"{type(e).__name__}: {e}"

        records.append({
            "id": r["id"], "cat": r["cat"], "q": r["q"],
            "must_contain": r["must_contain"],
            "vector_text": v_text, "graph_text": g_text,
            "graph_cypher": cypher, "graph_error": g_err, "vector_error": v_err,
        })
        print(f"[{i:2d}/{len(runs)}] {r['id']} 向量原文 {len(v_text):5d} 字  图谱原文 {len(g_text):5d} 字"
              f"{'  [图谱失败]' if g_err else ''}")

    with open(OUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n检索原文已落盘: {OUT}")
    print("以后调整判分口径只需重读此文件，零 API 成本。")


if __name__ == "__main__":
    main()
