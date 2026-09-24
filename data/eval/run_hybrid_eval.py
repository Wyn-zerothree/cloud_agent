# -*- coding: utf-8 -*-
"""向量路 vs 图谱路 检索对照。

判定口径（重要）：两条路都只看**检索到的原始内容**，不看 LLM 合成后的答案。
  - 向量路：Milvus top-3 chunk（与 vector_tool.py 生产参数一致，k=3）
  - 图谱路：GraphCypherQAChain 生成的 Cypher 直接执行，取原始返回行
理由：GraphCypherQAChain 的 answer LLM 可以靠参数化记忆补答，用它判分会高估图谱路。
两边都算 raw retrieval 才是同口径对比。

命中判定：检索到的文本里出现该题 must_contain 中任意一个串（大小写不敏感）。

用法：
  python data/eval/run_hybrid_eval.py --dry-run
  python data/eval/run_hybrid_eval.py --output data/eval/results/hybrid_runs.jsonl
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent"
sys.path.insert(0, str(AGENT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# 必须先于任何 agents.* 导入：core.workflow.state 先入 sys.modules 才能解开
# agents.orchestrator <-> core.workflow.graph_manager 的循环导入（生产 main.py 同理）。
import core.workflow.state  # noqa: E402,F401

BASE = Path(__file__).resolve().parent
SET_PATH = BASE / "spec_query_set.jsonl"
DEFAULT_OUT = BASE / "results" / "hybrid_runs.jsonl"
TOP_K = 3


def load_set():
    items = []
    with open(SET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def is_hit(text: str, needles) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(n.lower() in low for n in needles)


def vector_retrieve(store, q: str) -> str:
    res = store.similarity_search_with_score(q, k=TOP_K)
    return "\n\n".join(doc.page_content for doc, _ in res)


def _graph_rows_text(graph, cypher: str) -> str:
    rows = graph.query(cypher)
    return json.dumps(rows, ensure_ascii=False, default=str)


def _chain_with_steps():
    """拿生产用的 GraphCypherQAChain，并打开 intermediate_steps 以取回生成的 Cypher。"""
    import tools.graph_tool as gt
    chain = gt._get_graph_chain()
    try:
        chain.return_intermediate_steps = True
    except Exception as e:
        raise RuntimeError(f"无法打开 return_intermediate_steps: {e}")
    return chain, gt._graph_instance


def _extract_cypher(result) -> str:
    steps = result.get("intermediate_steps") or []
    for step in steps:
        if isinstance(step, dict):
            for key in ("query", "cypher", "cypher_query"):
                if step.get(key):
                    return step[key]
    return ""


def graph_retrieve(chain, graph, q: str):
    """返回 (原始行文本, cypher, 错误)。"""
    try:
        result = chain.invoke({"query": q})
    except Exception as e:
        return "", "", f"chain: {type(e).__name__}: {e}"
    cypher = _extract_cypher(result)
    if not cypher:
        # 没拿到 Cypher，退而用 QA 结果本身（会在记录里标注）
        return "", "", "no-intermediate-steps"
    try:
        return _graph_rows_text(graph, cypher), cypher, None
    except Exception as e:
        return "", cypher, f"exec: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    items = load_set()
    print(f"对照集: {len(items)} 条，分类 {dict((k, sum(1 for i in items if i['cat']==k)) for k in dict.fromkeys(i['cat'] for i in items))}")

    if args.dry_run:
        print("\n[dry-run] 不调 LLM。检查数据通路…")
        from tools.vector_tool import _get_milvus_store
        store = _get_milvus_store()
        # 向量路要一次 embedding 才能确认，这里用已有集合的 count 代替，避免任何付费调用
        from pymilvus import Collection, connections
        connections.connect(alias="dry", host=os.getenv("MILVUS_HOST", "localhost"),
                            port=os.getenv("MILVUS_PORT", "19530"))
        c = Collection("cloud_product_docs", using="dry")
        print(f"[dry-run] Milvus cloud_product_docs = {c.num_entities} 条 (生产 k={TOP_K})")
        import tools.graph_tool as gt
        chain = gt._get_graph_chain()
        print(f"[dry-run] 图谱 chain 就绪，schema 长度 = {len(gt._graph_instance.schema or '')}")
        print(f"[dry-run] 真实运行将产生约 {len(items)*2} 次 qwen-plus 调用"
              f"（Cypher 生成 + 答案合成各 {len(items)} 次）"
              f" + {len(items)} 次 text-embedding-v2")
        return

    from tools.vector_tool import _get_milvus_store
    store = _get_milvus_store()
    chain, graph = _chain_with_steps()

    records, t0 = [], time.time()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 增量落盘：网络抖动时进程可能被长挂起或中断，逐条追加保证已完成的部分不丢。
    sink = open(out_path, "w", encoding="utf-8")
    for i, item in enumerate(items, 1):
        q = item["q"]
        needles = item["must_contain"]

        vt, verr = "", None
        try:
            vt = vector_retrieve(store, q)
        except Exception as e:
            verr = f"{type(e).__name__}: {e}"

        gt_text, cypher, gerr = graph_retrieve(chain, graph, q)

        v_hit, g_hit = is_hit(vt, needles), is_hit(gt_text, needles)
        rec = {
            "id": item["id"], "cat": item["cat"], "q": q,
            "must_contain": needles,
            "vector_hit": v_hit, "graph_hit": g_hit,
            "graph_cypher": cypher, "graph_error": gerr, "vector_error": verr,
        }
        records.append(rec)
        sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        sink.flush()
        flag = "补回" if (g_hit and not v_hit) else ("都中" if (v_hit and g_hit) else
                                                   ("都漏" if not (v_hit or g_hit) else "向量中"))
        print(f"[{i:2d}/{len(items)}] {item['id']} {item['cat']:6s} "
              f"向量={'✓' if v_hit else '✗'} 图谱={'✓' if g_hit else '✗'} {flag}  | {q}",
              flush=True)
        if gerr:
            print(f"         图谱错误: {gerr[:100]}", flush=True)

    elapsed = time.time() - t0
    sink.close()

    n = len(records)
    v_hits = sum(r["vector_hit"] for r in records)
    g_hits = sum(r["graph_hit"] for r in records)
    recovered = [r for r in records if r["graph_hit"] and not r["vector_hit"]]
    regressed = [r for r in records if r["vector_hit"] and not r["graph_hit"]]
    both_miss = [r for r in records if not r["vector_hit"] and not r["graph_hit"]]

    print("\n" + "=" * 68)
    print(f"向量路命中: {v_hits}/{n} = {v_hits/n*100:.1f}%")
    print(f"图谱路命中: {g_hits}/{n} = {g_hits/n*100:.1f}%")
    print(f"并集命中  : {len(recovered)+v_hits}/{n} = {(len(recovered)+v_hits)/n*100:.1f}%")
    print("-" * 68)
    print(f"图谱补回向量漏召: {len(recovered)}/{n}")
    if recovered:
        for r in recovered:
            print(f"   + {r['id']} [{r['cat']}] {r['q']}")
    print(f"仅向量命中(图谱漏): {len(regressed)}")
    if regressed:
        for r in regressed:
            print(f"   - {r['id']} [{r['cat']}] {r['q']}")
    print(f"两路都漏: {len(both_miss)}")
    if both_miss:
        for r in both_miss:
            print(f"   . {r['id']} [{r['cat']}] {r['q']}")
    print("=" * 68)

    errs = [r for r in records if r["graph_error"]]
    if errs:
        print(f"\n图谱路异常 {len(errs)} 条:")
        for r in errs[:10]:
            print(f"  {r['id']}: {r['graph_error'][:120]}")

    print(f"\n耗时 {elapsed:.1f}s，原始记录已写入: {out_path}")


if __name__ == "__main__":
    main()
