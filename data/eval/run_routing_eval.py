# -*- coding: utf-8 -*-
"""意图路由准确率评测。

测的是 OrchestratorAgent.route() 的 5 分类决策：
  product_agent / billing_agent / promotion_agent / recommendation_agent / finops_agent_trigger

注意：finops_agent_trigger 在图上落到 billing_agent 节点（靠 metadata.is_finops_workflow
区分），所以这里从 (next_agent, is_finops_workflow) 反推 5 分类标签，而不是比图节点名。

用法：
  python data/eval/run_routing_eval.py --dry-run          # 不调 LLM，只校验接线
  python data/eval/run_routing_eval.py --output data/eval/results/routing_runs.jsonl
"""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
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
SET_PATH = BASE / "routing_set.jsonl"
DEFAULT_OUT = BASE / "results" / "routing_runs.jsonl"

CLASSES = ["product_agent", "billing_agent", "promotion_agent",
           "recommendation_agent", "finops_agent_trigger"]


def load_set():
    items = []
    with open(SET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def decision_from_state(next_agent: str, meta: dict) -> str:
    """把 (节点, finops 标记) 还原成 5 分类决策。"""
    if next_agent == "billing_agent" and meta.get("is_finops_workflow"):
        return "finops_agent_trigger"
    return next_agent


async def run_one(orch, item, sem):
    async with sem:
        state = {
            "messages": [("user", item["q"])],
            "user_id": "eval_user",
            "session_id": f"eval_{item['id']}",
            "memory_context": "",
            "next_agent": "",
            "metadata": {},
        }
        t0 = time.time()
        try:
            out = await orch.route(state)
            decision = decision_from_state(out.get("next_agent", ""), out.get("metadata", {}))
            err = None
        except Exception as e:
            decision, err = None, f"{type(e).__name__}: {e}"
        return {
            "id": item["id"],
            "q": item["q"],
            "kind": item.get("kind", "scored"),
            "gold": item.get("gold"),
            "decision": decision,
            "correct": (decision == item["gold"]) if item.get("gold") else None,
            "note": item.get("note", ""),
            "error": err,
            "latency_s": round(time.time() - t0, 2),
        }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只校验接线，不调 LLM")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()

    items = load_set()
    scored = [i for i in items if i.get("kind") == "scored"]
    print(f"测试集: {len(items)} 条（计分 {len(scored)} / 闲聊 {sum(1 for i in items if i.get('kind')=='chitchat')} "
          f"/ 边界 {sum(1 for i in items if i.get('kind')=='boundary')}）")
    print(f"gold 分布: {dict(Counter(i['gold'] for i in scored))}")

    if args.dry_run:
        print("\n[dry-run] 不调用 LLM。检查模块导入与状态构造…")
        from agents.orchestrator import OrchestratorAgent  # noqa: F401
        probe = {"messages": [("user", "你好")], "user_id": "u", "session_id": "s",
                 "memory_context": "", "next_agent": "", "metadata": {}}
        print(f"[dry-run] 状态构造 OK，keys = {sorted(probe)}")
        print("[dry-run] 说明: 真实运行会调用 OrchestratorAgent().route()，"
              f"共 {len(items)} 次 qwen-plus 调用")
        return

    from agents.orchestrator import OrchestratorAgent
    orch = OrchestratorAgent()
    sem = asyncio.Semaphore(args.concurrency)

    t0 = time.time()
    results = await asyncio.gather(*(run_one(orch, it, sem) for it in items))
    elapsed = time.time() - t0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    errors = [r for r in results if r["error"]]
    good = [r for r in results if r["correct"] is not None and not r["error"]]
    n_correct = sum(1 for r in good if r["correct"])

    print("\n" + "=" * 64)
    print(f"计分题准确率: {n_correct}/{len(good)} = {n_correct/len(good)*100:.1f}%"
          if good else "计分题: 无有效记录")
    print(f"调用错误: {len(errors)}   总耗时: {elapsed:.1f}s")
    print("=" * 64)

    per_class = defaultdict(lambda: [0, 0])
    for r in good:
        per_class[r["gold"]][1] += 1
        per_class[r["gold"]][0] += int(r["correct"])
    print("\n分类别准确率:")
    for c in CLASSES:
        hit, tot = per_class.get(c, [0, 0])
        if tot:
            print(f"  {c:24s} {hit}/{tot}  {hit/tot*100:5.1f}%")

    fixed = [f"{r['gold']}->{r['decision']}" for r in good if not r["correct"]]
    if fixed:
        print("\n误分类样本:")
        for r in good:
            if not r["correct"]:
                print(f"  {r['id']} 期望 {r['gold']:22s} 实际 {str(r['decision']):22s} | {r['q']}")

    print("\n域外输入（无 gold，仅记录去向，不计入准确率）:")
    for r in results:
        if r["kind"] in ("chitchat", "boundary"):
            print(f"  [{r['kind']:8s}] {r['q'][:34]:36s} -> {r['decision']}")

    if errors:
        print("\n错误明细:")
        for r in errors[:10]:
            print(f"  {r['id']}: {r['error']}")

    print(f"\n原始记录已写入: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
