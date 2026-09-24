# -*- coding: utf-8 -*-
"""用 LLM 判官重判两条检索路，替代裸串匹配的判分口径。

为什么要换口径：向量路返回 ~1200 字散文、图谱路返回 ~30 字结构化裸值，同一份
ground truth 串没法同时适配两边——典型如 s08，文档写「1200 万 PPS」而图谱存
「max_pps_millions=12」，裸串匹配必然误判。改由判官按参考答案判断「检索到的
内容是否足以得出该答案」。

判官只看 (问题, 参考答案, 检索内容)，不看是哪条路，避免偏向。
空检索内容直接判否，不浪费调用。

用法：
  python data/eval/judge_hybrid.py --dry-run
  python data/eval/judge_hybrid.py --output data/eval/results/hybrid_judged.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent"
sys.path.insert(0, str(AGENT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import core.workflow.state  # noqa: E402,F401

BASE = Path(__file__).resolve().parent
CTX = BASE / "results" / "hybrid_contexts.jsonl"
REFS = BASE / "judge_refs.jsonl"
DEFAULT_OUT = BASE / "results" / "hybrid_judged.jsonl"

PROMPT = """你是检索质量判官。判断下面【检索内容】是否包含足以回答【用户问题】的信息。

【用户问题】
{question}

【参考答案要点】
{ref}

判定标准：
- 只要检索内容里能读出参考答案要点（允许单位换算、结构化字段、同义表述），判「是」。
- 检索内容为空、只沾边但答不出要点、或答的是别的对象，判「否」。
- 不要依赖你自己的领域知识补全，只能依据检索内容本身。

只输出一个字：是 或 否。"""


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def verdict(text: str) -> bool | None:
    m = re.search(r"[是否]", text or "")
    if not m:
        return None
    return m.group(0) == "是"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    ctxs = load_jsonl(CTX)
    refs = {r["id"]: r["ref"] for r in load_jsonl(REFS)}
    missing = [c["id"] for c in ctxs if c["id"] not in refs]

    n_calls = 2 * len(ctxs)
    print(f"对照项 {len(ctxs)} 条，判官调用约 {n_calls} 次")
    if missing:
        print(f"⚠️ 缺参考答案: {missing}")

    if args.dry_run:
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv
        load_dotenv(AGENT_DIR / ".env")
        llm = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                         model=os.getenv("MODEL", "qwen-plus"),
                         base_url=os.getenv("BASE_URL"), temperature=0)
        sample = ctxs[0]
        msg = PROMPT.format(question=sample["q"], ref=refs[sample["id"]])
        print(f"[dry-run] 判官 prompt 约 {len(msg)} 字，"
              f"样例(s01)向量原文 {len(sample['vector_text'])} 字")
        print(f"[dry-run] 预计 {n_calls} 次 x ≈{700} input tokens ≈ ¥0.035")
        return

    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv
    load_dotenv(AGENT_DIR / ".env")
    llm = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                     model=os.getenv("MODEL", "qwen-plus"),
                     base_url=os.getenv("BASE_URL"), temperature=0, max_retries=2)

    out, t0 = [], time.time()
    for i, c in enumerate(ctxs, 1):
        rec = {"id": c["id"], "cat": c["cat"], "q": c["q"]}
        for side, key in (("vector", "vector_text"), ("graph", "graph_text")):
            text = c.get(key) or ""
            if not text.strip():
                rec[f"{side}_hit"] = False
                rec[f"{side}_raw"] = "(空)"
                continue
            prompt = PROMPT.format(question=c["q"], ref=refs.get(c["id"], ""))
            try:
                resp = llm.invoke([("system", prompt), ("user", f"【检索内容】\n{text[:6000]}")])
                raw = getattr(resp, "content", str(resp))
                rec[f"{side}_hit"] = verdict(raw)
                rec[f"{side}_raw"] = raw.strip()[:40]
            except Exception as e:
                rec[f"{side}_hit"] = None
                rec[f"{side}_raw"] = f"ERR {type(e).__name__}: {e}"[:80]
        out.append(rec)
        mark = lambda k: "✓" if rec.get(k) is True else ("✗" if rec.get(k) is False else "?")
        print(f"[{i:2d}/{len(ctxs)}] {c['id']} 向量={mark('vector_hit')} 图谱={mark('graph_hit')} | {c['q']}")

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(out)
    v = sum(1 for r in out if r.get("vector_hit") is True)
    g = sum(1 for r in out if r.get("graph_hit") is True)
    rec_ = [r for r in out if r.get("graph_hit") is True and r.get("vector_hit") is False]
    reg = [r for r in out if r.get("vector_hit") is True and r.get("graph_hit") is False]
    both = [r for r in out if r.get("vector_hit") is not True and r.get("graph_hit") is not True]

    print("\n" + "=" * 68)
    print(f"向量路命中: {v}/{n} = {v/n*100:.1f}%")
    print(f"图谱路命中: {g}/{n} = {g/n*100:.1f}%")
    print(f"并集命中  : {n - len(both)}/{n} = {(n-len(both))/n*100:.1f}%")
    print("-" * 68)
    print(f"图谱补回向量漏召: {len(rec_)}/{n}")
    for r in rec_:
        print(f"   + {r['id']} [{r['cat']}] {r['q']}")
    print(f"仅向量命中(图谱漏): {len(reg)}")
    for r in reg:
        print(f"   - {r['id']} [{r['cat']}] {r['q']}")
    print(f"两路都漏: {len(both)}")
    for r in both:
        print(f"   . {r['id']} [{r['cat']}] {r['q']}")
    print("=" * 68)
    print(f"\n耗时 {time.time()-t0:.1f}s，写入 {out_path}")


if __name__ == "__main__":
    main()
