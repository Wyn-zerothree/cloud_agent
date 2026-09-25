"""测量 L1 语义缓存的真实命中延迟。

分两部分，默认只跑免费部分：

  --dump     仅列出 qa_semantic_cache 当前内容（零成本）
  --measure  实测延迟。会花 DashScope embedding 调用：
             精确命中 0 次、语义命中 1 次、未命中 1 次、HTTP 端到端 1 次。

之所以要分开，是因为 `SemanticCache.get_cache()` 是两级判定：
先按 question_norm 做标量精确匹配（不调 embedding），
只有精确未命中才付 embedding + 向量检索。两者延迟差一个量级，
混在一起报一个数没有意义。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# 必须在 import requests/httpx 之前设：本机开着本地代理（127.0.0.1:26561），
# requests/httpx 会读系统代理把*本机*请求也塞进去，代理无法路由 localhost 就回 404，
# 表现为「后端整站 404」的假象。DashScope 的 embedding 走代理还会撞
# SSLCertVerificationError（代理 MITM，certifi 不认它的 CA）。
# no_proxy='*' 让所有出站走直连——本机与 DashScope 都直接可达。
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "agent"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from infra.cache import COLLECTION_NAME, semantic_cache  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "cache_latency.txt"
lines: list[str] = []


def emit(text: str = "") -> None:
    """打到屏幕，并写进落盘的结果文件。"""
    print(text)
    lines.append(text)


def say(text: str = "") -> None:
    """只打屏幕，不落盘。

    缓存内容的 dump 只用来挑基准条目和人工核对，属于运行期状态，
    不进结果文件——它会把当时缓存里的碎片条目和降级答案一并固化下来。
    """
    print(text)


async def dump() -> list[dict]:
    await semantic_cache.initialize()
    if not semantic_cache.available:
        emit("SemanticCache 不可用")
        return []
    rows = semantic_cache._client.query(
        collection_name=COLLECTION_NAME,
        filter="id >= 0",
        output_fields=["question", "question_norm", "scope", "user_id", "answer"],
        limit=100,
    )
    say(f"qa_semantic_cache 共 {len(rows)} 条（仅屏幕，不落盘）：")
    for r in rows:
        say(
            f"  [scope={r.get('scope')} user={r.get('user_id') or '-'}] "
            f"{r.get('question_norm')!r}"
        )
        say(f"      answer: {str(r.get('answer'))[:80]}...")
    return rows


async def timed_get(query: str, user_id: str) -> tuple[dict | None, float]:
    t0 = time.perf_counter()
    hit = await semantic_cache.get_cache(query, user_id)
    return hit, time.perf_counter() - t0


async def measure() -> None:
    rows = await dump()
    emit()

    # 挑一条公共域条目做精确命中基准；没有就用脚本自己写一条
    public = [r for r in rows if r.get("scope") == "public"]
    if public:
        exact_q = public[0]["question"]
        scope_note = "复用已有公共域条目"
    else:
        exact_q = "测试用的缓存延迟探针问句"
        emit("缓存中没有公共域条目，先写入一条作为探针（花 1 次 embedding）")
        await semantic_cache.set_cache(
            query=exact_q, response="这是探针写入的占位答案。" * 5, user_id=None
        )
        scope_note = "脚本新写入"
    emit(f"精确命中基准问句（{scope_note}）：{exact_q!r}")
    emit()

    emit("=" * 70)
    emit("1) 精确命中（question_norm 标量匹配，不调 embedding）")
    emit("=" * 70)
    for i in range(3):
        hit, dt = await timed_get(exact_q, "user_1001")
        emit(f"  第 {i + 1} 次: {dt * 1000:.1f} ms  level={hit['level'] if hit else None}")

    emit()
    emit("=" * 70)
    emit("2) 语义命中（改写问句，需 embedding + 向量检索）")
    emit("=" * 70)
    paraphrase = exact_q + " 请再说一遍"
    for i in range(3):
        hit, dt = await timed_get(paraphrase, "user_1001")
        sim = 1.0 - hit["distance"] if hit else None
        emit(
            f"  第 {i + 1} 次: {dt * 1000:.1f} ms  "
            f"level={hit['level'] if hit else None}  cosine={sim if sim is None else round(sim, 4)}"
        )

    emit()
    emit("=" * 70)
    emit("3) 未命中（embedding + 向量检索，但无条目过阈值）")
    emit("=" * 70)
    for i in range(2):
        hit, dt = await timed_get("量子计算与云原生存储的拓扑关系是什么", "user_1001")
        emit(f"  第 {i + 1} 次: {dt * 1000:.1f} ms  hit={'是' if hit else '否'}")

    emit()
    emit("=" * 70)
    emit("4) HTTP 端到端首字延迟（打真实后端 /api/chat，SSE 流）")
    emit("=" * 70)
    try:
        import httpx

        for i in range(3):
            t0 = time.perf_counter()
            ttft = None
            status = None
            body_head = ""
            async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
                async with client.stream(
                    "POST",
                    "http://127.0.0.1:5000/api/chat",
                    json={
                        "query": exact_q,
                        "user_id": "user_1001",
                        "session_id": "latency_probe",
                    },
                ) as resp:
                    status = resp.status_code
                    async for chunk in resp.aiter_bytes():
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        if len(body_head) < 200:
                            body_head += chunk.decode("utf-8", "replace")
                        if ttft is not None:
                            break
            if ttft:
                emit(f"  第 {i + 1} 次: HTTP {status} 首字 {ttft * 1000:.1f} ms")
                emit(f"      首分片: {body_head[:120]!r}")
            else:
                emit(f"  第 {i + 1} 次: HTTP {status} 无输出")
    except Exception as exc:
        emit(f"  HTTP 测量失败（后端未启动？）: {type(exc).__name__}: {exc}")

    emit()
    emit("=" * 70)
    emit("汇总")
    emit("=" * 70)
    emit("  精确命中 = 纯 Milvus 标量查询，不含 embedding 网络往返")
    emit("  语义命中 = 精确未命中后追加 embedding + IVF_FLAT/COSINE 检索")
    emit("  端到端 = HTTP + 上述查询 + SSE 首分片")


async def cold() -> None:
    """跑一条真实冷问题走完整 Agent 推理链路，作为命中延迟的对照基线。

    会真花 LLM 钱（一条 ≈¥0.003），跑完把落进缓存的探针条目删掉。
    """
    import httpx

    await semantic_cache.initialize()
    probe_q = "ECS 的安全组和网络 ACL 有什么区别？"
    emit("=" * 70)
    emit(f"5) 冷启动对照（不透缓存，走完整 Agent 推理）：{probe_q!r}")
    emit("=" * 70)

    normalized = semantic_cache._normalize(probe_q)
    safe = normalized.replace('"', '\\"')
    ttfts: list[float] = []

    for i in range(3):
        t0 = time.perf_counter()
        ttft = None
        text = ""
        async with httpx.AsyncClient(timeout=180.0, trust_env=False) as client:
            async with client.stream(
                "POST",
                "http://127.0.0.1:5000/api/chat",
                json={"query": probe_q, "user_id": "latency_probe", "session_id": "cold_probe"},
            ) as resp:
                emit(f"  第 {i + 1} 次: HTTP {resp.status_code}")
                async for chunk in resp.aiter_bytes():
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text += chunk.decode("utf-8", "replace")
        total = time.perf_counter() - t0
        ttfts.append(ttft)
        emit(f"    首字 {ttft * 1000:.0f} ms / 全文 {total:.2f} s / {len(text)} 字符")

        # 每轮都清掉被准入策略写进缓存的条目，否则下一轮就成了命中
        try:
            semantic_cache._client.delete(
                collection_name=COLLECTION_NAME,
                filter=f'question_norm == "{safe}"',
            )
            semantic_cache._client.flush(COLLECTION_NAME)
        except Exception as exc:
            emit(f"    清理失败，下一轮会变成命中: {type(exc).__name__}: {exc}")

    emit(f"  冷启动首字区间: {min(ttfts) * 1000:.0f}–{max(ttfts) * 1000:.0f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true", help="跑缓存命中延迟实测（约 5 次 embedding）")
    ap.add_argument("--cold", action="store_true", help="额外跑一条冷问题做对照（真花 LLM 钱，≈¥0.003）")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    async def run() -> None:
        if args.measure or args.cold:
            await measure()
        else:
            await dump()
            emit()
            emit("（仅 dump，未测量。加 --measure 跑缓存实测、--cold 加冷启动对照）")
        if args.cold:
            emit()
            await cold()

    asyncio.run(run())

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n结果已写入 {OUT}")


if __name__ == "__main__":
    main()
