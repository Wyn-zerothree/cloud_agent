# -*- coding: utf-8 -*-
"""验证 L1 语义缓存的用户域隔离：A 的私有条目，B 拿同一句话取不到。

为什么要两级都走一遍：`SemanticCache.get_cache()` 是两级判定——先按
`question_norm` 做标量精确匹配，未命中才走 embedding + 向量检索。两处各有一个
user_id 约束（`scope == "user" and user_id == ...`，以及语义那级的 `scoped_filter`），
**只测精确那级不够**：语义那级的 filter 写漏了照样能跨用户串号，而且因为向量相似度
对同一句话必然是 1.0，一旦漏了必定命中。所以这里用同一个问句把两级都过一遍。

再加一组对照（公共域条目对 B 可见），否则「B 取不到」无法区分是隔离生效还是缓存整个坏了。

走真实的 `semantic_cache.get_cache()`，不是裸 query，这样验证的是实际跑在链路里的那段代码。
跑完删掉自己造的两行 + flush，不留残留。

付费调用：3 次 text-embedding-v2（2 次写入 + 1 次 B 的语义检索），< ¥0.001。
用法: python data/eval/verify_cache_isolation.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# 必须在 import requests/httpx 之前设：本机开着本地代理，requests/httpx 会读
# 系统代理，把 DashScope 的 embedding 请求塞进去并撞 SSLCertVerificationError。
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "agent"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from infra.cache import COLLECTION_NAME, semantic_cache  # noqa: E402

OWNER = "probe_cache_owner"
INTRUDER = "probe_cache_intruder"
PRIVATE_Q = "探针私有问句：我的实例到期时间怎么查"
PRIVATE_A = "探针私有答案：登录控制台即可查看。"
PUBLIC_Q = "探针公共问句：量子纠缠现象与云存储之间有什么关联"
PUBLIC_A = "探针公共答案：两者没有直接关联。"
OUT = Path(__file__).resolve().parent / "results" / "verify_cache_isolation.txt"

client = None


def rows_for(norm: str, user: str) -> int:
    """按 get_cache 精确那级用的条件数一遍行数。"""
    return len(
        client.query(
            collection_name=COLLECTION_NAME,
            filter=f'enabled == 1 and question_norm == "{norm}" and scope == "user" and user_id == "{user}"',
            output_fields=["answer"],
        )
    )


def public_rows(norm: str) -> int:
    """数公共域条目（scope == "public"，user_id 为空串）。"""
    return len(
        client.query(
            collection_name=COLLECTION_NAME,
            filter=f'question_norm == "{norm}" and scope == "public"',
            output_fields=["answer"],
        )
    )


def drop(norm: str, scope: str, user: str) -> None:
    client.delete(
        collection_name=COLLECTION_NAME,
        filter=f'question_norm == "{norm}" and scope == "{scope}" and user_id == "{user}"',
    )


async def main() -> None:
    global client
    await semantic_cache.initialize()
    client = semantic_cache._client

    lines = [f"available = {semantic_cache.available}"]
    if not semantic_cache.available:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines), encoding="utf-8")
        print(f"SemanticCache 不可用，已写入 {OUT}")
        return

    norm_priv = semantic_cache._normalize(PRIVATE_Q)
    norm_pub = semantic_cache._normalize(PUBLIC_Q)

    # 清掉上一轮可能的残留，保证本次数字是自己造出来的
    drop(norm_priv, "user", OWNER)
    drop(norm_pub, "public", "")

    await semantic_cache.set_cache(PRIVATE_Q, PRIVATE_A, user_id=OWNER)
    await semantic_cache.set_cache(PUBLIC_Q, PUBLIC_A, scope="public")
    client.flush(COLLECTION_NAME)

    created = rows_for(norm_priv, OWNER)
    lines.append(f"造数据：{OWNER} 的私有条目 {created} 行（应为 1）")
    lines.append(f"         公共域条目 {public_rows(norm_pub)} 行（应为 1）")
    lines.append("")

    own = await semantic_cache.get_cache(PRIVATE_Q, OWNER)
    lines.append(
        f"① 本人取自己的私有条目 -> {own or None}\n"
        f"   期望命中、且答案是私有答案：{'通过' if own and own.get('answer') == PRIVATE_A else '不通过'}"
    )

    stolen = await semantic_cache.get_cache(PRIVATE_Q, INTRUDER)
    lines.append(
        f"\n② 侵入者拿同一句话取 -> {stolen or None}\n"
        f"   期望为 None（此句与公共条目语义无关，所以走过语义那级也取不到）："
        f"{'通过' if stolen is None else '不通过——跨用户串号'}"
    )

    ctrl = await semantic_cache.get_cache(PUBLIC_Q, INTRUDER)
    lines.append(
        f"\n③ 对照：侵入者取公共域条目 -> {(ctrl or {}).get('level')}\n"
        f"   期望命中，用来排除「缓存整个坏了所以②才取不到」："
        f"{'通过' if ctrl and ctrl.get('answer') == PUBLIC_A else '不通过'}"
    )

    lines.append("")
    lines.append("机制：按 get_cache 精确那级的条件，同一条私有记录对不同用户的可见行数")
    lines.append(f"  user_id == {OWNER!r}    -> {rows_for(norm_priv, OWNER)} 行")
    lines.append(f"  user_id == {INTRUDER!r} -> {rows_for(norm_priv, INTRUDER)} 行")

    drop(norm_priv, "user", OWNER)
    drop(norm_pub, "public", "")
    client.flush(COLLECTION_NAME)
    lines.append("")
    lines.append(f"清理后：私有条目 {rows_for(norm_priv, OWNER)} 行 | 公共条目 {public_rows(norm_pub)} 行")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
