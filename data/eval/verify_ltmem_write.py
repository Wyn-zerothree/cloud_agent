# -*- coding: utf-8 -*-
"""验证长期记忆写路径：造一段含明确偏好的对话 → background_extract → 回读。

验证的是修复后的真实调用链（memory_manager.background_extract），不是绕开它直
接调 save_memory。跑完会清掉自己造的数据：Redis 键 + Milvus 里该测试 user 的行，
不在用户库里留残留。

付费调用：1 次 qwen-plus（偏好抽取，约 600 in / 60 out）
        + 2 次 text-embedding-v2（1 次检索去重 + 1 次写入），合计 < ¥0.01。

用法: python data/eval/verify_ltmem_write.py
"""
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent"
sys.path.insert(0, str(AGENT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(AGENT_DIR / ".env")

import core.workflow.state  # noqa: E402,F401  (解循环导入，与生产入口一致)
from config import get_settings  # noqa: E402
from core.memory import MemoryManager  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

USER = "probe_lt_verify"
SESSION = "s_probe_verify"
OUT = Path(__file__).parent / "results" / "verify_ltmem.txt"

CONVO = [
    {"role": "user", "content": "我在北京，想给一个内部管理系统选一台云服务器，预算别太高。"},
    {"role": "assistant", "content": "建议从通用型 g8a 规格族起步，2 核 4G 足够内部系统。"},
    {"role": "user", "content": "回答尽量短一点，别给我贴一堆参数表。另外我不用命令行，要那种点几下就能买好的方式。"},
    {"role": "assistant", "content": "好的。控制台选 g8a.large，按量付费即可。"},
]


async def main() -> None:
    lines = []
    settings = get_settings()
    mem = MemoryManager(
        redis_url=settings.redis_url,
        redis_ttl=settings.redis_ttl,
        milvus_host=settings.milvus_host,
        milvus_port=settings.milvus_port,
        milvus_api_key=settings.milvus_api_key,
        embedding_api_key=settings.dashscope_api_key,
    )
    await mem.initialize()
    lines.append(f"short_term.available = {mem.short_term.available}")
    lines.append(f"long_term.available  = {mem.long_term.available}")

    llm = ChatOpenAI(**settings.get_model_config(), temperature=0)

    # 清掉上一次可能残留的
    await mem.short_term.clear(USER, SESSION)
    mem.long_term._client.delete(collection_name="long_term_memory",
                                 filter=f'user_id == "{USER}"')

    await mem.short_term.save_messages(USER, SESSION, CONVO)
    lines.append(f"写入 Redis 消息数 = {len(await mem.short_term.get_messages(USER, SESSION))}")

    lines.append("调用 background_extract ...")
    new_items = await mem.background_extract(USER, SESSION, llm)
    lines.append(f"background_extract 返回 = {new_items}")

    # 回读：先确认 Redis 没被清（background_extract 不应清会话）
    still = await mem.short_term.get_messages(USER, SESSION)
    lines.append(f"会话后 Redis 消息数 = {len(still)} (应仍为 4，未被清)")

    prefs = await mem.load_preferences(USER)
    lines.append(f"load_preferences 回读 {len(prefs)} 条 = {prefs}")

    # Milvus 里实际行数（按 user 过滤）
    rows = mem.long_term._client.query(
        collection_name="long_term_memory",
        filter=f'user_id == "{USER}"',
        output_fields=["user_id", "content", "memory_type"],
    )
    lines.append(f"Milvus 该 user 行数 = {len(rows)}")
    for r in rows:
        lines.append(f"   {r}")

    # 清理（Milvus 的 delete 要 flush 之后才不再被 query 看到，否则会留残留）
    await mem.short_term.clear(USER, SESSION)
    mem.long_term._client.delete(collection_name="long_term_memory",
                                 filter=f'user_id == "{USER}"')
    mem.long_term._client.flush("long_term_memory")
    left = mem.long_term._client.query(
        collection_name="long_term_memory",
        filter=f'user_id == "{USER}"',
        output_fields=["user_id"],
    )
    lines.append(f"清理后该 user 行数 = {len(left)}")

    await mem.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
