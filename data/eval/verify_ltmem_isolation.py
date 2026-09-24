# -*- coding: utf-8 -*-
"""验证 long_term.py 的 user_id 过滤转义修复：注入取值必须读不到别人的行。

走真实的 LongTermMemory.retrieve_relevant（不是裸 query），这样能同时验证
转义写法本身有没有写错。跑完删掉自己造的行 + flush，不留残留。

付费调用：约 3 次 text-embedding-v2（2 次写入 1 次检索），< ¥0.001。
用法: python data/eval/verify_ltmem_isolation.py
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

from dotenv import load_dotenv  # noqa: E402

load_dotenv(AGENT_DIR / ".env")

from config import get_settings  # noqa: E402
from core.memory.long_term import LongTermMemory  # noqa: E402

VICTIM = "probe_victim"
ATTACKER = 'x" or user_id != "x'
OUT = Path(__file__).parent / "results" / "verify_ltmem_isolation.txt"


async def main() -> None:
    settings = get_settings()
    mem = LongTermMemory(
        host=settings.milvus_host,
        port=settings.milvus_port,
        api_key=settings.milvus_api_key,
        embedding_api_key=settings.dashscope_api_key,
    )
    await mem.initialize()

    lines = [f"available = {mem.available}"]
    client = mem._client
    client.delete(collection_name="long_term_memory", filter=f'user_id == "{VICTIM}"')

    await mem.save_memory(VICTIM, "城市: 上海", memory_type="preference")
    await mem.save_memory(VICTIM, "偏好: 回答简短", memory_type="preference")
    client.flush("long_term_memory")
    rows = client.query(collection_name="long_term_memory",
                        filter=f'user_id == "{VICTIM}"', output_fields=["content"])
    lines.append(f"造数据：{VICTIM} 现有 {len(rows)} 行 {[r['content'] for r in rows]}")

    own = await mem.retrieve_relevant(VICTIM, "用户偏好习惯个性特点")
    lines.append(f"本人检索 -> {len(own)} 条 {own}（应 > 0）")

    stolen = await mem.retrieve_relevant(ATTACKER, "用户偏好习惯个性特点")
    lines.append(f"注入取值检索 -> {len(stolen)} 条 {stolen}（应为 0）")

    client.delete(collection_name="long_term_memory", filter=f'user_id == "{VICTIM}"')
    client.flush("long_term_memory")
    left = client.query(collection_name="long_term_memory",
                        filter=f'user_id == "{VICTIM}"', output_fields=["content"])
    lines.append(f"清理后 {VICTIM} 行数 = {len(left)}")

    await mem.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
