import asyncio
import json
import sys
import os

# 初始化 Agent 和 Graph
AGENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent")
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

from core.workflow.graph_manager import AgentGraphManager
from core.memory.memory_manager import MemoryManager
from infra.cache import semantic_cache

# Global variables for graph and memory
graph = None
memory = None

async def init_agent_system():
    global graph, memory
    if graph is None:
        print("🚀 初始化 Multi-Agent 图编排...")
        graph_manager = AgentGraphManager()
        graph = graph_manager.build_graph()
        
        print("🧠 初始化 Memory 系统...")
        from config import get_settings
        settings = get_settings()
        memory = MemoryManager(
            redis_url=settings.redis_url,
            redis_ttl=settings.redis_ttl,
            milvus_host=settings.milvus_host,
            milvus_port=settings.milvus_port,
            milvus_api_key=settings.milvus_api_key,
            embedding_api_key=settings.dashscope_api_key,
        )
        await memory.initialize()
        await semantic_cache.initialize()
        print("✅ Agent 系统初始化完成！")

# 写入准入：只有纯产品咨询的回答才进公共缓存。账单/推广/推荐的回答里含用户私有数据
# （实例 ID、专属返佣链接等），写进公共域会造成跨用户泄露，因此只对该用户生效。
PUBLIC_SCOPE_AGENTS = {"product_agent"}
CACHE_MIN_ANSWER_CHARS = 40
# Milvus 的 VARCHAR max_length 以字节计，answer 字段上限 8192 字节；中文 UTF-8 占 3 字节
CACHE_MAX_ANSWER_BYTES = 7800
CACHE_MAX_QUESTION_BYTES = 1900
CACHE_REJECT_MARKERS = ("发生错误", "查询失败", "暂不可用")


async def _maybe_cache(query: str, answer: str, user_id: str, agent_name: str) -> None:
    if not answer or len(answer) < CACHE_MIN_ANSWER_CHARS:
        return
    if any(marker in answer for marker in CACHE_REJECT_MARKERS):
        return
    if len(answer.encode("utf-8")) > CACHE_MAX_ANSWER_BYTES:
        return
    if len(query.encode("utf-8")) > CACHE_MAX_QUESTION_BYTES:
        return
    if agent_name in PUBLIC_SCOPE_AGENTS:
        await semantic_cache.set_cache(query, answer)
    else:
        await semantic_cache.set_cache(query, answer, user_id=user_id)


async def _extract_memory_context(user_id: str, session_id: str, query: str) -> str:
    context_parts = []
    if memory and memory.short_term.available:
        history = await memory.short_term.get_messages(user_id, session_id)
        if history:
            recent_history = history[-10:] if len(history) > 10 else history
            context_parts.append("【近期对话历史】:")
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                context_parts.append(f"{role}: {msg['content']}")
    
    if memory and memory.long_term.available:
        prefs = await memory.long_term.retrieve_relevant(user_id, query)
        if prefs:
            context_parts.append("\n【用户长期偏好/背景】:")
            for p in prefs:
                context_parts.append(f"- {p}")
                
    return "\n".join(context_parts)

async def stream_chat(query: str, user_id: str, session_id: str):
    cache_hit = await semantic_cache.get_cache(query, user_id)
    if cache_hit:
        response_text = cache_hit["answer"]
        print(
            f"⚡ 语义缓存命中: {cache_hit['level']} distance={cache_hit['distance']:.4f} matched='{cache_hit['matched_question']}'"
        )
    else:
        print("🏃 进入 Agent 工作流推理...")
        mem_context = await _extract_memory_context(user_id, session_id, query)
        state = {
            "messages": [("user", query)],
            "user_id": user_id,
            "session_id": session_id,
            "memory_context": mem_context,
            "next_agent": "",
            "metadata": {}
        }
        config = {"configurable": {"user_id": user_id}}
        result = await asyncio.to_thread(asyncio.run, graph.ainvoke(state, config=config)) if not asyncio.iscoroutinefunction(graph.ainvoke) else await graph.ainvoke(state, config=config)
        response_text = result["messages"][-1].content
        await _maybe_cache(query, response_text, user_id, result.get("next_agent", ""))

    # 保存短时记忆
    if memory and memory.short_term.available:
        turn = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response_text},
        ]
        await memory.save_conversation(user_id, session_id, turn)
        
    # 流式返回大模型结果
    chunk_size = 5
    for i in range(0, len(response_text), chunk_size):
        chunk = response_text[i:i+chunk_size]
        yield f"data: {json.dumps({'content': chunk})}\n\n"
        await asyncio.sleep(0.02)
        
    yield f"data: {json.dumps({'done': True})}\n\n"
