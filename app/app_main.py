import sys
import os

# 同时把 app/ 和 agent/ 加入 sys.path，使本文件可从仓库任意目录启动
APP_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(os.path.dirname(APP_DIR), "agent")
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, APP_DIR)

# Windows 中文环境控制台默认 GBK，本项目日志含 emoji，会触发 UnicodeEncodeError
# 使进程启动即崩。统一改用 UTF-8；环境变量保证子进程（MCP stdio server）同样生效。
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from router import chat
from service.chat_service import init_agent_system

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    await init_agent_system()
    yield
    # 关闭时清理
    pass

app = FastAPI(title="Multi-Agent Cloud Service API", lifespan=lifespan)

# 配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_main:app", host="0.0.0.0", port=5000, reload=True)
