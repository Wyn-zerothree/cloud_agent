# -*- coding: utf-8 -*-
"""验证 MCP 工具 generate_ai_poster 真能出图（此前全仓库唯一没有产物、没有评测覆盖的能力）。

为什么这么写：
- **直接调用被验证的那个函数本身**（import `mcp_servers/cloud_platform_server.py` 里的
  `generate_ai_poster`），不是另写一份等价请求——这样验的是实际跑在链路里的那段代码。
- **必须把 PNG 下下来**：接口返回的是有效期约 24 小时的临时 URL，URL 本身不能当产物。
- `NO_PROXY` 要在 import requests 之前设：用户机器开着本地代理，会把 DashScope 请求塞进去
  并撞 SSLCertVerificationError。

付费调用：一次 qwen-image-2.0 生成（1536*2688）。
用法: python data/eval/verify_poster.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODULE = ROOT / "agent" / "mcp_servers" / "cloud_platform_server.py"
OUT_TXT = Path(__file__).resolve().parent / "results" / "verify_poster.txt"
OUT_PNG = Path(__file__).resolve().parent / "results" / "poster_sample.png"

PROMPT = "赛博朋克风格的服务器机房，炫酷的蓝色霓虹灯，科技感，竖屏海报风格"


def load_poster_tool():
    # cloud_platform_server 顶层 import pymysql，但海报工具不碰 MySQL；
    # 本机解释器没装 pymysql，塞个占位模块即可，不影响被验的那段代码。
    if importlib.util.find_spec("pymysql") is None:
        stub = types.ModuleType("pymysql")
        stub.cursors = types.SimpleNamespace(DictCursor=object)
        stub.connect = lambda *a, **k: None
        sys.modules["pymysql"] = stub

    spec = importlib.util.spec_from_file_location("cloud_platform_server", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # mcp.tool() 装饰后拿到的是原始函数还是包装器，取决于 SDK 版本；两种都兜住
    fn = getattr(mod, "generate_ai_poster")
    return getattr(fn, "fn", fn)


def main() -> None:
    lines = [f"prompt = {PROMPT}", ""]
    tool = load_poster_tool()

    raw = tool(PROMPT)
    lines.append(f"工具原始返回:\n  {raw[:400]}")
    lines.append("")

    try:
        data = json.loads(raw)
    except Exception as e:
        data = {"status": "error", "message": f"返回不是 JSON: {e}"}

    ok = data.get("status") == "success"
    lines.append(f"status = {data.get('status')}")

    if not ok:
        lines.append(f"message = {data.get('message')}")
        lines.append("")
        lines.append("结论：**没跑通**。README 那两处「海报」要么按 A 摊开声明，要么查权限/开通状态。")
        OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
        OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        print(f"\nwritten -> {OUT_TXT}")
        return

    d = data["data"]
    url = d["poster_url"]
    # 这个 URL 带 OSSAccessKeyId + Signature，是带签名的临时凭证，
    # 产物文件要进公开仓库，只留路径、丢掉查询串。
    lines.append(f"request_id = {d.get('request_id')}")
    lines.append(f"临时 URL   = {url.split('?')[0]}?<签名参数已移除>")
    lines.append("")

    import requests  # 放在 NO_PROXY 之后

    r = requests.get(url, timeout=120)
    r.raise_for_status()
    OUT_PNG.write_bytes(r.content)
    size_mb = len(r.content) / 1024 / 1024
    lines.append(f"已下载 -> {OUT_PNG}")
    lines.append(f"文件大小 = {size_mb:.2f} MB")
    lines.append("")
    lines.append(
        "结论：**跑通了**。generate_ai_poster 是真实可用的能力，不是桩——"
        "README 无需改动（声明已有据），简历「等 7 个后台能力」的写法也自动成立。"
    )

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten -> {OUT_TXT}")


if __name__ == "__main__":
    main()
