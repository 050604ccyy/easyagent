"""web.py —— 纯标准库本地 Web 服务，为 Agent 提供浏览器前端。

只使用 Python 标准库（http.server / json / pathlib / sys），零第三方依赖：

- ``GET  /``          返回单页前端 index.html
- ``GET  /api/tools`` 返回已注册工具清单（JSON）
- ``POST /api/run``   执行任务，返回结构化 ReAct 轨迹（JSON）

启动方式：
    python web.py          # 默认 8000 端口
    python web.py 8080     # 指定端口

启动后浏览器访问 http://127.0.0.1:8000 即可使用。
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List

from agent import Agent
from tools import registry

# 前端页面路径（与 web.py 同目录）
INDEX_PATH = Path(__file__).resolve().parent / "index.html"

# 全局共享的 Agent 实例：本地单用户场景，串行处理请求即可。
# 与 CLI 完全共用同一内核——run() 在打印控制台日志的同时把事件流写入 last_trace。
AGENT: Agent = Agent(registry=registry)


class ReActWebHandler(BaseHTTPRequestHandler):
    """请求处理器：首页 + 两个 JSON 接口。"""

    server_version = "ReActAgent/1.0"

    # -- 响应辅助 ----------------------------------------------------------

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        """按统一口径写回响应（统一 UTF-8，禁用缓存便于调试）。"""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        """以 JSON 格式响应（ensure_ascii=False 保证中文原样输出）。"""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_html(self, html: str) -> None:
        """响应 HTML 页面。"""
        self._send_bytes(200, html.encode("utf-8"), "text/html; charset=utf-8")

    # -- 路由 --------------------------------------------------------------

    def do_GET(self) -> None:
        """页面与工具清单接口。"""
        if self.path in ("/", "/index.html"):
            self._send_html(INDEX_PATH.read_text(encoding="utf-8"))
        elif self.path == "/api/tools":
            tools: List[Dict[str, Any]] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "params": tool.params_desc,
                }
                for tool in registry.items()
            ]
            self._send_json({"tools": tools})
        else:
            self._send_json({"error": "接口不存在"}, status=404)

    def do_POST(self) -> None:
        """执行任务接口：POST /api/run，请求体 {"task": "..."}。"""
        if self.path != "/api/run":
            self._send_json({"error": "接口不存在"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "请求体不是合法 JSON"}, status=400)
            return

        task = str(payload.get("task", "")).strip()
        if not task:
            self._send_json({"error": "task 不能为空"}, status=400)
            return

        try:
            answer = AGENT.run(task)
            events = AGENT.last_trace
        except Exception as exc:  # noqa: BLE001 —— 服务端兜底，避免连接直接断掉
            self._send_json({"error": f"服务内部错误：{exc}"}, status=500)
            return
        self._send_json({"answer": answer, "events": events})

    def log_message(self, fmt: str, *args: Any) -> None:
        """精简访问日志，避免刷屏。"""
        sys.stdout.write(f"[web] {fmt % args}\n")


def main() -> None:
    """启动本地 Web 服务。"""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    try:
        server = HTTPServer(("127.0.0.1", port), ReActWebHandler)
    except OSError as exc:
        print(f"端口 {port} 被占用或无法绑定：{exc}")
        print(f"可换一个端口重试，例如：python web.py {port + 1}")
        sys.exit(1)

    print("=" * 60)
    print("ReAct Agent Web 界面已启动（纯标准库 / 离线）")
    print(f"请在浏览器打开： http://127.0.0.1:{port}")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()


if __name__ == "__main__":
    main()
