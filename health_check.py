"""连通性探测：连上 WebSocket 收到 session.created 即成功"""
import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    sys.exit("请先安装: pip install websockets")


async def check(url: str):
    try:
        async with websockets.connect(url) as ws:
            msg = json.loads(await ws.recv())
            assert msg["type"] == "session.created", f"unexpected: {msg}"
            print(f"✅ 服务正常  session_id={msg['id']}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/v1/realtime"
    asyncio.run(check(url))
