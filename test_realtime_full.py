"""完整音频测试 - 发送和接收并行"""
import asyncio
import json
import sys
import wave
import base64
import numpy as np

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/v1/realtime"
WAV = sys.argv[2] if len(sys.argv) > 2 else "chinese_16k.wav"
MODEL = "Qwen/Qwen3-ASR-0.6B-hf"

with wave.open(WAV, "rb") as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

chunk_ms = 100
chunk_size = int(16000 * chunk_ms / 1000)
total_chunks = (len(audio) + chunk_size - 1) // chunk_size

print(f"音频: {WAV}  {len(audio)/16000:.1f}s  {chunk_ms}ms/块  {total_chunks}块")
print(f"目标: {URL}\n")


async def main():
    async with websockets.connect(
        URL, ping_interval=20, ping_timeout=60, close_timeout=30, max_size=2**24
    ) as ws:
        msg = json.loads(await ws.recv())
        print(f"[<-] {msg['type']}")

        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
        print(f"[->] session.update: {MODEL}\n")

        # 先 commit 启动 streaming
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("[->] commit (streaming 开始)")

        # ---- 发送任务 ----
        async def sender():
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size].astype(np.int16)
                b64 = base64.b64encode(chunk.tobytes()).decode()
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.append", "audio": b64})
                )
                await asyncio.sleep(chunk_ms / 1000)
            # 发送完毕
            await ws.send(
                json.dumps({"type": "input_audio_buffer.commit", "final": True})
            )
            print("\n[->] commit (final, 发送完毕)\n")

        # ---- 接收任务 ----
        async def receiver():
            print("转写:\n")
            while True:
                msg = json.loads(await ws.recv())
                t = msg["type"]
                if t == "transcription.delta":
                    print(msg["delta"], end="", flush=True)
                elif t == "transcription.done":
                    print(f"\n\n=== 完整文本 ===\n{msg['text']}")
                    return
                elif t == "error":
                    print(f"\n错误: {msg['error']}")
                    return

        # 并行运行
        await asyncio.gather(sender(), receiver())


asyncio.run(main())
