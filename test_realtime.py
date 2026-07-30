"""Qwen3-ASR 实时转写测试工具

用法:
    python test_realtime.py --url ws://localhost:8000/v1/realtime  # 本地测试
    python test_realtime.py --url wss://xxx.modal.run/v1/realtime  # Modal 测试
    python test_realtime.py --generate --duration 5                 # 先生成测试音频
"""

import argparse
import asyncio
import json
import struct
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MODEL_NAME = "Qwen/Qwen3-ASR-0.6B-hf"


# ============================================================
# 1. 生成测试 wav 文件
# ============================================================
def generate_test_wav(path: str, duration_s: float = 5.0, freq: float = 440):
    """生成简单的正弦波 wav 文件用于连通性测试。
    注意：这不是语音，ASR 不会转写出有意义的文字，
    仅用于验证 WebSocket 连接和实时流是否通畅。
    """
    samples = np.sin(
        2 * np.pi * freq * np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    )
    samples = (samples * 0.5 * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())

    print(f"Generated: {path} ({duration_s}s, {freq}Hz sine)")


def generate_speech_like_wav(path: str, duration_s: float = 5.0):
    """生成模拟语音的 wav：静音 + 不同频率的交替
    比纯正弦波更像语音信号，用于测试静音/有声切换。
    """
    total_samples = int(duration_s * SAMPLE_RATE)
    samples = np.zeros(total_samples, dtype=np.float32)

    for i in range(0, total_samples, SAMPLE_RATE // 4):
        end = min(i + SAMPLE_RATE // 2, total_samples)
        freq = 300 + (i // (SAMPLE_RATE // 4)) * 200  # 变化频率模拟语音
        t = np.arange(end - i) / SAMPLE_RATE
        samples[i:end] = (
            np.sin(2 * np.pi * freq * t) * (1 - np.exp(-t * 10)) * 0.5
        )

        # 每段之间加短暂静音
        gap_start = end
        gap_end = min(gap_start + SAMPLE_RATE // 20, total_samples)
        samples[gap_start:gap_end] = 0

    samples = (samples * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())

    print(f"Generated: {path} ({duration_s}s, speech-like)")


# ============================================================
# 2. 读取 wav 并模拟实时流推送
# ============================================================
def load_wav_pcm16(path: str) -> np.ndarray:
    """读取 wav，返回 int16 的 numpy 数组。"""
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {wf.getframerate()}Hz")
        data = wf.readframes(wf.getnframes())
    return np.frombuffer(data, dtype=np.int16)


async def realtime_test(url: str, wav_path: str, chunk_ms: int = 100):
    """模拟实时流：读取 wav 文件，按 chunk 切分，通过 WebSocket 发送。"""
    try:
        import websockets
    except ImportError:
        print("请安装: pip install websockets")
        return

    import base64

    audio = load_wav_pcm16(wav_path)
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)  # 每块的采样数
    total_chunks = (len(audio) + chunk_size - 1) // chunk_size

    print(f"音频: {wav_path}")
    print(f"时长: {len(audio) / SAMPLE_RATE:.1f}s, 块大小: {chunk_ms}ms, 共 {total_chunks} 块")
    print(f"连接到: {url}\n")

    async with websockets.connect(url) as ws:
        # 等待 session.created
        msg = json.loads(await ws.recv())
        if msg["type"] == "session.created":
            print(f"[<-] session.created: {msg['id']}")

        # 验证模型
        await ws.send(json.dumps({"type": "session.update", "model": MODEL_NAME}))
        print(f"[->] session.update: {MODEL_NAME}")

        # 发送 non-final commit，触发 streaming 开始
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("[->] commit (start streaming)")

        # 切分并发送音频
        print(f"\n发送音频中 ({chunk_ms}ms/块)...")
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size].astype(np.int16)
            b64 = base64.b64encode(chunk.tobytes()).decode()
            await ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": b64})
            )
            # 模拟实时间隔
            await asyncio.sleep(chunk_ms / 1000)

        # 发送 final commit
        await ws.send(
            json.dumps({"type": "input_audio_buffer.commit", "final": True})
        )
        print("[->] commit (final)\n")

        # 接收转写结果
        print("转写结果:")
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "transcription.delta":
                print(msg["delta"], end="", flush=True)
            elif msg["type"] == "transcription.done":
                print(f"\n\n完整文本: {msg['text']}")
                if msg.get("usage"):
                    u = msg["usage"]
                    print(f"用量: prompt={u['prompt_tokens']} "
                          f"completion={u['completion_tokens']}")
                break
            elif msg["type"] == "error":
                print(f"\n错误: {msg['error']}")
                break


# ============================================================
# 3. 静音/语音交替测试
# ============================================================
async def silence_alternating_test(url: str, cycle_count: int = 3):
    """交替发送静音和有声片段，测试静音时是否复读。"""
    try:
        import websockets
    except ImportError:
        print("请安装: pip install websockets")
        return

    import base64

    SEGMENT_S = 5  # 5秒 = Qwen3ASR 分段大小
    SEGMENT_SIZE = SEGMENT_S * SAMPLE_RATE
    CHUNK_MS = 100
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)

    async with websockets.connect(url) as ws:
        msg = json.loads(await ws.recv())
        assert msg["type"] == "session.created"

        await ws.send(json.dumps({"type": "session.update", "model": MODEL_NAME}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        for cycle in range(cycle_count):
            # --- 有声段：440Hz 正弦波 ---
            print(f"\n[Cycle {cycle + 1}] 有声段 {SEGMENT_S}s")
            audio = (
                np.sin(2 * np.pi * 440 * np.arange(SEGMENT_SIZE) / SAMPLE_RATE)
                * 0.5
                * 32767
            ).astype(np.int16)

            for i in range(0, SEGMENT_SIZE, CHUNK_SIZE):
                chunk = audio[i : i + CHUNK_SIZE]
                b64 = base64.b64encode(chunk.tobytes()).decode()
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.append", "audio": b64})
                )
                await asyncio.sleep(CHUNK_MS / 1000)

            # --- 静音段：全零 ---
            print(f"[Cycle {cycle + 1}] 静音段 {SEGMENT_S}s")
            silence = np.zeros(SEGMENT_SIZE, dtype=np.int16)
            for i in range(0, SEGMENT_SIZE, CHUNK_SIZE):
                chunk = silence[i : i + CHUNK_SIZE]
                b64 = base64.b64encode(chunk.tobytes()).decode()
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.append", "audio": b64})
                )
                await asyncio.sleep(CHUNK_MS / 1000)

        await ws.send(
            json.dumps({"type": "input_audio_buffer.commit", "final": True})
        )

        print("\n转写:")
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "transcription.delta":
                print(msg["delta"], end="", flush=True)
            elif msg["type"] == "transcription.done":
                print(f"\n完整: {msg['text']}")
                break
            elif msg["type"] == "error":
                print(f"\n错误: {msg['error']}")
                break


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR 实时转写测试")
    parser.add_argument("--url", default="ws://localhost:8000/v1/realtime")
    parser.add_argument("--wav", help="测试的 wav 文件路径")
    parser.add_argument("--generate", action="store_true", help="生成测试 wav")
    parser.add_argument("--duration", type=float, default=10, help="生成音频时长(秒)")
    parser.add_argument("--chunk-ms", type=int, default=100, help="实时分块大小(ms)")
    parser.add_argument("--silence-test", action="store_true", help="静音交替测试")
    parser.add_argument("--cycles", type=int, default=3, help="交替次数")
    args = parser.parse_args()

    if args.generate:
        wav = args.wav or "test_speech.wav"
        generate_speech_like_wav(wav, args.duration)
        return

    if args.silence_test:
        asyncio.run(silence_alternating_test(args.url, args.cycles))
        return

    if not args.wav:
        # 默认生成一个
        wav = "test_speech.wav"
        if not Path(wav).exists():
            generate_speech_like_wav(wav, args.duration)
    else:
        wav = args.wav

    asyncio.run(realtime_test(args.url, wav, args.chunk_ms))


if __name__ == "__main__":
    main()
