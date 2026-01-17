import asyncio
import websockets
import grpc
import grpc.aio
import pyaudio
import os
import json
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()
YANDEX_API_KEY = os.getenv("YANDEX_STT_API_KEY")
PORT = int(os.getenv("PORT", 8000))

import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_pb2_grpc as stt_service_pb2_grpc


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 8192


RECOGNITION_CONFIG = stt_pb2.StreamingOptions(
    recognition_model=stt_pb2.RecognitionModelOptions(
        audio_format=stt_pb2.AudioFormatOptions(
            raw_audio=stt_pb2.RawAudio(
                audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                sample_rate_hertz=RATE,
                audio_channel_count=CHANNELS,
            )
        ),
        text_normalization=stt_pb2.TextNormalizationOptions(
            text_normalization=stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
            profanity_filter=True,
        ),
        language_restriction=stt_pb2.LanguageRestrictionOptions(
            restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
            language_code=["ru-RU", "en-US"],
        ),
        audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
    )
)


async def handle_websocket(websocket):
    print("🔌 New client connected")
    try:
        cred = grpc.ssl_channel_credentials()
        channel = grpc.aio.secure_channel("stt.api.cloud.yandex.net:443", cred)
        stub = stt_service_pb2_grpc.RecognizerStub(channel)

        async def generate_requests():
            print("📡 Sending session options to Yandex STT...")
            yield stt_pb2.StreamingRequest(session_options=RECOGNITION_CONFIG)
            async for message in websocket:
                print(f"📡 Received {len(message)} bytes of audio from client")
                if len(message) == 0:
                    print("⚠️ Empty audio chunk received")
                elif all(b == 0 for b in message):
                    print("🔇 All-zero (silent) audio chunk detected")
                else:
                    # Decode first few samples to check for real audio
                    import struct
                    samples = struct.unpack(f"<{len(message)//2}h", message)
                    non_zero = [s for s in samples[:10] if s != 0]
                    if len(non_zero) == 0:
                        print("📉 First 10 samples are zero (possibly silent)")
                    else:
                        print(f"🔊 First non-zero samples: {non_zero}")
                yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=message))

        responses = stub.RecognizeStreaming(
            generate_requests(),
            metadata=(("authorization", f"Api-Key {YANDEX_API_KEY}"),),
        )
        print("🟢 gRPC stream established to Yandex STT")

        async for response in responses:
            # Получаем имена всех полей
            field_names = [fd.name for fd, _ in response.ListFields()]
            print("📩 Full STT response fields: " + str(field_names))

            if "partial" in field_names:
                if response.partial.alternatives:
                    text = response.partial.alternatives[0].text
                    await websocket.send(json.dumps({"type": "partial", "text": text}))
                    print(f"📤 Sent partial: {text}")
                else:
                    print("🟡 Partial response, but no alternatives")

            elif "final" in field_names:
                print("🔍 Processing 'final' response...")
                if response.final.alternatives:
                    text = response.final.alternatives[0].text
                    print(f"📝 Final text ready: '{text}'")
                    if text.strip():
                        try:
                            await websocket.send(json.dumps({"type": "final", "text": text}))
                            print(f"📤 Sent final: {text}")
                        except Exception as e:
                            print(f"❌ Failed to send final: {e}")
                    else:
                        print("🟡 Final response, but text is empty")
                else:
                    print("🟡 Final response, but no alternatives")
                    print(f"Raw final: {response.final}")

            elif "final_refinement" in field_names:
                if response.final_refinement.normalized_text.alternatives:
                    text = response.final_refinement.normalized_text.alternatives[0].text
                    if text.strip():
                        try:
                            await websocket.send(json.dumps({"type": "final", "text": text}))
                            print(f"📤 Sent final (refined): {text}")
                        except Exception as e:
                            print(f"❌ Failed to send refined: {e}")
                    else:
                        print("🟡 Final refinement, but text is empty")
                else:
                    print("🟡 Final refinement, but no alternatives")

            else:
                print("🟡 No transcription field in response. All fields: " + str(field_names))

    except websockets.exceptions.ConnectionClosed:
        print("connection closed")
    except Exception as e:
        print(f"client error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    print(f"🚀 Starting server on 0.0.0.0:{PORT}...")
    try:
        server = await websockets.serve(
            handle_websocket,
            "0.0.0.0",
            PORT,
            ping_interval=None,
            origins=None,
            compression=None
        )
        print(f"✅ Server started ws://0.0.0.0:{PORT}")
        print(f"Server is serving: {server.is_serving()}")
        print(f"API key loaded: {'yes' if YANDEX_API_KEY else 'no'}")
        print(f"api key loaded: {'true' if YANDEX_API_KEY else 'false'}")

        # Add periodic check
        while server.is_serving():
            print("💡 Server is still serving...")
            await asyncio.sleep(5)

        print("🔚 Server is no longer serving.")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔧 DEBUG: main.py started")
    import json
    import sys
    if sys.platform == "darwin":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    asyncio.run(main())
