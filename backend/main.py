import asyncio
import logging
import os
import sys
import json
import struct
import math
from pathlib import Path

import grpc
import grpc.aio
import websockets
from dotenv import load_dotenv

import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_pb2_grpc as stt_service_pb2_grpc

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

YANDEX_API_KEY = os.getenv("YANDEX_STT_API_KEY")
PORT = int(os.getenv("PORT", 8000))

CHANNELS = 1
RATE = 16000

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
    logger.info("New client connected")
    channel = None
    try:
        if not YANDEX_API_KEY:
            raise ValueError("YANDEX_STT_API_KEY is not set")

        cred = grpc.ssl_channel_credentials()
        channel = grpc.aio.secure_channel("stt.api.cloud.yandex.net:443", cred)
        stub = stt_service_pb2_grpc.RecognizerStub(channel)

        async def request_generator():
            yield stt_pb2.StreamingRequest(session_options=RECOGNITION_CONFIG)
            async for message in websocket:
                if len(message) == 0:
                    logger.debug("Received empty audio chunk")
                else:
                    # Directly pass through the audio data (test client sends proper int16 data)
                    yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=message))

        async def send_result(result_type, text):
            if text and text.strip():
                try:
                    payload = json.dumps({"type": result_type, "text": text})
                    await websocket.send(payload)
                    logger.info(f"Sent {result_type}: {text}")
                except Exception as e:
                    logger.error(f"Failed to send message to client: {e}")

        responses = stub.RecognizeStreaming(
            request_generator(),
            metadata=(("authorization", f"Api-Key {YANDEX_API_KEY}"),),
        )
        logger.info("gRPC stream established")

        async for response in responses:
            if response.HasField("partial") and response.partial.alternatives:
                await send_result("partial", response.partial.alternatives[0].text)

            elif response.HasField("final") and response.final.alternatives:
                await send_result("final", response.final.alternatives[0].text)

            elif (
                response.HasField("final_refinement")
                and response.final_refinement.normalized_text.alternatives
            ):
                await send_result(
                    "final",
                    response.final_refinement.normalized_text.alternatives[0].text,
                )

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected normally")
    except grpc.aio.AioRpcError as e:
        logger.error(f"gRPC error: {e.code()} - {e.details()}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        if channel:
            await channel.close()


async def main():
    server = await websockets.serve(
        handle_websocket,
        "0.0.0.0",
        PORT,
        ping_interval=None,
        origins=None,
        compression=None,
    )

    logger.info(f"Server started on port {PORT}")
    logger.info(f"API Key loaded: {bool(YANDEX_API_KEY)}")

    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.critical(f"Server crashed: {e}", exc_info=True)
