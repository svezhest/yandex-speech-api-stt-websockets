import asyncio
import websockets
import wave
import struct
import numpy as np

def read_wav_chunked(file_path, target_sample_rate=16000):
    with wave.open(file_path, 'rb') as wav:
        # Параметры оригинального файла
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()  # 2 bytes для PCM16
        frames = wav.readframes(wav.getnframes())
        
        # Конвертируем в numpy
        if width == 2:
            # PCM16
            audio = np.frombuffer(frames, dtype=np.int16)
        else:
            raise ValueError("Only PCM16 WAV files supported")

        # Если stereo — оставляем только левый канал
        if channels == 2:
            audio = audio[::2]

        # Ресемплинг, если нужно
        if sample_rate != target_sample_rate:
            import scipy.signal
            number_of_samples = int(len(audio) * target_sample_rate / sample_rate)
            audio = scipy.signal.resample(audio, number_of_samples)

        # Преобразуем обратно в int16
        audio = np.clip(audio, -32768, 32767).astype(np.int16)

        # Пакуем в байты
        return audio.tobytes()

async def send_audio():
    uri = "ws://localhost:8769"
    wav_path = "../russian-speech-sample.wav"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to server")
            
            # Читаем и конвертируем WAV
            audio_data = read_wav_chunked(wav_path)
            chunk_size = 8192  # как в CHUNK
            
            # Отправляем чанками
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                if len(chunk) > 0:
                    print(f"📤 Sending audio chunk: {len(chunk)} bytes")
                    await websocket.send(chunk)
                    await asyncio.sleep(0.1)  # имитация потока

            print("CloseOperation")
            await asyncio.sleep(2)
    except Exception as e:
        print(f"❌ Connection failed: {e}")

asyncio.run(send_audio())