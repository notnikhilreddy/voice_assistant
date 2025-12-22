import asyncio
import websockets
import pyaudio
import numpy as np

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

async def main():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        print("Connected to server.")
        
        # This is a placeholder for a real audio stream.
        # It generates silent audio chunks and sends them.
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("Streaming audio...")
        try:
            while True:
                data = stream.read(CHUNK)
                await websocket.send(data)
                # In a real app, we would also be receiving data here.
                # response = await websocket.recv()
        except websockets.exceptions.ConnectionClosed:
            print("Connection to server closed.")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Client stopped by user.")
