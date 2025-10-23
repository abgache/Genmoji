from scripts.time_log import time_log_module as tlm
from fastapi import FastAPI
import torch
import torch.optim as optim
from scripts.prompt_enhancer import enhance_prompt
from pydantic import BaseModel
import base64
import asyncio
import uvicorn
import socket
import threading
import io
import base64

class API:
    class GenerateRequest(BaseModel):
        prompt: str

    def __init__(self, pipe, port=12470, tcp_port=5555):
        self.port = port
        self.pipe = pipe
        self.tcp_host = "127.0.0.1"
        self.tcp_port = tcp_port
        self.app = FastAPI()
        self.tcp_response = None

        # Création d'un serveur uvicorn mais non lancé
        self.config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="info")
        self.server = uvicorn.Server(self.config)

        # Routes
        self.app.post("/generate")(self.generate)
        self.app.post("/kill")(self.kill)

    def tcp_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.tcp_host, self.tcp_port))
            s.listen()
            while True:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024)
                    self.prompt = data.decode()
                    print(f"Received prompt: {data.decode()}")
                    # Remplace par ton vrai générateur
                    image_bytes = b"FakeImageBytesForTesting"
                    self.tcp_response = base64.b64encode(image_bytes).decode()
                    conn.sendall(self.tcp_response.encode())

    async def generate(self, req: GenerateRequest):
        self.tcp_response = None
        # Envoie prompt au TCP
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.tcp_host, self.tcp_port))
            s.sendall(req.prompt.encode())

        timeout = 600
        start = asyncio.get_event_loop().time()
        self.enhanced_prompt = enhance_prompt(enhance_prompt(self.prompt))
        self.buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")  # ou "JPEG" si tu veux
        self.image_bytes = buffer.getvalue()
        self.tcp_response = base64.b64encode(self.pipe(self.enhanced_prompt, num_inference_steps=25).images[0]).decode("utf-8")

        return {"image_b64": self.tcp_response}

    async def kill(self):
        # Stoppe uniquement FastAPI
        await self.server.shutdown()
        return {"status": "API stopped"}

    def serve(self):
        # Lancer TCP en thread
        threading.Thread(target=self.tcp_server, daemon=True).start()
        # Lancer FastAPI
        self.server.run()
