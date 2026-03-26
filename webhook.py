import os
import tempfile
import requests
from http.server import BaseHTTPRequestHandler


WHATSAPI_URL    = "https://meuzapi.vercel.app"
WHATSAPI_TOKEN  = os.environ["WHATSAPI_TOKEN"]
WHATSAPI_INSTANCE = os.environ["WHATSAPI_INSTANCE"]
GROQ_API_KEY    = os.environ["GROQ_API_KEY"]


def download_audio(url: str) -> tuple[bytes, str]:
    """Baixa o áudio e retorna (bytes, extensão)."""
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    ct = r.headers.get("Content-Type", "")
    ext = ".mp4" if "mp4" in ct or "mp4" in url else ".ogg"
    return r.content, ext


def transcribe_with_groq(audio_bytes: bytes, ext: str) -> str:
    """Envia o áudio para a API da Groq (Whisper Large v3) e retorna o texto."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as audio_file:
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (f"audio{ext}", audio_file, "audio/ogg")},
                data={
                    "model": "whisper-large-v3-turbo",
                    "language": "pt",
                    "response_format": "text",
                },
                timeout=60,
            )
        response.raise_for_status()
        return response.text.strip()
    finally:
        os.unlink(tmp_path)


def send_message(number: str, text: str, reply_id: str = None):
    """Envia mensagem de texto via WhatsAPI."""
    payload = {"number": number, "text": text}
    if reply_id:
        payload["replyid"] = reply_id

    requests.post(
        f"{WHATSAPI_URL}/message/sendText/{WHATSAPI_INSTANCE}",
        json=payload,
        headers={"Content-Type": "application/json", "token": WHATSAPI_TOKEN},
        timeout=15,
    )


class handler(BaseHTTPRequestHandler):
    """Handler padrão para funções serverless na Vercel com Python."""

    def do_POST(self):
        import json

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body)
        except Exception:
            self._respond(200, {"status": "ignored", "reason": "invalid json"})
            return

        event    = data.get("event", "")
        msg_data = data.get("data", {})
        message  = msg_data.get("message", {})
        from_me  = msg_data.get("key", {}).get("fromMe", False)
        msg_type = message.get("type", "")

        # Ignora tudo que não for áudio recebido
        if from_me or msg_type not in ("audio", "ptt"):
            self._respond(200, {"status": "ignored"})
            return

        number     = msg_data.get("key", {}).get("remoteJid", "")
        message_id = msg_data.get("key", {}).get("id", "")
        audio_url  = (
            message.get("audioMessage", {}).get("url")
            or message.get("url")
        )

        if not audio_url or not number:
            self._respond(200, {"status": "error", "reason": "missing data"})
            return

        try:
            audio_bytes, ext = download_audio(audio_url)
            transcription    = transcribe_with_groq(audio_bytes, ext)

            if transcription:
                reply = f"🎙️ *Transcrição:*\n\n_{transcription}_"
            else:
                reply = "⚠️ Não consegui transcrever esse áudio."

            send_message(number, reply, reply_id=message_id)
            self._respond(200, {"status": "ok"})

        except Exception as e:
            print(f"Erro: {e}")
            send_message(number, "⚠️ Erro ao tentar transcrever o áudio.")
            self._respond(200, {"status": "error", "reason": str(e)})

    def do_GET(self):
        self._respond(200, {"status": "online"})

    def _respond(self, code: int, body: dict):
        import json
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *args):
        pass  # silencia logs padrão do BaseHTTPRequestHandler
