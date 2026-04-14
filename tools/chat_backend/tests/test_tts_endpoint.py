import base64
import json
import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from tools.chat_backend import serve


class _FakeUrlopenResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestTTSEndpoint(unittest.IsolatedAsyncioTestCase):
    async def test_api_tts_returns_audio_mpeg(self) -> None:
        requests: list[dict] = []

        def _fake_urlopen(request, timeout=0):
            requests.append(
                {
                    "url": request.full_url,
                    "payload": json.loads(request.data.decode("utf-8")),
                    "timeout": timeout,
                }
            )
            return _FakeUrlopenResponse(
                {"audioContent": base64.b64encode(b"fake-mp3").decode("ascii")}
            )

        with patch.dict(os.environ, {"GOOGLE_TTS_API_KEY": "tts-key"}, clear=False), patch(
            "tools.chat_backend.serve.urllib.request.urlopen",
            new=_fake_urlopen,
        ):
            response = await serve.api_tts(
                serve.TTSRequest(text="Hello from Cheri", voice="en-US-Neural2-C")
            )

        self.assertEqual(response.media_type, "audio/mpeg")
        self.assertEqual(response.body, b"fake-mp3")
        self.assertEqual(len(requests), 1)
        self.assertIn("key=tts-key", requests[0]["url"])
        self.assertEqual(requests[0]["payload"]["voice"]["name"], "en-US-Neural2-C")
        self.assertEqual(requests[0]["payload"]["voice"]["languageCode"], "en-US")
        self.assertEqual(requests[0]["payload"]["audioConfig"]["audioEncoding"], "MP3")

    def test_synthesize_google_tts_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(HTTPException) as ctx:
                serve._synthesize_google_tts("Hello")

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("not configured", str(ctx.exception.detail))

    def test_synthesize_google_tts_rejects_text_over_limit(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_TTS_API_KEY": "tts-key"}, clear=False):
            with self.assertRaises(HTTPException) as ctx:
                serve._synthesize_google_tts("x" * (serve._MAX_TTS_TEXT_CHARS + 1))

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("exceeds", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()
