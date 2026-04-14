import tempfile
import unittest
from pathlib import Path
import sys
import types
from unittest.mock import patch

if "anthropic" not in sys.modules:
    sys.modules["anthropic"] = types.SimpleNamespace(Anthropic=None)
if "sentence_transformers" not in sys.modules:
    class _DummyCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["sentence_transformers"] = types.SimpleNamespace(CrossEncoder=_DummyCrossEncoder)
if "src.index_build" not in sys.modules:
    sys.modules["src.index_build"] = types.SimpleNamespace(RetrievalIndex=object)
if "fastapi" not in sys.modules:
    fastapi_mod = types.ModuleType("fastapi")

    class _DummyFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def add_middleware(self, *args, **kwargs):
            return None

        def mount(self, *args, **kwargs):
            return None

        def on_event(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def get(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        post = get
        put = get
        delete = get

    class _DummyHTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _DummyRequest:
        pass

    fastapi_mod.FastAPI = _DummyFastAPI
    fastapi_mod.HTTPException = _DummyHTTPException
    fastapi_mod.Request = _DummyRequest
    sys.modules["fastapi"] = fastapi_mod

    responses_mod = types.ModuleType("fastapi.responses")

    class _DummyResponse:
        def __init__(self, content=None, media_type=None, status_code=200):
            self.content = content
            self.media_type = media_type
            self.status_code = status_code

    class _DummyJSONResponse(_DummyResponse):
        pass

    responses_mod.HTMLResponse = _DummyResponse
    responses_mod.FileResponse = _DummyResponse
    responses_mod.JSONResponse = _DummyJSONResponse
    responses_mod.Response = _DummyResponse
    responses_mod.RedirectResponse = _DummyResponse
    responses_mod.StreamingResponse = _DummyResponse
    sys.modules["fastapi.responses"] = responses_mod

    cors_mod = types.ModuleType("fastapi.middleware.cors")

    class _DummyCORSMiddleware:
        pass

    cors_mod.CORSMiddleware = _DummyCORSMiddleware
    sys.modules["fastapi.middleware.cors"] = cors_mod

    staticfiles_mod = types.ModuleType("fastapi.staticfiles")

    class _DummyStaticFiles:
        def __init__(self, *args, **kwargs):
            pass

    staticfiles_mod.StaticFiles = _DummyStaticFiles
    sys.modules["fastapi.staticfiles"] = staticfiles_mod

if "pydantic" not in sys.modules:
    pydantic_mod = types.ModuleType("pydantic")

    class _DummyBaseModel:
        def __init__(self, **data):
            annotations: dict[str, object] = {}
            for cls in reversed(self.__class__.mro()):
                annotations.update(getattr(cls, "__annotations__", {}))
            for name in annotations:
                if name in data:
                    value = data[name]
                elif hasattr(self.__class__, name):
                    value = getattr(self.__class__, name)
                else:
                    value = None
                setattr(self, name, value)

    pydantic_mod.BaseModel = _DummyBaseModel
    sys.modules["pydantic"] = pydantic_mod

if "starlette.middleware.base" not in sys.modules:
    starlette_base_mod = types.ModuleType("starlette.middleware.base")

    class _DummyBaseHTTPMiddleware:
        def __init__(self, *args, **kwargs):
            pass

    starlette_base_mod.BaseHTTPMiddleware = _DummyBaseHTTPMiddleware
    sys.modules["starlette.middleware.base"] = starlette_base_mod
if "bs4" not in sys.modules:
    bs4_mod = types.ModuleType("bs4")

    class _DummyBeautifulSoup:
        def __init__(self, *args, **kwargs):
            pass

    bs4_mod.BeautifulSoup = _DummyBeautifulSoup
    sys.modules["bs4"] = bs4_mod

from src import chat as chat_module
from src.models import ChatResponse
from tools.chat_backend import serve


class _FakeAnthropicResponse:
    def __init__(self, stop_reason: str, content: list[dict]):
        self.stop_reason = stop_reason
        self.content = content


class _FakeAnthropicMessages:
    def __init__(self, responses: list[_FakeAnthropicResponse]):
        self._responses = list(responses)

    def create(self, **kwargs):
        if not self._responses:
            raise AssertionError("No more fake anthropic responses queued")
        return self._responses.pop(0)


class _FakeAnthropicClient:
    def __init__(self, responses: list[_FakeAnthropicResponse]):
        self.messages = _FakeAnthropicMessages(responses)


class _FakeJSONRequest:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


class TestNotesPipeline(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "cheri.db"
        self.db_patch = patch.object(serve, "DB_PATH", self.db_path)
        self.db_patch.start()
        serve._init_db()

    def tearDown(self) -> None:
        self.db_patch.stop()
        self.tempdir.cleanup()

    def _insert_note(
        self,
        *,
        note_id: str,
        title: str,
        content: str,
        project_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        now = 1_700_000_000_000
        with serve._get_db() as conn:
            conn.execute(
                "INSERT INTO notes (id, project_id, session_id, created, updated, title, content, tags, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    note_id,
                    project_id,
                    session_id,
                    now,
                    now,
                    title,
                    content,
                    serve.json.dumps(tags or []),
                    "cheri_doctor",
                ),
            )

    def test_build_messages_includes_notebook_context(self) -> None:
        payload = chat_module._build_messages(
            query="what have we ruled out",
            reranked=[],
            figures=[],
            conversation=[],
            config={},
            project_context="Project: Lean diagnosis",
            notes_context="Saved notes from the current chat notebook:\n- Fuel pressure test passed.",
            vehicle_settings=None,
            images=[],
        )

        self.assertIn("## NOTEBOOK CONTEXT", payload[0]["system"])
        self.assertIn("Fuel pressure test passed", payload[0]["system"])

    def test_load_notes_for_scope_has_no_global_fallback(self) -> None:
        self._insert_note(
            note_id="note-session",
            session_id="session-1",
            title="Session note",
            content="Lean condition present.",
        )

        notes, scope_kind = serve._load_notes_for_scope()

        self.assertEqual(scope_kind, "none")
        self.assertEqual(notes, [])

    async def test_create_note_rejects_unscoped_payload(self) -> None:
        with self.assertRaises(serve.HTTPException) as ctx:
            await serve.create_note(
                _FakeJSONRequest({"title": "Loose note", "content": "Should not save globally."})
            )

        self.assertEqual(ctx.exception.status_code, 400)

    async def test_run_chat_request_passes_scoped_notes_to_chat(self) -> None:
        self._insert_note(
            note_id="note-session",
            session_id="session-1",
            title="Lean note",
            content="Fuel delivery issue ruled out after pressure test.",
            tags=["diagnostic", "finding"],
        )
        prior_index = serve.index
        serve.index = object()
        captured: dict = {}

        def _fake_chat(
            query: str,
            conversation: list[dict],
            index,
            config: dict,
            skip_vision: bool = False,
            project_context: str | None = None,
            notes_context: str | None = None,
            vehicle_settings: str | None = None,
            images=None,
            progress_cb=None,
            note_callback=None,
            retrieve_notes=None,
            deep_research: bool = False,
        ) -> ChatResponse:
            captured["notes_context"] = notes_context
            captured["tool_result"] = retrieve_notes({"keywords": "fuel pressure"}) if retrieve_notes else None
            return ChatResponse(answer="Reviewed the saved notes.")

        req = serve.ChatRequest(
            query="what have we ruled out so far",
            conversation=[],
            session_id="session-1",
        )

        try:
            with patch.object(serve, "chat", new=_fake_chat):
                payload = await serve._run_chat_request(req)
        finally:
            serve.index = prior_index

        self.assertEqual(payload["answer"], "Reviewed the saved notes.")
        self.assertIn("Lean note", captured["notes_context"])
        self.assertIn("Fuel delivery issue ruled out", captured["notes_context"])
        self.assertEqual(captured["tool_result"][1], False)
        self.assertIn("Lean note", captured["tool_result"][0])

    def test_call_claude_handles_retrieve_notes_tool(self) -> None:
        callback_inputs: list[dict] = []

        def _fake_retrieve_notes(tool_input: dict) -> tuple[str, bool]:
            callback_inputs.append(tool_input)
            return ("Saved notes from the current chat notebook:\n- Ignition coil failed test.", False)

        responses = [
            _FakeAnthropicResponse(
                stop_reason="tool_use",
                content=[
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "retrieve_notes",
                        "input": {"keywords": "ignition coil", "limit": 1},
                    }
                ],
            ),
            _FakeAnthropicResponse(
                stop_reason="end_turn",
                content=[{"type": "text", "text": "The saved notes show the ignition coil failed its test."}],
            ),
        ]

        payload = [{"system": "system", "messages": [{"role": "user", "content": "question"}]}]
        config = {"chat": {"model": "claude-test", "max_tokens": 200, "temperature": 0.1}}

        with patch.object(
            chat_module.anthropic,
            "Anthropic",
            new=lambda: _FakeAnthropicClient(responses),
        ):
            answer = chat_module._call_claude(
                payload,
                config,
                index=None,
                progress_cb=None,
                note_callback=None,
                retrieve_notes=_fake_retrieve_notes,
            )

        self.assertEqual(
            answer,
            "The saved notes show the ignition coil failed its test.",
        )
        self.assertEqual(callback_inputs, [{"keywords": "ignition coil", "limit": 1}])


if __name__ == "__main__":
    unittest.main()
