import unittest
from unittest.mock import patch

from tools.chat_backend import serve


class _DummyChatResponse:
    def __init__(self, answer: str):
        self.answer = answer
        self.citations = []
        self.figure_refs = []


class TestShopFollowup(unittest.IsolatedAsyncioTestCase):
    async def test_llm_first_followup_executes_emitted_shop_tags(self) -> None:
        serve._shop_sessions.clear()
        prior_index = serve.index
        serve.index = object()

        parts_calls: list[dict] = []

        async def _fake_parts_search(
            part_name: str,
            sources: list[str] | None = None,
            manual_context: str | None = None,
            progress_cb=None,
        ) -> serve.PartsSearchResult:
            parts_calls.append(
                {
                    "part_name": part_name,
                    "sources": sources or [],
                    "manual_context": manual_context,
                }
            )
            if len(parts_calls) == 1:
                return serve.PartsSearchResult(
                    llm_text="[PARTS SEARCH: no results]",
                    sources=["parts search (amazon): cigarette lighter socket"],
                    items=[],
                    queries_used=["cigarette lighter socket"],
                    sources_searched=["amazon"],
                    source_outcomes={"amazon": {"status": "no_results", "count": 0}},
                    manual_context="cached-manual-context",
                )
            return serve.PartsSearchResult(
                llm_text="[PARTS SEARCH — 1 results for 'cigarette lighter socket' from ebay, rockauto]",
                sources=["parts search (ebay, rockauto): cigarette lighter socket"],
                items=[
                    {
                        "source": "ebay",
                        "title": "12V cigarette lighter socket kit",
                        "item_url": "https://example.com/socket",
                    }
                ],
                queries_used=["cigarette lighter socket"],
                sources_searched=["ebay", "rockauto"],
                source_outcomes={
                    "ebay": {"status": "ok", "count": 1},
                    "rockauto": {"status": "no_results", "count": 0},
                },
                    manual_context="cached-manual-context",
                )

        def _fake_chat(
            query: str,
            conversation: list[dict],
            index,
            config,
            skip_vision: bool,
            project_context,
            vehicle_settings: str,
            images=None,
            progress_cb=None,
        ) -> _DummyChatResponse:
            if "[PARTS SEARCH" in query and "no results" in query.lower():
                return _DummyChatResponse(
                    "Amazon came up empty. Want me to check eBay and RockAuto?"
                )
            if "[PARTS SEARCH" in query:
                return _DummyChatResponse("Found options on eBay and RockAuto.")
            if "sure take a look there" in query.lower():
                return _DummyChatResponse(
                    "Sure thing.\n[SHOP_SEARCH: cigarette lighter socket | ebay, rockauto]"
                )
            return _DummyChatResponse(
                "Let me check Amazon first.\n[SHOP_SEARCH: cigarette lighter socket | amazon]"
            )

        first_query = (
            "Cheri's cigerette lighter is broken and i need to replace it. "
            "Can you find one that will work on amazon"
        )
        req1 = serve.ChatRequest(
            query=first_query,
            conversation=[],
            shop_mode_hint=True,
            shop_part_hint=None,
            tech_mode_hint=False,
        )

        req2 = serve.ChatRequest(
            query="Sure take a look there.",
            conversation=[
                {"role": "user", "text": first_query},
                {
                    "role": "assistant",
                    "text": "Amazon came up empty. Want me to check eBay and RockAuto?",
                },
            ],
            shop_mode_hint=False,
            shop_part_hint=None,
            tech_mode_hint=False,
        )

        try:
            with patch.object(serve, "_parts_search", new=_fake_parts_search), \
                 patch.object(serve, "chat", new=_fake_chat):
                await serve._run_chat_request(req1)
                payload = await serve._run_chat_request(req2)
        finally:
            serve.index = prior_index

        self.assertEqual(len(parts_calls), 2)
        self.assertEqual(parts_calls[0]["sources"], ["amazon"])
        self.assertEqual(parts_calls[1]["sources"], ["ebay", "rockauto"])
        self.assertEqual(parts_calls[1]["part_name"], "cigarette lighter socket")
        self.assertTrue(payload.get("shopping_results"), "follow-up should include attempted results")

    async def test_watchdog_injects_shop_tag_when_search_is_promised(self) -> None:
        serve._shop_sessions.clear()
        prior_index = serve.index
        serve.index = object()

        parts_calls: list[dict] = []

        async def _fake_enrich(query: str, conversation: list[dict] | None = None):
            return query, [], []

        async def _fake_parts_search(
            part_name: str,
            sources: list[str] | None = None,
            manual_context: str | None = None,
            progress_cb=None,
        ) -> serve.PartsSearchResult:
            parts_calls.append(
                {
                    "part_name": part_name,
                    "sources": sources or [],
                }
            )
            return serve.PartsSearchResult(
                llm_text="[PARTS SEARCH — 1 result]",
                sources=["parts search (amazon, ebay, rockauto): usb charger socket"],
                items=[
                    {
                        "source": "amazon",
                        "title": "Flush mount USB charger 12V socket",
                        "item_url": "https://example.com/usb",
                    }
                ],
                queries_used=["flush mount usb charger 12v 20.63mm socket"],
                sources_searched=["amazon", "ebay", "rockauto"],
                source_outcomes={
                    "amazon": {"status": "ok", "count": 1},
                    "ebay": {"status": "no_results", "count": 0},
                    "rockauto": {"status": "no_results", "count": 0},
                },
            )

        def _fake_chat(
            query: str,
            conversation: list[dict],
            index,
            config,
            skip_vision: bool,
            project_context,
            vehicle_settings: str,
            images=None,
            progress_cb=None,
        ) -> _DummyChatResponse:
            if "[PARTS SEARCH" in query:
                return _DummyChatResponse("Here are the best USB charger socket options.")
            return _DummyChatResponse(
                "Great, let me do a proper targeted search for a flush-mount USB charger "
                "that fits that 20.63mm socket hole. Stand by!"
            )

        req = serve.ChatRequest(
            query=(
                "The manual calls it a CIGAR LIGHTER ASSEMBLY. "
                "Find a flush-mount USB charger for the 20.63mm opening."
            ),
            conversation=[],
            shop_mode_hint=False,
            shop_part_hint=None,
            tech_mode_hint=False,
        )

        try:
            with patch.object(serve, "_enrich_query", new=_fake_enrich), \
                 patch.object(serve, "_parts_search", new=_fake_parts_search), \
                 patch.object(serve, "chat", new=_fake_chat):
                payload = await serve._run_chat_request(req)
        finally:
            serve.index = prior_index

        self.assertEqual(len(parts_calls), 1, "watchdog should trigger exactly one parts search")
        self.assertEqual(parts_calls[0]["sources"], ["amazon", "ebay", "rockauto"])
        self.assertIn("usb charger", parts_calls[0]["part_name"].lower())
        self.assertTrue(payload.get("shopping_results"), "watchdog-triggered search should return results")

    async def test_query_quality_guard_falls_back_to_session_part(self) -> None:
        serve._shop_sessions.clear()
        prior_index = serve.index
        serve.index = object()

        parts_calls: list[dict] = []

        async def _fake_parts_search(
            part_name: str,
            sources: list[str] | None = None,
            manual_context: str | None = None,
            progress_cb=None,
        ) -> serve.PartsSearchResult:
            parts_calls.append({"part_name": part_name, "sources": sources or []})
            return serve.PartsSearchResult(
                llm_text="[PARTS SEARCH — 1 result]",
                sources=["parts search (amazon): cigarette lighter socket"],
                items=[
                    {
                        "source": "amazon",
                        "title": "Socket",
                        "item_url": "https://example.com/socket",
                    }
                ],
                queries_used=["cigarette lighter socket"],
                sources_searched=["amazon"],
                source_outcomes={"amazon": {"status": "ok", "count": 1}},
            )

        def _fake_chat(
            query: str,
            conversation: list[dict],
            index,
            config,
            skip_vision: bool,
            project_context,
            vehicle_settings: str,
            images=None,
            progress_cb=None,
        ) -> _DummyChatResponse:
            if "[PARTS SEARCH" in query:
                return _DummyChatResponse("Found one option.")
            return _DummyChatResponse("Let me run that.\n[SHOP_SEARCH: is something that | amazon]")

        req = serve.ChatRequest(
            query="Go ahead",
            conversation=[],
            shop_mode_hint=False,
            shop_part_hint=None,
            tech_mode_hint=False,
        )
        session = serve._get_shop_session(req.query, req.conversation, req.project_context)
        session.last_part_name = "cigarette lighter socket"

        try:
            with patch.object(serve, "_parts_search", new=_fake_parts_search), \
                 patch.object(serve, "chat", new=_fake_chat):
                await serve._run_chat_request(req)
        finally:
            serve.index = prior_index

        self.assertEqual(len(parts_calls), 1)
        self.assertEqual(parts_calls[0]["part_name"], "cigarette lighter socket")
        self.assertEqual(parts_calls[0]["sources"], ["amazon"])

    async def test_query_quality_guard_skips_search_when_no_fallback(self) -> None:
        serve._shop_sessions.clear()
        prior_index = serve.index
        serve.index = object()
        parts_called = False

        async def _fake_parts_search(
            part_name: str,
            sources: list[str] | None = None,
            manual_context: str | None = None,
            progress_cb=None,
        ) -> serve.PartsSearchResult:
            nonlocal parts_called
            parts_called = True
            return serve.PartsSearchResult(
                llm_text="[PARTS SEARCH]",
                sources=[],
                items=[],
            )

        def _fake_chat(
            query: str,
            conversation: list[dict],
            index,
            config,
            skip_vision: bool,
            project_context,
            vehicle_settings: str,
            images=None,
            progress_cb=None,
        ) -> _DummyChatResponse:
            return _DummyChatResponse("Trying now.\n[SHOP_SEARCH: is something that | amazon]")

        req = serve.ChatRequest(
            query="go ahead",
            conversation=[],
            shop_mode_hint=False,
            shop_part_hint=None,
            tech_mode_hint=False,
        )

        try:
            with patch.object(serve, "_parts_search", new=_fake_parts_search), \
                 patch.object(serve, "chat", new=_fake_chat):
                payload = await serve._run_chat_request(req)
        finally:
            serve.index = prior_index

        self.assertFalse(parts_called, "invalid tag terms without fallback must skip search")
        self.assertNotIn("[SHOP_SEARCH", payload["answer"])
        self.assertEqual(payload.get("shopping_results"), [])

    async def test_query_quality_guard_strips_conversational_prefix(self) -> None:
        serve._shop_sessions.clear()
        prior_index = serve.index
        serve.index = object()
        parts_calls: list[dict] = []

        async def _fake_parts_search(
            part_name: str,
            sources: list[str] | None = None,
            manual_context: str | None = None,
            progress_cb=None,
        ) -> serve.PartsSearchResult:
            parts_calls.append({"part_name": part_name, "sources": sources or []})
            return serve.PartsSearchResult(
                llm_text="[PARTS SEARCH — 1 result]",
                sources=["parts search (amazon): usb adapter lighter socket"],
                items=[
                    {
                        "source": "amazon",
                        "title": "USB socket adapter",
                        "item_url": "https://example.com/usb-adapter",
                    }
                ],
                queries_used=[part_name],
                sources_searched=["amazon"],
                source_outcomes={"amazon": {"status": "ok", "count": 1}},
            )

        def _fake_chat(
            query: str,
            conversation: list[dict],
            index,
            config,
            skip_vision: bool,
            project_context,
            vehicle_settings: str,
            images=None,
            progress_cb=None,
        ) -> _DummyChatResponse:
            if "[PARTS SEARCH" in query:
                return _DummyChatResponse("Found an option.")
            return _DummyChatResponse(
                "Searching now.\n"
                "[SHOP_SEARCH: Yeah now look on for a USB adapter that will take the place "
                "of the lighter like we talked about | amazon]"
            )

        req = serve.ChatRequest(
            query=(
                "Yeah now look on amazon for a USB adapter that will take "
                "the place of the lighter like we talked about"
            ),
            conversation=[],
            shop_mode_hint=True,
            shop_part_hint="Yeah now look on for a USB adapter that will take the place of the lighter like we talked about",
            tech_mode_hint=False,
        )

        try:
            with patch.object(serve, "_parts_search", new=_fake_parts_search), \
                 patch.object(serve, "chat", new=_fake_chat):
                await serve._run_chat_request(req)
        finally:
            serve.index = prior_index

        self.assertEqual(len(parts_calls), 1, "conversational tag should still resolve to one search")
        searched = parts_calls[0]["part_name"].lower()
        self.assertIn("usb", searched)
        self.assertNotEqual(searched, req.query.lower())
        self.assertFalse(searched.startswith("yeah"))
        self.assertNotIn("look on", searched)

    async def test_watchdog_triggers_on_on_it_searching_now_phrase(self) -> None:
        serve._shop_sessions.clear()
        prior_index = serve.index
        serve.index = object()
        parts_calls: list[dict] = []

        async def _fake_parts_search(
            part_name: str,
            sources: list[str] | None = None,
            manual_context: str | None = None,
            progress_cb=None,
        ) -> serve.PartsSearchResult:
            parts_calls.append({"part_name": part_name, "sources": sources or []})
            return serve.PartsSearchResult(
                llm_text="[PARTS SEARCH — 1 result]",
                sources=["parts search (ebay): panel mount cigarette lighter socket"],
                items=[
                    {
                        "source": "ebay",
                        "title": "Panel-mount cigar lighter socket",
                        "item_url": "https://example.com/ebay/socket",
                    }
                ],
                queries_used=[part_name],
                sources_searched=["ebay"],
                source_outcomes={"ebay": {"status": "ok", "count": 1}},
            )

        def _fake_chat(
            query: str,
            conversation: list[dict],
            index,
            config,
            skip_vision: bool,
            project_context,
            vehicle_settings: str,
            images=None,
            progress_cb=None,
        ) -> _DummyChatResponse:
            if "[PARTS SEARCH" in query:
                return _DummyChatResponse("Found eBay options.")
            return _DummyChatResponse(
                "On it — searching eBay now for the correct Suzuki/Geo Metro "
                "panel-mount cigar lighter socket."
            )

        req = serve.ChatRequest(
            query="Search on ebay for it",
            conversation=[
                {
                    "role": "user",
                    "text": "Cheri needs a new cigarette lighter socket the current one is broken. Find me one on amazon",
                },
                {
                    "role": "assistant",
                    "text": "I found options. Want me to check eBay too?",
                },
            ],
            shop_mode_hint=False,
            shop_part_hint=None,
            tech_mode_hint=False,
        )
        session = serve._get_shop_session(req.query, req.conversation, req.project_context)
        session.last_part_name = "panel mount cigarette lighter socket"

        try:
            with patch.object(serve, "_parts_search", new=_fake_parts_search), \
                 patch.object(serve, "chat", new=_fake_chat):
                await serve._run_chat_request(req)
        finally:
            serve.index = prior_index

        self.assertEqual(len(parts_calls), 1, "watchdog should fire for 'On it — searching ... now'")
        self.assertEqual(parts_calls[0]["sources"], ["ebay"])
        self.assertEqual(parts_calls[0]["part_name"], "panel mount cigarette lighter socket")

    async def test_query_quality_guard_user_fallback_extracts_part_not_sentence(self) -> None:
        serve._shop_sessions.clear()
        prior_index = serve.index
        serve.index = object()
        parts_calls: list[dict] = []

        async def _fake_parts_search(
            part_name: str,
            sources: list[str] | None = None,
            manual_context: str | None = None,
            progress_cb=None,
        ) -> serve.PartsSearchResult:
            parts_calls.append({"part_name": part_name, "sources": sources or []})
            return serve.PartsSearchResult(
                llm_text="[PARTS SEARCH — 1 result]",
                sources=["parts search (oreilly): alternator"],
                items=[
                    {
                        "source": "oreilly",
                        "title": "Alternator",
                        "item_url": "https://example.com/oreilly/alt",
                    }
                ],
                queries_used=[part_name],
                sources_searched=["oreilly"],
                source_outcomes={"oreilly": {"status": "ok", "count": 1}},
            )

        def _fake_chat(
            query: str,
            conversation: list[dict],
            index,
            config,
            skip_vision: bool,
            project_context,
            vehicle_settings: str,
            images=None,
            progress_cb=None,
        ) -> _DummyChatResponse:
            if "[PARTS SEARCH" in query:
                return _DummyChatResponse("Found options.")
            return _DummyChatResponse("[SHOP_SEARCH: O'Reilly | oreilly]")

        req = serve.ChatRequest(
            query="does oriley have cheris altenator stock",
            conversation=[],
            shop_mode_hint=False,
            shop_part_hint=None,
            tech_mode_hint=False,
        )

        try:
            with patch.object(serve, "_parts_search", new=_fake_parts_search), \
                 patch.object(serve, "chat", new=_fake_chat):
                await serve._run_chat_request(req)
        finally:
            serve.index = prior_index

        self.assertEqual(len(parts_calls), 1)
        term = parts_calls[0]["part_name"].lower()
        self.assertIn("altenator", term)
        self.assertNotIn("does", term)
        self.assertNotIn("oriley", term)
        self.assertNotIn("stock", term)
        self.assertEqual(parts_calls[0]["sources"], ["oreilly"])


if __name__ == "__main__":
    unittest.main()
