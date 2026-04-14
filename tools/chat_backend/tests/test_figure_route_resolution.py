import unittest
from pathlib import Path
from unittest.mock import patch

from tools.chat_backend import serve
from src import chat as chat_module
from src.utils import load_jsonl


class TestFigureRouteResolution(unittest.IsolatedAsyncioTestCase):
    async def test_serve_figure_resolves_chunk_style_id(self) -> None:
        figures_path = Path("/home/abe/metro-project/build/figures.jsonl")
        figure = next(iter(load_jsonl(figures_path)))
        asset_id = figure["figure_id"]
        asset_path = Path("/home/abe/metro-project") / figure["asset_path"]

        old_lookup = serve.fig_lookup
        old_chunk_map = serve.chunk_fig_map
        try:
            serve.fig_lookup = {asset_id: figure}
            serve.chunk_fig_map = {"fig_test_chunk_0": asset_id}

            response = await serve.serve_figure("fig_test_chunk_0")

            self.assertEqual(response.path, str(asset_path))
            self.assertIn(response.media_type, {"image/webp", "image/png"})
        finally:
            serve.fig_lookup = old_lookup
            serve.chunk_fig_map = old_chunk_map

    async def test_serve_figure_resolves_source_label_id(self) -> None:
        figures_path = Path("/home/abe/metro-project/build/figures.jsonl")
        figure = next(iter(load_jsonl(figures_path)))
        asset_id = figure["figure_id"]
        asset_path = Path("/home/abe/metro-project") / figure["asset_path"]

        old_lookup = serve.fig_lookup
        old_chunk_map = serve.chunk_fig_map
        try:
            serve.fig_lookup = {asset_id: figure}
            serve.chunk_fig_map = {"7A-1": asset_id}

            response = await serve.serve_figure("7A-1")

            self.assertEqual(response.path, str(asset_path))
            self.assertIn(response.media_type, {"image/webp", "image/png"})
        finally:
            serve.fig_lookup = old_lookup
            serve.chunk_fig_map = old_chunk_map

    def test_parse_response_resolves_source_label_figure_citation(self) -> None:
        with patch.object(
            chat_module,
            "_chunk_fig_lookup",
            return_value={"7A-1@477": "fig_p0477_000", "7A-1": "fig_p0477_000"},
        ):
            response = chat_module._parse_response(
                "See the layout [p477 | fig: 7A-1].",
                reranked=[],
            )

        self.assertEqual(response.figure_refs, ["fig_p0477_000"])
        self.assertIn("[p477 | fig: fig_p0477_000]", response.answer)


if __name__ == "__main__":
    unittest.main()
