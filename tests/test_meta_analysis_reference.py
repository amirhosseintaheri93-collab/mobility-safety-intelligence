"""Verify attribution and navigation for MSI's separate literature layer."""

import ast
import json
from pathlib import Path
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


class MetaAnalysisReferenceTests(unittest.TestCase):
    def setUp(self):
        self.benchmark = json.loads(
            (ROOT / "llm/literature_benchmark.json").read_text(encoding="utf-8")
        )
        tree = ast.parse((ROOT / "app/dashboard.py").read_text(encoding="utf-8"))
        selected = {"render_meta_analysis_reference", "open_research_view"}
        nodes = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in selected]
        self.ui = Mock()
        self.ui.session_state = {}
        self.route = Mock()
        self.scope = {"st": self.ui, "set_research_route": self.route, "queue_ui_sound": Mock()}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "meta_analysis", "exec"), self.scope)

    def test_compact_citation_identifies_journal_and_paper(self):
        self.scope["render_meta_analysis_reference"](self.benchmark)
        text = self.ui.caption.call_args.args[0]
        for item in ("European Transport Research Review", "2026", "Published meta-analysis",
                     "https://doi.org/10.1186/s12544-026-00774-9"):
            self.assertIn(item, text)
        self.ui.markdown.assert_not_called()

    def test_full_citation_names_the_article_and_separates_evidence(self):
        self.scope["render_meta_analysis_reference"](self.benchmark, compact=False)
        text = "\n".join(call.args[0] for call in self.ui.method_calls)
        self.assertIn(self.benchmark["title"], text)
        self.assertIn("kept separate from the Berlin simulation results", text)

    def test_literature_shortcut_opens_the_correct_results_module(self):
        self.scope["open_research_view"]("literature")
        self.assertTrue(self.ui.session_state["entry_portal_open"])
        self.route.assert_called_once_with("Results", "Literature benchmark", remember=False)

    def test_missing_metadata_does_not_create_a_broken_link(self):
        self.scope["render_meta_analysis_reference"]({})
        self.assertEqual(self.ui.method_calls, [])


if __name__ == "__main__":
    unittest.main()
