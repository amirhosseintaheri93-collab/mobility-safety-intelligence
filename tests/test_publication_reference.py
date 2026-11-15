"""Keep MSI's study citation visible without loading the full dashboard."""

import ast
import json
import unittest
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1016/j.aap.2026.108835"
STUDY_PAGE = "https://amirhtaheri.com/publications/automated-vehicle-shares-safer-roads/"


class PublicationReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.references = json.loads(
            (ROOT / "llm" / "academic_references.json").read_text(encoding="utf-8")
        )["references"]
        tree = ast.parse((ROOT / "app" / "dashboard.py").read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "render_study_reference"
        )
        self.ui = Mock()
        scope = {"st": self.ui}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "study_reference", "exec"), scope)
        self.render = scope["render_study_reference"]

    def test_published_reference_and_documentation_agree(self) -> None:
        study = next(item for item in self.references if item["id"] == "ref_taheri_aap_2027")
        self.assertEqual(study["doi"], DOI)
        self.assertEqual(study["url"], f"https://doi.org/{DOI}")
        self.assertEqual(study["website_url"], STUDY_PAGE)
        self.assertIn("(2027)", study["citation"])
        for filename in ("README.md", "CITATION.cff", "DATA_LICENSE.md", "data/DEMO_DATASET.md"):
            self.assertIn(DOI, (ROOT / filename).read_text(encoding="utf-8"), filename)

    def test_compact_header_keeps_only_the_citation_links(self) -> None:
        self.render(self.references)
        caption = self.ui.caption.call_args.args[0]
        self.assertIn(DOI, caption)
        self.assertIn(STUDY_PAGE, caption)
        self.ui.write.assert_not_called()
        self.ui.markdown.assert_not_called()

    def test_overview_explains_provenance_and_links_source_code(self) -> None:
        self.render(self.references, compact=False)
        text = "\n".join(call.args[0] for call in self.ui.method_calls)
        for phrase in (
            DOI, STUDY_PAGE, "not the complete research dataset",
            "interface extensions", "not additional findings validated",
            "mobility-safety-intelligence#citation", "also credit MSI",
        ):
            self.assertIn(phrase, text)

    def test_missing_study_does_not_render_a_misleading_reference(self) -> None:
        self.render([])
        self.assertEqual(self.ui.method_calls, [])


if __name__ == "__main__":
    unittest.main()
