import sys
import types
import unittest
from pathlib import Path

if "faster_whisper" not in sys.modules:
    fake_faster_whisper = types.ModuleType("faster_whisper")

    class DummyWhisperModel:
        def __init__(self, *args, **kwargs):
            return

    fake_faster_whisper.WhisperModel = DummyWhisperModel
    sys.modules["faster_whisper"] = fake_faster_whisper

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conspectum.process import get_preferred_latex_engines  # noqa: E402
from conspectum.process import latex_engine_available  # noqa: E402
from conspectum.process import latex_to_pdf  # noqa: E402
from conspectum.process import localize_template  # noqa: E402
from conspectum.process import prepare_latex_document  # noqa: E402


TEMPLATE_TEXT = (SRC_DIR / "conspectum" / "prompts" / "template.tex").read_text(encoding="utf-8")


def build_prepared_document(language: str, title: str, abstract: str, body: str):
    tex = localize_template(TEMPLATE_TEXT, language)
    tex = tex.replace("<INSERT TITLE HERE>", title)
    tex = tex.replace("<INSERT ABSTRACT HERE>", abstract)
    tex = tex.replace("%% <INSERT CONTENT HERE>", body)
    return prepare_latex_document(tex, language)


class ProcessLatexTests(unittest.TestCase):
    def test_unicode_engines_are_preferred_when_available(self):
        prepared = build_prepared_document(
            "en",
            "Study of Time and Space",
            "We compare τ₂ ≤ 5 in a short English abstract.",
            "\\section{Overview}\nEnglish text.\n",
        )
        engines = get_preferred_latex_engines(prepared.tex, "en")

        if latex_engine_available("xelatex"):
            self.assertEqual(engines[0], "xelatex")
        elif latex_engine_available("lualatex"):
            self.assertEqual(engines[0], "lualatex")
        else:
            self.assertEqual(engines[0], "pdflatex")

    def test_prepare_latex_document_normalizes_russian_unicode_math_tokens(self):
        prepared = build_prepared_document(
            "ru",
            "\u0418\u0437\u0443\u0447\u0435\u043d\u0438\u0435 \u0432\u0440\u0435\u043c\u0435\u043d\u0438 \u0438 \u043f\u0440\u043e\u0441\u0442\u0440\u0430\u043d\u0441\u0442\u0432\u0430",
            "\u0420\u0430\u0441\u0441\u043c\u0430\u0442\u0440\u0438\u0432\u0430\u0435\u043c \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440 \u03c4 \u0438 \u043c\u043e\u043b\u0435\u043a\u0443\u043b\u0443 CO\u2082, \u0433\u0434\u0435 \u03c4\u2082 \u2264 5.",
            "\\section{\u041e\u0441\u043d\u043e\u0432\u043d\u0430\u044f \u0438\u0434\u0435\u044f}\n"
            "\u041f\u0443\u0441\u0442\u044c \u03c4\u2082 \u2264 5 \u0438 x\u2082 = 3. "
            "\u0422\u043e\u0433\u0434\u0430 \u0440\u0430\u0441\u0441\u043c\u043e\u0442\u0440\u0438\u043c $τ₂ + 1 \\\\ge 0$ "
            "\u0438 \u043c\u043d\u043e\u0436\u0435\u0441\u0442\u0432\u043e ℝ.\n",
        )

        document_body = prepared.tex.split(r"\begin{document}", 1)[1]
        self.assertIn(r"\tau", document_body)
        self.assertIn(r"CO_{2}", document_body)
        self.assertIn(r"\leq", document_body)
        self.assertIn(r"\mathbb{R}", document_body)
        self.assertNotIn("\u03c4", document_body)
        self.assertNotIn("\u2082", document_body)
        self.assertNotIn("\u2264", document_body)
        self.assertNotIn("\u211d", document_body)

    def test_prepare_latex_document_keeps_cyrillic_text(self):
        prepared = build_prepared_document(
            "ru",
            "\u0418\u0437\u0443\u0447\u0435\u043d\u0438\u0435 \u0432\u0440\u0435\u043c\u0435\u043d\u0438",
            "\u041a\u0440\u0430\u0442\u043a\u0430\u044f \u0430\u043d\u043d\u043e\u0442\u0430\u0446\u0438\u044f.",
            "\\section{\u041e\u0431\u0437\u043e\u0440}\n"
            "\u042d\u0442\u043e \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442 \u0431\u0435\u0437 \u043f\u043e\u0442\u0435\u0440\u0438 \u043a\u0438\u0440\u0438\u043b\u043b\u0438\u0446\u044b.\n",
        )

        self.assertIn("\u0410\u043d\u043d\u043e\u0442\u0430\u0446\u0438\u044f", prepared.tex)
        self.assertIn("\u042d\u0442\u043e \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442", prepared.tex)

    def test_english_document_with_unicode_math_compiles(self):
        engines_available = get_preferred_latex_engines("plain", "en")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")

        prepared = build_prepared_document(
            "en",
            "Study of Time and Space",
            "We compare \u03c4\u2082 \u2264 5 while keeping $x_2$ intact.",
            "\\section{Overview}\nEnglish text with τ and CO₂ outside math, plus $τ₂ + 1$ inside math.\n",
        )
        engine = get_preferred_latex_engines(prepared.tex, "en")[0]
        pdf_bytes, _diagnostics = latex_to_pdf(prepared.tex, engine)

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)

    def test_russian_document_with_unicode_math_compiles(self):
        engines_available = get_preferred_latex_engines("plain", "ru")
        if not engines_available:
            self.skipTest("No LaTeX engine is available in PATH.")

        prepared = build_prepared_document(
            "ru",
            "\u0418\u0437\u0443\u0447\u0435\u043d\u0438\u0435 \u0432\u0440\u0435\u043c\u0435\u043d\u0438 \u0438 \u043f\u0440\u043e\u0441\u0442\u0440\u0430\u043d\u0441\u0442\u0432\u0430",
            "\u0420\u0430\u0441\u0441\u043c\u0430\u0442\u0440\u0438\u0432\u0430\u0435\u043c \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440 \u03c4 \u0438 \u043c\u043e\u043b\u0435\u043a\u0443\u043b\u0443 CO\u2082, \u0433\u0434\u0435 \u03c4\u2082 \u2264 5.",
            "\\section{\u041e\u0441\u043d\u043e\u0432\u043d\u0430\u044f \u0438\u0434\u0435\u044f}\n"
            "\u041f\u0443\u0441\u0442\u044c \u03c4\u2082 \u2264 5 \u0438 x\u2082 = 3. "
            "\u0422\u043e\u0433\u0434\u0430 \u0440\u0430\u0441\u0441\u043c\u043e\u0442\u0440\u0438\u043c $τ₂ + 1 \\\\ge 0$ "
            "\u0438 \u043c\u043d\u043e\u0436\u0435\u0441\u0442\u0432\u043e ℝ.\n",
        )
        engine = get_preferred_latex_engines(prepared.tex, "ru")[0]
        pdf_bytes, _diagnostics = latex_to_pdf(prepared.tex, engine)

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)


if __name__ == "__main__":
    unittest.main()
