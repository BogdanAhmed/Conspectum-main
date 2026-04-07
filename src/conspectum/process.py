import io
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import typing
import unicodedata
from dataclasses import dataclass

import openai

from .logger import Logger
from .summary import (
    LANGUAGE_NAMES,
    detect_language_from_text,
    make_summary_from_transcript,
    normalize_latex_text,
    postprocess_summary,
    transcribe_audio,
)


# Поддержка macOS, если pdflatex лежит в стандартной папке TeX
DETAIL_LEVEL_GUIDANCE = {
    "brief": (
        "Prefer a concise result. Focus on the core ideas, keep the number of sections small, and omit secondary remarks."
    ),
    "standard": (
        "Produce a balanced result with the main sections, key formulas, and essential examples."
    ),
    "detailed": (
        "Produce a richer result with more subsections, careful explanations, more definitions, and more lecture-derived examples."
    ),
}

PROTECTED_AMPERSAND_ENVIRONMENTS = {
    "tabular",
    "tabular*",
    "array",
    "align",
    "align*",
    "aligned",
    "eqnarray",
    "eqnarray*",
    "split",
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "Vmatrix",
    "smallmatrix",
    "cases",
}

MATH_ENVIRONMENTS = (
    "equation",
    "equation*",
    "align",
    "align*",
    "aligned",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "displaymath",
    "array",
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "Vmatrix",
    "smallmatrix",
    "cases",
    "split",
)

UNICODE_LATEX_REPLACEMENTS = {
    "\u00a0": " ",
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2013": "--",
    "\u2014": "---",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u2116": "No. ",
    "\u2212": "-",
    "\u2260": r"\ensuremath{\neq}",
    "\u2264": r"\ensuremath{\leq}",
    "\u2265": r"\ensuremath{\geq}",
    "\u2190": r"\ensuremath{\leftarrow}",
    "\u2192": r"\ensuremath{\to}",
    "\u221e": r"\ensuremath{\infty}",
    "\u2208": r"\ensuremath{\in}",
    "\u00b1": r"\ensuremath{\pm}",
    "\u00d7": r"\ensuremath{\times}",
    "\u00f7": r"\ensuremath{\div}",
    "\u00b0": r"\ensuremath{^\circ}",
    "\u2248": r"\ensuremath{\approx}",
    "\u2261": r"\ensuremath{\equiv}",
    "\u2194": r"\ensuremath{\leftrightarrow}",
    "\u21d2": r"\ensuremath{\Rightarrow}",
    "\u21d4": r"\ensuremath{\Leftrightarrow}",
    "\u2202": r"\ensuremath{\partial}",
    "\u2207": r"\ensuremath{\nabla}",
    "\u2211": r"\ensuremath{\sum}",
    "\u220f": r"\ensuremath{\prod}",
    "\u222b": r"\ensuremath{\int}",
    "\u221a": r"\ensuremath{\sqrt{}}",
    "\u226a": r"\ensuremath{\ll}",
    "\u226b": r"\ensuremath{\gg}",
    "\u2282": r"\ensuremath{\subset}",
    "\u2286": r"\ensuremath{\subseteq}",
    "\u2209": r"\ensuremath{\notin}",
}

CYRILLIC_TO_ASCII = {
    "\u0410": "A",
    "\u0411": "B",
    "\u0412": "V",
    "\u0413": "G",
    "\u0414": "D",
    "\u0415": "E",
    "\u0401": "E",
    "\u0416": "Zh",
    "\u0417": "Z",
    "\u0418": "I",
    "\u0419": "I",
    "\u041a": "K",
    "\u041b": "L",
    "\u041c": "M",
    "\u041d": "N",
    "\u041e": "O",
    "\u041f": "P",
    "\u0420": "R",
    "\u0421": "S",
    "\u0422": "T",
    "\u0423": "U",
    "\u0424": "F",
    "\u0425": "Kh",
    "\u0426": "Ts",
    "\u0427": "Ch",
    "\u0428": "Sh",
    "\u0429": "Shch",
    "\u042a": "",
    "\u042b": "Y",
    "\u042c": "",
    "\u042d": "E",
    "\u042e": "Yu",
    "\u042f": "Ya",
    "\u0430": "a",
    "\u0431": "b",
    "\u0432": "v",
    "\u0433": "g",
    "\u0434": "d",
    "\u0435": "e",
    "\u0451": "e",
    "\u0436": "zh",
    "\u0437": "z",
    "\u0438": "i",
    "\u0439": "i",
    "\u043a": "k",
    "\u043b": "l",
    "\u043c": "m",
    "\u043d": "n",
    "\u043e": "o",
    "\u043f": "p",
    "\u0440": "r",
    "\u0441": "s",
    "\u0442": "t",
    "\u0443": "u",
    "\u0444": "f",
    "\u0445": "kh",
    "\u0446": "ts",
    "\u0447": "ch",
    "\u0448": "sh",
    "\u0449": "shch",
    "\u044a": "",
    "\u044b": "y",
    "\u044c": "",
    "\u044d": "e",
    "\u044e": "yu",
    "\u044f": "ya",
}

UNICODE_GREEK_MAP = {
    "Α": "A",
    "Β": "B",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Θ": r"\Theta",
    "Ι": "I",
    "Κ": "K",
    "Λ": r"\Lambda",
    "Μ": "M",
    "Ν": "N",
    "Ξ": r"\Xi",
    "Ο": "O",
    "Π": r"\Pi",
    "Ρ": "P",
    "Σ": r"\Sigma",
    "Τ": "T",
    "Υ": r"\Upsilon",
    "Φ": r"\Phi",
    "Χ": "X",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "ο": "o",
    "π": r"\pi",
    "ρ": r"\rho",
    "ς": r"\varsigma",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "ϑ": r"\vartheta",
    "ϕ": r"\varphi",
    "ϖ": r"\varpi",
    "ϵ": r"\varepsilon",
    "ϱ": r"\varrho",
}

UNICODE_MATH_TEXT_MAP = {
    "ℝ": r"\mathbb{R}",
    "ℂ": r"\mathbb{C}",
    "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}",
    "ℚ": r"\mathbb{Q}",
    "ℓ": r"\ell",
    "∅": r"\varnothing",
}

UNICODE_MATH_OPERATOR_MAP = {
    "−": "-",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "≡": r"\equiv",
    "±": r"\pm",
    "×": r"\times",
    "÷": r"\div",
    "·": r"\cdot",
    "⋅": r"\cdot",
    "∞": r"\infty",
    "∈": r"\in",
    "∉": r"\notin",
    "⊂": r"\subset",
    "⊆": r"\subseteq",
    "→": r"\to",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow",
    "∂": r"\partial",
    "∇": r"\nabla",
    "∑": r"\sum",
    "∏": r"\prod",
    "∫": r"\int",
    "°": r"^\circ",
}

UNICODE_SUBSCRIPT_MAP = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₊": "+",
    "₋": "-",
    "₌": "=",
    "₍": "(",
    "₎": ")",
}

UNICODE_SUPERSCRIPT_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁼": "=",
    "⁽": "(",
    "⁾": ")",
    "ⁿ": "n",
}

_LATEX_BASE_COMMANDS: dict[str, list[str]] = {}
_PDF_FALLBACK_FONT_NAME: typing.Optional[str] = None
LANGUAGE_PREAMBLE_START = "% <CONSPECTUM LANGUAGE SETUP START>"
LANGUAGE_PREAMBLE_END = "% <CONSPECTUM LANGUAGE SETUP END>"
TEXT_MATH_TOKEN_PATTERN = re.compile(
    r"(?<!\\)(?P<base>[A-Za-z0-9]+|[Α-Ωα-ωϐ-ϖ])(?P<sub>[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎]+)?(?P<sup>[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ]+)?"
)


@dataclass
class ProcessResult:
    transcript: str
    language: str
    title: str
    abstract: str
    tex: str
    pdf: typing.Optional[bytes]
    pdf_warning: typing.Optional[str] = None


@dataclass
class LatexPreparationResult:
    tex: str
    notes: list[str]


@dataclass
class LatexCompilationError(RuntimeError):
    engine: str
    summary: str
    diagnostics: str

    def __str__(self) -> str:
        return self.summary


if not shutil.which("pdflatex"):
    _tex_bin = "/Library/TeX/texbin"
    if os.path.isdir(_tex_bin):
        os.environ["PATH"] = _tex_bin + os.pathsep + os.environ.get("PATH", "")


LANG_CONFIG = {
    "en": {
        "fontenc": "T1",
        "babel": "english",
        "polyglossia": "english",
        "other_language": "russian",
        "abstract_label": "Abstract",
        "theorem": "Theorem",
        "definition": "Definition",
        "lemma": "Lemma",
        "proposition": "Proposition",
        "corollary": "Corollary",
        "example": "Example",
        "remark": "Remark",
    },
    "ru": {
        "fontenc": "T2A",
        "babel": "russian",
        "polyglossia": "russian",
        "other_language": "english",
        "abstract_label": "\u0410\u043d\u043d\u043e\u0442\u0430\u0446\u0438\u044f",
        "theorem": "Теорема",
        "definition": "Определение",
        "lemma": "Лемма",
        "proposition": "Утверждение",
        "corollary": "Следствие",
        "example": "Пример",
        "remark": "Замечание",
    },
}


def localize_template(tex_template: str, language: str) -> str:
    config = LANG_CONFIG[language]

    replacements = {
        "<FONTENC>": config["fontenc"],
        "<BABEL_LANG>": config["babel"],
        "<POLYGLOSSIA_LANG>": config["polyglossia"],
        "<OTHER_LANG>": config["other_language"],
        "<THEOREM_NAME>": config["theorem"],
        "<DEFINITION_NAME>": config["definition"],
        "<LEMMA_NAME>": config["lemma"],
        "<PROPOSITION_NAME>": config["proposition"],
        "<COROLLARY_NAME>": config["corollary"],
        "<EXAMPLE_NAME>": config["example"],
        "<REMARK_NAME>": config["remark"],
        "<ABSTRACT_LABEL>": config["abstract_label"],
    }

    for placeholder, value in replacements.items():
        tex_template = tex_template.replace(placeholder, value)

    return tex_template


def build_language_setup_block(language: str) -> str:
    config = LANG_CONFIG[language]
    return "\n".join(
        [
            LANGUAGE_PREAMBLE_START,
            r"\usepackage{iftex}",
            r"\ifPDFTeX",
            r"  \usepackage[utf8]{inputenc}",
            fr"  \usepackage[{config['fontenc']}]{{fontenc}}",
            fr"  \usepackage[{config['babel']}]{{babel}}",
            r"\else",
            r"  \usepackage{fontspec}",
            r"  \defaultfontfeatures{Ligatures=TeX,Scale=MatchLowercase}",
            r"  \IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{%",
            r"    \IfFontExistsTF{Noto Serif}{\setmainfont{Noto Serif}}{%",
            r"      \IfFontExistsTF{DejaVu Serif}{\setmainfont{DejaVu Serif}}{%",
            r"        \setmainfont{Latin Modern Roman}%",
            r"      }%",
            r"    }%",
            r"  }",
            r"  \IfFontExistsTF{Arial}{\setsansfont{Arial}}{%",
            r"    \IfFontExistsTF{Noto Sans}{\setsansfont{Noto Sans}}{%",
            r"      \IfFontExistsTF{DejaVu Sans}{\setsansfont{DejaVu Sans}}{%",
            r"        \setsansfont{Latin Modern Sans}%",
            r"      }%",
            r"    }%",
            r"  }",
            r"  \IfFontExistsTF{Consolas}{\setmonofont{Consolas}}{%",
            r"    \IfFontExistsTF{DejaVu Sans Mono}{\setmonofont{DejaVu Sans Mono}}{%",
            r"      \setmonofont{Latin Modern Mono}%",
            r"    }%",
            r"  }",
            r"\fi",
            LANGUAGE_PREAMBLE_END,
        ]
    )


def ensure_multilingual_latex_preamble(latex_content: str, language: str) -> str:
    language_setup = build_language_setup_block(language)
    marked_setup_pattern = re.compile(
        re.escape(LANGUAGE_PREAMBLE_START) + r".*?" + re.escape(LANGUAGE_PREAMBLE_END),
        flags=re.DOTALL,
    )
    if marked_setup_pattern.search(latex_content):
        return marked_setup_pattern.sub(lambda _match: language_setup, latex_content, count=1)

    preamble_cleanup_patterns = [
        r"\\usepackage\{iftex\}\s*",
        r"\\ifPDFTeX\b.*?\\fi\s*",
        r"\\usepackage\[[^\]]*\]\{inputenc\}\s*",
        r"\\usepackage\[[^\]]*\]\{fontenc\}\s*",
        r"\\usepackage\[[^\]]*\]\{babel\}\s*",
        r"\\usepackage\{fontspec\}\s*",
        r"\\setmainlanguage\{[^{}]+\}\s*",
        r"\\setotherlanguage\{[^{}]+\}\s*",
        r"\\defaultfontfeatures\{[^{}]*\}\s*",
        r"\\setmainfont\{[^{}]+\}\s*",
        r"\\setsansfont\{[^{}]+\}\s*",
        r"\\setmonofont\{[^{}]+\}\s*",
        r"\\newfontfamily\\[A-Za-z@]+\{[^{}]+\}\s*",
    ]

    preamble_ready = latex_content
    for pattern in preamble_cleanup_patterns:
        preamble_ready = re.sub(pattern, "", preamble_ready, flags=re.DOTALL)

    documentclass_match = re.search(
        r"(\\documentclass(?:\[[^\]]*\])?\{[^{}]+\})",
        preamble_ready,
    )
    if not documentclass_match:
        return language_setup + "\n" + preamble_ready

    insert_at = documentclass_match.end()
    return (
        preamble_ready[:insert_at]
        + "\n\n"
        + language_setup
        + "\n"
        + preamble_ready[insert_at:]
    )


def latex_engine_available(engine: str) -> bool:
    return shutil.which(engine) is not None


def any_latex_engine_available() -> bool:
    return any(
        latex_engine_available(engine) for engine in ("pdflatex", "xelatex", "lualatex")
    )


def get_latex_base_command(engine: str = "pdflatex") -> list[str]:
    if engine not in _LATEX_BASE_COMMANDS:
        command = [engine]
        try:
            version = subprocess.run(
                [engine, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            if "MiKTeX" in f"{version.stdout}\n{version.stderr}":
                command.append("--disable-installer")
        except Exception:
            pass

        _LATEX_BASE_COMMANDS[engine] = command

    return list(_LATEX_BASE_COMMANDS[engine])


def contains_non_ascii_characters(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def contains_cyrillic_characters(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def contains_unicode_math_characters(text: str) -> bool:
    math_chars = (
        set(UNICODE_GREEK_MAP)
        | set(UNICODE_MATH_OPERATOR_MAP)
        | set(UNICODE_MATH_TEXT_MAP)
        | set(UNICODE_SUBSCRIPT_MAP)
        | set(UNICODE_SUPERSCRIPT_MAP)
    )
    return any(char in math_chars for char in text)


def get_preferred_latex_engines(latex_content: str, language: str) -> list[str]:
    prefer_unicode = (
        latex_engine_available("xelatex")
        or latex_engine_available("lualatex")
        or language == "ru"
        or contains_cyrillic_characters(latex_content)
        or contains_unicode_math_characters(latex_content)
        or contains_non_ascii_characters(latex_content)
    )
    engines = ["xelatex", "lualatex", "pdflatex"] if prefer_unicode else ["pdflatex"]
    return [engine for engine in engines if latex_engine_available(engine)]


def make_ascii_safe_latex(latex_content: str) -> str:
    ascii_safe = latex_content

    for source, replacement in UNICODE_LATEX_REPLACEMENTS.items():
        ascii_safe = ascii_safe.replace(source, replacement)

    converted: list[str] = []
    for char in ascii_safe:
        if char in CYRILLIC_TO_ASCII:
            converted.append(CYRILLIC_TO_ASCII[char])
        elif ord(char) <= 127:
            converted.append(char)
        else:
            converted.append("?")

    return "".join(converted)


def _consume_latex_command(text: str, start_index: int) -> tuple[str, int]:
    if start_index >= len(text) or text[start_index] != "\\":
        return "", start_index

    if start_index + 1 >= len(text):
        return "\\", start_index + 1

    next_char = text[start_index + 1]
    if next_char.isalpha() or next_char == "@":
        end_index = start_index + 2
        while end_index < len(text) and (text[end_index].isalpha() or text[end_index] == "@"):
            end_index += 1
        if end_index < len(text) and text[end_index] == "*":
            end_index += 1
        return text[start_index:end_index], end_index

    return text[start_index : start_index + 2], start_index + 2


def _normalize_script_sequence(sequence: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(char, char) for char in sequence)


def _normalize_math_base(base: str) -> str:
    if len(base) == 1 and base in UNICODE_GREEK_MAP:
        return UNICODE_GREEK_MAP[base]
    if len(base) == 1 and base in UNICODE_MATH_TEXT_MAP:
        return UNICODE_MATH_TEXT_MAP[base]
    return base


def _replace_text_math_tokens(text_segment: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        base = match.group("base")
        sub = match.group("sub") or ""
        sup = match.group("sup") or ""
        is_greek_or_math_symbol = len(base) == 1 and (base in UNICODE_GREEK_MAP or base in UNICODE_MATH_TEXT_MAP)

        if not sub and not sup and not is_greek_or_math_symbol:
            return match.group(0)

        math_token = _normalize_math_base(base)
        if sub:
            math_token += f"_{{{_normalize_script_sequence(sub, UNICODE_SUBSCRIPT_MAP)}}}"
        if sup:
            math_token += f"^{{{_normalize_script_sequence(sup, UNICODE_SUPERSCRIPT_MAP)}}}"
        return rf"\ensuremath{{{math_token}}}"

    return TEXT_MATH_TOKEN_PATTERN.sub(replacer, text_segment)


def normalize_text_unicode_segment(text_segment: str) -> str:
    normalized = _replace_text_math_tokens(text_segment)
    output: list[str] = []
    index = 0

    while index < len(normalized):
        char = normalized[index]

        if char == "\\":
            command, index = _consume_latex_command(normalized, index)
            output.append(command)
            continue

        if char in UNICODE_MATH_TEXT_MAP:
            output.append(rf"\ensuremath{{{UNICODE_MATH_TEXT_MAP[char]}}}")
        elif char in UNICODE_GREEK_MAP:
            output.append(rf"\ensuremath{{{UNICODE_GREEK_MAP[char]}}}")
        elif char in UNICODE_MATH_OPERATOR_MAP:
            output.append(rf"\ensuremath{{{UNICODE_MATH_OPERATOR_MAP[char]}}}")
        elif char in UNICODE_SUBSCRIPT_MAP:
            start = index
            while index < len(normalized) and normalized[index] in UNICODE_SUBSCRIPT_MAP:
                index += 1
            output.append(rf"\ensuremath{{_{{{_normalize_script_sequence(normalized[start:index], UNICODE_SUBSCRIPT_MAP)}}}}}")
            continue
        elif char in UNICODE_SUPERSCRIPT_MAP:
            start = index
            while index < len(normalized) and normalized[index] in UNICODE_SUPERSCRIPT_MAP:
                index += 1
            output.append(rf"\ensuremath{{^{{{_normalize_script_sequence(normalized[start:index], UNICODE_SUPERSCRIPT_MAP)}}}}}")
            continue
        elif char in UNICODE_LATEX_REPLACEMENTS:
            output.append(UNICODE_LATEX_REPLACEMENTS[char])
        else:
            output.append(char)
        index += 1

    return "".join(output)


def normalize_math_unicode_segment(math_segment: str) -> str:
    output: list[str] = []
    index = 0

    while index < len(math_segment):
        char = math_segment[index]

        if char == "\\":
            command, index = _consume_latex_command(math_segment, index)
            output.append(command)
            continue

        if char in UNICODE_MATH_TEXT_MAP:
            output.append(UNICODE_MATH_TEXT_MAP[char])
        elif char in UNICODE_GREEK_MAP:
            output.append(UNICODE_GREEK_MAP[char])
        elif char in UNICODE_MATH_OPERATOR_MAP:
            output.append(UNICODE_MATH_OPERATOR_MAP[char])
        elif char in UNICODE_SUBSCRIPT_MAP:
            start = index
            while index < len(math_segment) and math_segment[index] in UNICODE_SUBSCRIPT_MAP:
                index += 1
            output.append(f"_{{{_normalize_script_sequence(math_segment[start:index], UNICODE_SUBSCRIPT_MAP)}}}")
            continue
        elif char in UNICODE_SUPERSCRIPT_MAP:
            start = index
            while index < len(math_segment) and math_segment[index] in UNICODE_SUPERSCRIPT_MAP:
                index += 1
            output.append(f"^{{{_normalize_script_sequence(math_segment[start:index], UNICODE_SUPERSCRIPT_MAP)}}}")
            continue
        elif char in UNICODE_LATEX_REPLACEMENTS:
            replacement = UNICODE_LATEX_REPLACEMENTS[char]
            if replacement.startswith(r"\ensuremath{") and replacement.endswith("}"):
                replacement = replacement[len(r"\ensuremath{") : -1]
            output.append(replacement)
        else:
            output.append(char)
        index += 1

    return "".join(output)


def split_latex_math_segments(latex_content: str) -> list[tuple[bool, str]]:
    math_patterns = [
        r"\\\(.+?\\\)",
        r"\\\[.+?\\\]",
        r"\$\$.*?\$\$",
        r"(?<!\$)\$(?:\\.|[^$\\])+\$",
    ]
    math_patterns.extend(
        rf"\\begin\{{{re.escape(environment)}\}}.*?\\end\{{{re.escape(environment)}\}}"
        for environment in MATH_ENVIRONMENTS
    )
    math_pattern = re.compile("(" + "|".join(math_patterns) + ")", flags=re.DOTALL)

    segments: list[tuple[bool, str]] = []
    last_index = 0
    for match in math_pattern.finditer(latex_content):
        if match.start() > last_index:
            segments.append((False, latex_content[last_index : match.start()]))
        segments.append((True, match.group(0)))
        last_index = match.end()

    if last_index < len(latex_content):
        segments.append((False, latex_content[last_index:]))

    return segments


def normalize_unicode_latex_document(latex_content: str) -> LatexPreparationResult:
    normalized = unicodedata.normalize("NFC", latex_content)
    notes: list[str] = []
    if normalized != latex_content:
        notes.append("Applied Unicode NFC normalization.")

    rebuilt_segments: list[str] = []
    changed_segments = False
    for is_math, segment in split_latex_math_segments(normalized):
        normalized_segment = (
            normalize_math_unicode_segment(segment)
            if is_math
            else normalize_text_unicode_segment(segment)
        )
        if normalized_segment != segment:
            changed_segments = True
        rebuilt_segments.append(normalized_segment)

    rebuilt = "".join(rebuilt_segments)
    if changed_segments:
        notes.append("Normalized raw Unicode math symbols, Greek letters, and script digits.")

    return LatexPreparationResult(tex=rebuilt, notes=notes)


def simplify_latex_math(math_content: str) -> str:
    simplified = math_content

    while True:
        previous = simplified
        simplified = re.sub(
            r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|operatorname|emph)\{([^{}]*)\}",
            r"\1",
            simplified,
        )
        simplified = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1) / (\2)", simplified)
        simplified = re.sub(
            r"\\sqrt(?:\[[^\]]*\])?\{([^{}]*)\}",
            r"sqrt(\1)",
            simplified,
        )
        if simplified == previous:
            break

    replacements = {
        r"\cdot": "*",
        r"\times": "x",
        r"\to": "->",
        r"\rightarrow": "->",
        r"\leftarrow": "<-",
        r"\Rightarrow": "=>",
        r"\Leftarrow": "<=",
        r"\neq": "!=",
        r"\leq": "<=",
        r"\geq": ">=",
        r"\approx": "~",
        r"\pm": "+/-",
        r"\mp": "-/+",
        r"\infty": "infinity",
        r"\sum": "sum",
        r"\prod": "prod",
        r"\int": "int",
        r"\log": "log",
        r"\ln": "ln",
        r"\sin": "sin",
        r"\cos": "cos",
        r"\tan": "tan",
        r"\ldots": "...",
        r"\dots": "...",
        r"\quad": " ",
        r"\qquad": " ",
        r"\left": "",
        r"\right": "",
        r"\,": " ",
        r"\;": " ",
        r"\:": " ",
        r"\!": "",
    }
    for source, replacement in replacements.items():
        simplified = simplified.replace(source, replacement)

    simplified = re.sub(r"\\(?:label|ref|cref|eqref|cite)\{[^{}]*\}", "", simplified)
    simplified = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", simplified)
    simplified = simplified.replace(r"\{", "{")
    simplified = simplified.replace(r"\}", "}")
    simplified = re.sub(r"[{}]", "", simplified)
    simplified = re.sub(r"\s+", " ", simplified)
    return simplified.strip()


def latex_to_readable_text(latex_content: str) -> str:
    document_match = re.search(
        r"\\begin\{document\}(.*?)\\end\{document\}",
        latex_content,
        flags=re.DOTALL,
    )
    text = document_match.group(1) if document_match else latex_content

    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(
        r"\\begin\{center\}(.*?)\\end\{center\}",
        lambda match: "\n" + match.group(1) + "\n",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(?<!\\)\\\[(.*?)\\\]",
        lambda match: f"\n{simplify_latex_math(match.group(1))}\n",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: simplify_latex_math(match.group(1)),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\$(.+?)\$",
        lambda match: simplify_latex_math(match.group(1)),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"\\\\(?:\[[^\]]*\])?", "\n", text)
    text = text.replace(r"\par", "\n")
    text = re.sub(r"\\item\b", "\n- ", text)

    heading_prefixes = {
        "section": "## ",
        "subsection": "### ",
        "subsubsection": "#### ",
    }
    for command, prefix in heading_prefixes.items():
        text = re.sub(
            rf"\\{command}\*?\{{([^{{}}]*)\}}",
            rf"\n\n{prefix}\1\n",
            text,
        )

    for environment in (
        "equation",
        "equation*",
        "align",
        "align*",
        "gather",
        "gather*",
        "multline",
        "multline*",
        "displaymath",
    ):
        text = re.sub(rf"\\begin\{{{environment}\}}", "\n", text)
        text = re.sub(rf"\\end\{{{environment}\}}", "\n", text)

    for environment in ("enumerate", "itemize", "quote", "center", "customtable", "table", "tabular"):
        text = re.sub(rf"\\begin\{{{environment}\}}(\[[^\]]*\])?(\{{[^{{}}]*\}})*", "\n", text)
        text = re.sub(rf"\\end\{{{environment}\}}", "\n", text)

    box_defaults = {
        "thmbox": "Теорема",
        "defbox": "Определение",
        "lembox": "Лемма",
        "propbox": "Утверждение",
        "corbox": "Следствие",
        "exbox": "Пример",
        "rembox": "Замечание",
    }
    for environment, fallback_title in box_defaults.items():
        text = re.sub(
            rf"\\begin\{{{environment}\}}(?:\[([^\]]*)\])?",
            lambda match: f"\n\n> {match.group(1) or fallback_title}\n",
            text,
        )
        text = re.sub(rf"\\end\{{{environment}\}}", "\n", text)

    text = re.sub(r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:label|ref|cref|eqref|cite)\{[^{}]*\}", "", text)

    while True:
        previous = text
        text = re.sub(
            r"\\(?:text|textbf|textit|texttt|textsf|textrm|emph|underline|mathrm|mathbf|mathit|operatorname|boxed|url)\{([^{}]*)\}",
            r"\1",
            text,
        )
        if text == previous:
            break

    replacements = {
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\#": "#",
        r"\$": "$",
        r"\{": "{",
        r"\}": "}",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)

    text = re.sub(
        r"\b(?:thmbox|defbox|lembox|propbox|corbox|exbox|rembox)\[([^\]]*)\]",
        r"\n\n> \1\n",
        text,
    )
    text = re.sub(r"(?m)^\s*1em\]\s*", "", text)
    text = text.replace("&", " | ")
    text = re.sub(r"\\begin\{[^{}]+\}", "\n", text)
    text = re.sub(r"\\end\{[^{}]+\}", "\n", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text)
    text = re.sub(r"\\.", "", text)
    text = re.sub(r"(?m)^\s*(?:center|itemize|enumerate|quote|tabular|table)\s*$", "", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_pdf_fallback_font_name() -> str:
    global _PDF_FALLBACK_FONT_NAME

    if _PDF_FALLBACK_FONT_NAME is not None:
        return _PDF_FALLBACK_FONT_NAME

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        _PDF_FALLBACK_FONT_NAME = "Helvetica"
        return _PDF_FALLBACK_FONT_NAME

    font_candidates = [
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf"),
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "calibri.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    ]

    for font_path in font_candidates:
        if not os.path.exists(font_path):
            continue
        try:
            pdfmetrics.registerFont(TTFont("ConspectumFallback", font_path))
            _PDF_FALLBACK_FONT_NAME = "ConspectumFallback"
            return _PDF_FALLBACK_FONT_NAME
        except Exception:
            continue

    _PDF_FALLBACK_FONT_NAME = "Helvetica"
    return _PDF_FALLBACK_FONT_NAME


def text_to_pdf_bytes(text: str, title: str | None = None) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4
    left_margin = 54
    right_margin = 54
    top_margin = 56
    bottom_margin = 48
    max_width = page_width - left_margin - right_margin
    body_font = get_pdf_fallback_font_name()
    title_font_size = 18
    body_font_size = 11
    line_height = 15
    y = page_height - top_margin

    def new_page() -> None:
        nonlocal y
        pdf.showPage()
        pdf.setFont(body_font, body_font_size)
        y = page_height - top_margin

    if title:
        pdf.setFont(body_font, title_font_size)
        for line in simpleSplit(title, body_font, title_font_size, max_width):
            if y < bottom_margin + line_height:
                new_page()
                pdf.setFont(body_font, title_font_size)
            pdf.drawString(left_margin, y, line)
            y -= 24
        y -= 8

    def draw_paragraph(
        paragraph: str,
        *,
        font_size: int = body_font_size,
        indent: int = 0,
        gap_after: int = 4,
    ) -> None:
        nonlocal y

        wrapped_lines = simpleSplit(
            paragraph,
            body_font,
            font_size,
            max_width - indent,
        ) or [paragraph]

        pdf.setFont(body_font, font_size)
        for line in wrapped_lines:
            if y < bottom_margin + line_height:
                new_page()
                pdf.setFont(body_font, font_size)
            pdf.drawString(left_margin + indent, y, line)
            y -= max(line_height, font_size + 3)
        y -= gap_after

    for raw_line in text.splitlines():
        paragraph = raw_line.strip()
        if not paragraph:
            y -= 8
            if y < bottom_margin + line_height:
                new_page()
            continue

        if paragraph.startswith("## "):
            draw_paragraph(paragraph[3:].strip(), font_size=16, gap_after=8)
            continue
        if paragraph.startswith("### "):
            draw_paragraph(paragraph[4:].strip(), font_size=14, gap_after=6)
            continue
        if paragraph.startswith("#### "):
            draw_paragraph(paragraph[5:].strip(), font_size=12, gap_after=4)
            continue
        if paragraph.startswith("> "):
            draw_paragraph(paragraph[2:].strip(), font_size=12, indent=10, gap_after=4)
            continue
        if paragraph.startswith("- "):
            draw_paragraph(paragraph, indent=12, gap_after=2)
            continue

        draw_paragraph(paragraph)

    pdf.save()
    return buffer.getvalue()


def latex_to_fallback_pdf(latex_content: str, title: str | None = None) -> bytes:
    readable_text = latex_to_readable_text(latex_content)
    return text_to_pdf_bytes(readable_text, title=title)


def latex_to_pdf(latex_content: str, engine: str = "pdflatex") -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        tex_path = os.path.join(temp_dir, "file.tex")
        pdf_path = os.path.join(temp_dir, "file.pdf")
        diagnostics_lines = [
            f"engine={engine}",
            f"temp_dir={temp_dir}",
            f"tex_path={tex_path}",
        ]

        with open(tex_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(latex_content)

        compiler_command = get_latex_base_command(engine) + [
            "-halt-on-error",
            "-interaction=nonstopmode",
            "-file-line-error",
            "file.tex",
        ]
        diagnostics_lines.append(f"command={' '.join(compiler_command)}")

        last_result: subprocess.CompletedProcess[str] | None = None
        for pass_index in (1, 2):
            result = subprocess.run(
                compiler_command,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=45,
            )
            last_result = result
            diagnostics_lines.append(f"pass={pass_index} returncode={result.returncode}")
            combined_output = f"{result.stdout}\n{result.stderr}".strip()
            if combined_output:
                diagnostics_lines.append(combined_output[-4000:])

            if result.returncode != 0 or not os.path.exists(pdf_path):
                raise LatexCompilationError(
                    engine=engine,
                    summary=format_latex_error(result),
                    diagnostics="\n\n".join(diagnostics_lines),
                )

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        if last_result is not None:
            diagnostics_lines.append(f"pdf_size={len(pdf_bytes)}")

        return pdf_bytes, "\n\n".join(diagnostics_lines)


def sanitize_generated_latex(latex_content: str) -> str:
    sanitized = normalize_latex_text(latex_content)

    replacements = [
        (r"\\begin\{section\}\{([^{}]+)\}", r"\\section{\1}"),
        (r"\\end\{section\}", ""),
        (r"\\begin\{subsection\}\{([^{}]+)\}", r"\\subsection{\1}"),
        (r"\\end\{subsection\}", ""),
        (r"\\begin\{subsubsection\}\{([^{}]+)\}", r"\\subsubsection{\1}"),
        (r"\\end\{subsubsection\}", ""),
        (r"\\begin\{remark\}", r"\\begin{rembox}"),
        (r"\\end\{remark\}", r"\\end{rembox}"),
        (r"\\begin\{definition\}", r"\\begin{defbox}"),
        (r"\\end\{definition\}", r"\\end{defbox}"),
        (r"\\begin\{example\}", r"\\begin{exbox}"),
        (r"\\end\{example\}", r"\\end{exbox}"),
    ]
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized)

    sanitized = re.sub(r"\\begin\{rembox\}\s*\n\s*([^\\\n][^\n]*)\n", r"\\begin{rembox}[Note]\n\1\n", sanitized)
    sanitized = re.sub(r"\\begin\{defbox\}\s*\n\s*([^\\\n][^\n]*)\n", r"\\begin{defbox}[Definition]\n\1\n", sanitized)
    sanitized = re.sub(r"\\begin\{exbox\}\s*\n\s*([^\\\n][^\n]*)\n", r"\\begin{exbox}[Example]\n\1\n", sanitized)

    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()


def escape_unescaped_ampersands(tex_content: str) -> str:
    document_start = tex_content.find(r"\begin{document}")
    document_end = tex_content.rfind(r"\end{document}")

    if document_start == -1 or document_end == -1 or document_end <= document_start:
        return tex_content

    prefix = tex_content[:document_start]
    body = tex_content[document_start:document_end]
    suffix = tex_content[document_end:]

    env_stack: list[str] = []
    escaped_lines: list[str] = []

    for line in body.splitlines():
        begin_envs = re.findall(r"\\begin\{([^{}]+)\}", line)
        end_envs = re.findall(r"\\end\{([^{}]+)\}", line)

        protected = any(env in PROTECTED_AMPERSAND_ENVIRONMENTS for env in env_stack) or any(
            env in PROTECTED_AMPERSAND_ENVIRONMENTS for env in begin_envs
        )

        if not protected:
            line = re.sub(r"(?<!\\)&", r"\\&", line)

        escaped_lines.append(line)

        for env in begin_envs:
            env_stack.append(env)
        for env in end_envs:
            if env in env_stack:
                env_stack.reverse()
                env_stack.remove(env)
                env_stack.reverse()

    return prefix + "\n".join(escaped_lines) + suffix


def prepare_latex_document(tex_content: str, language: str) -> LatexPreparationResult:
    notes: list[str] = []
    repaired = sanitize_generated_latex(tex_content)
    unicode_prepared = normalize_unicode_latex_document(repaired)
    repaired = escape_unescaped_ampersands(unicode_prepared.tex)
    repaired = ensure_multilingual_latex_preamble(repaired, language)
    notes.extend(unicode_prepared.notes)
    notes.append(f"Ensured multilingual LaTeX preamble for {language}.")
    return LatexPreparationResult(tex=repaired.strip(), notes=notes)


def repair_latex_document(tex_content: str, language: str = "en") -> str:
    return prepare_latex_document(tex_content, language).tex


def validate_complete_latex(tex_content: str) -> None:
    required_elements = [
        r"\documentclass",
        r"\begin{document}",
        r"\end{document}",
    ]
    missing = [element for element in required_elements if element not in tex_content]
    if missing:
        raise RuntimeError(f"Incomplete LaTeX document: missing {', '.join(missing)}")


def format_latex_error(result: subprocess.CompletedProcess[str]) -> str:
    combined_output = f"{result.stdout}\n{result.stderr}"
    relevant_lines = []

    for line in combined_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("!"):
            relevant_lines.append(stripped)
        elif "LaTeX Error:" in stripped or "Fatal error occurred" in stripped:
            relevant_lines.append(stripped)
        elif re.search(r"file\.tex:\d+:", stripped):
            relevant_lines.append(stripped)

    if not relevant_lines:
        tail = "\n".join(combined_output.splitlines()[-15:])
        return f"PDF was not generated.\n{tail}".strip()

    return "PDF was not generated.\n" + "\n".join(relevant_lines[:10])


async def compile_latex_pdf(
    latex_content: str,
    language: str,
    logger: Logger,
) -> tuple[typing.Optional[bytes], typing.Optional[str]]:
    engines = get_preferred_latex_engines(latex_content, language)
    if not engines:
        return None, (
            "PDF generation skipped: no LaTeX engine (pdflatex, xelatex, or lualatex) was found in PATH. "
            "Returning only .tex file."
        )

    errors: list[str] = []
    primary_engine = engines[0]
    await logger.partial_result(
        f"Selected {primary_engine} as the primary LaTeX engine for this document."
    )

    for attempt_index, engine in enumerate(engines):
        is_primary = attempt_index == 0
        stage_code = "pdf" if is_primary else "pdf_retry"
        stage_progress = 25 if is_primary else min(95, 45 + attempt_index * 18)
        await logger.stage(stage_code, stage_progress)
        await logger.partial_result(f"Compiling PDF with {engine}...")

        try:
            pdf_bytes, diagnostics = latex_to_pdf(latex_content, engine=engine)
            await logger.file(
                f"latex_compile_{engine}_success",
                diagnostics,
                Logger.FileType.TEXT,
            )
            await logger.stage(stage_code, 100)
            await logger.partial_result(f"PDF compilation succeeded with {engine}.")
            return pdf_bytes, None
        except LatexCompilationError as exc:
            await logger.file(
                f"latex_compile_{engine}_failure",
                exc.diagnostics,
                Logger.FileType.TEXT,
            )
            errors.append(f"{engine}: {exc.summary}")
            if attempt_index < len(engines) - 1:
                await logger.partial_result(
                    f"{engine} could not compile the document. Trying the next LaTeX engine..."
                )
            else:
                await logger.partial_result(
                    f"{engine} could not compile the document."
                )

    return None, "\n".join(errors)


async def split_into_chunks(transcript: str, logger: Logger = Logger()) -> typing.List[str]:
    """Split transcript into text chunks instead of audio chunks."""
    sentences = transcript.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        candidate = current_chunk + sentence

        if len(candidate) > 1500:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
            else:
                chunks.append((sentence.strip() + ". ").strip())
        else:
            current_chunk += sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    for i, chunk in enumerate(chunks):
        await logger.file(f"chunk_{i + 1}_text", chunk, Logger.FileType.TEXT)

    return chunks


async def process_chunk(
    text_chunk: str,
    chunk_num: int,
    total_chunks: int,
    tex_template: str,
    ai: openai.AsyncOpenAI,
    language: str,
    detail_level: str,
    previous_chunk_result: typing.Optional[str] = None,
):
    with open(
        pathlib.Path(__file__).parent / "prompts/system_prompt.txt",
        encoding="utf-8",
        errors="replace",
    ) as prompt_file:
        system_prompt = prompt_file.read()

    system_prompt = system_prompt.replace("<OUTPUT_LANGUAGE>", LANGUAGE_NAMES[language])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"The template:\n\n{tex_template}"},
        {
            "role": "system",
            "content": (
                f"Target detail level: {detail_level}. "
                f"{DETAIL_LEVEL_GUIDANCE.get(detail_level, DETAIL_LEVEL_GUIDANCE['standard'])}"
            ),
        },
        {
            "role": "user",
            "content": f"This is chunk {chunk_num}/{total_chunks} of the lecture transcript:\n\n{text_chunk}",
        },
    ]

    if previous_chunk_result is not None:
        last_section_start = max(
            previous_chunk_result.rfind("\\section"),
            previous_chunk_result.rfind("\\subsection"),
        )

        if last_section_start >= 0:
            last_section = (
                "Previous chunk finished with the following:\n\n"
                f"{previous_chunk_result[last_section_start:]}"
            )
            messages.append({"role": "system", "content": last_section})

    response = await ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
    )

    content = response.choices[0].message.content

    if content is None:
        raise RuntimeError("The model returned an empty response for a chunk.")

    return content


async def process(
    audio_file: bytes,
    ai: openai.AsyncOpenAI,
    logger: Logger = Logger(),
    language: typing.Optional[str] = None,
    detail_level: str = "standard",
    audio_filename: str | None = None,
    audio_mime_type: str | None = None,
) -> ProcessResult:
    if language is not None and language not in ("ru", "en"):
        raise ValueError(f"Unsupported language: {language}. Must be 'ru' or 'en'.")
    if detail_level not in DETAIL_LEVEL_GUIDANCE:
        raise ValueError(
            f"Unsupported detail level: {detail_level}. Must be one of {', '.join(DETAIL_LEVEL_GUIDANCE)}."
        )

    await logger.stage("starting", 40)
    await logger.partial_result("Starting transcription...")
    transcript = await transcribe_audio(
        audio_file,
        logger,
        filename=audio_filename,
        mime_type=audio_mime_type,
    )

    if language is None:
        await logger.stage("detect_language", 20)
        language = await detect_language_from_text(transcript, ai)
        await logger.stage("detect_language", 100)
        await logger.partial_result(f"Detected language: {language}")

    summary = await make_summary_from_transcript(
        transcript,
        ai,
        language,
        logger,
        detail_level=detail_level,
    )

    with open(
        pathlib.Path(__file__).parent / "prompts/template.tex",
        encoding="utf-8",
        errors="replace",
    ) as tex_template_file:
        tex_template = tex_template_file.read()

    tex_template = localize_template(tex_template, language)
    tex_template = tex_template.replace("<INSERT TITLE HERE>", summary.title)
    tex_template = tex_template.replace("<INSERT ABSTRACT HERE>", summary.abstract)

    await logger.file("tex_template", tex_template, Logger.FileType.TEX)

    chunks = await split_into_chunks(transcript, logger)
    await logger.stage("sections", 0)
    await logger.progress(2, 2 + len(chunks))

    results = []

    for i, chunk in enumerate(chunks):
        content = await process_chunk(
            text_chunk=chunk,
            chunk_num=i + 1,
            total_chunks=len(chunks),
            tex_template=tex_template,
            ai=ai,
            language=language,
            detail_level=detail_level,
            previous_chunk_result=results[-1] if results else None,
        )

        await logger.file(f"chunk_{i + 1}", content, Logger.FileType.TEXT)

        results.append(sanitize_generated_latex(content))
        await logger.progress(2 + len(results), 2 + len(chunks))

    tex = tex_template.replace("%% <INSERT CONTENT HERE>", "\n\n".join(results))
    prepared_initial = prepare_latex_document(normalize_latex_text(tex), language)
    tex = prepared_initial.tex
    await logger.file(
        "latex_preparation_before_postprocess",
        "\n".join(prepared_initial.notes) or "No preparation changes were needed.",
        Logger.FileType.TEXT,
    )
    if prepared_initial.notes:
        await logger.partial_result(
            "Normalized Unicode text and formulas for multilingual LaTeX compilation."
        )

    await logger.file("lecture_before_postprocess", tex, Logger.FileType.TEX)

    try:
        tex_postprocessed_raw = await postprocess_summary(tex, ai, language, logger)
        prepared_postprocessed = prepare_latex_document(tex_postprocessed_raw, language)
        tex_postprocessed = prepared_postprocessed.tex
        validate_complete_latex(tex_postprocessed)
        await logger.file(
            "latex_preparation_after_postprocess",
            "\n".join(prepared_postprocessed.notes) or "No preparation changes were needed.",
            Logger.FileType.TEXT,
        )
        await logger.file("lecture", tex_postprocessed, Logger.FileType.TEX)
    except Exception as e:
        await logger.stage("postprocess", 100)
        await logger.partial_result(
            f"Postprocessing failed: {e}. Continuing with original version..."
        )
        tex_postprocessed = tex
        await logger.file("lecture", tex, Logger.FileType.TEX)

    if not any_latex_engine_available():
        await logger.stage("tex_only", 100)
        warning_message = (
            "PDF generation skipped: no LaTeX engine (pdflatex, xelatex, or lualatex) was found in PATH. "
            "Returning only .tex file."
        )
        await logger.partial_result(warning_message)
        return ProcessResult(
            transcript=transcript,
            language=language,
            title=summary.title,
            abstract=summary.abstract,
            tex=tex_postprocessed,
            pdf=None,
            pdf_warning=warning_message,
        )

    pdf_source_tex = tex_postprocessed
    pdf, error_msg = await compile_latex_pdf(pdf_source_tex, language, logger)
    if pdf is not None:
        await logger.file("lecture", pdf, Logger.FileType.PDF)
        return ProcessResult(
            transcript=transcript,
            language=language,
            title=summary.title,
            abstract=summary.abstract,
            tex=pdf_source_tex,
            pdf=pdf,
        )

    fallback_warning = (
        "PDF was generated in a readable fallback layout because LaTeX compilation failed. "
        "The original UTF-8 TEX file is still available."
    )
    try:
        await logger.stage("pdf_retry", 85)
        await logger.partial_result(
            "Retrying PDF generation with a readable fallback layout..."
        )
        pdf = latex_to_fallback_pdf(pdf_source_tex, title=summary.title)
        await logger.stage("pdf_retry", 100)
        await logger.file("lecture_fallback", pdf, Logger.FileType.PDF)
        await logger.partial_result(fallback_warning)
        return ProcessResult(
            transcript=transcript,
            language=language,
            title=summary.title,
            abstract=summary.abstract,
            tex=pdf_source_tex,
            pdf=pdf,
            pdf_warning=fallback_warning,
        )
    except Exception as fallback_pdf_error:
        error_msg = f"Readable PDF fallback also failed: {fallback_pdf_error}"
        await logger.partial_result(error_msg)

    if contains_non_ascii_characters(pdf_source_tex):
        ascii_fallback_tex = make_ascii_safe_latex(pdf_source_tex)
        if ascii_fallback_tex != pdf_source_tex:
            transliteration_warning = (
                "PDF was generated with an ASCII/transliterated fallback because the local "
                "LaTeX installation cannot typeset this document's Unicode characters. "
                "The original UTF-8 TEX file is still available."
            )
            try:
                await logger.stage("pdf_retry", 92)
                await logger.partial_result(
                    "Retrying PDF generation with an ASCII-safe transliteration fallback..."
                )
                pdf, diagnostics = latex_to_pdf(ascii_fallback_tex)
                await logger.stage("pdf_retry", 100)
                await logger.file(
                    "lecture_ascii_fallback",
                    ascii_fallback_tex,
                    Logger.FileType.TEX,
                )
                await logger.file(
                    "latex_compile_ascii_fallback_success",
                    diagnostics,
                    Logger.FileType.TEXT,
                )
                await logger.file("lecture", pdf, Logger.FileType.PDF)
                await logger.partial_result(transliteration_warning)
                return ProcessResult(
                    transcript=transcript,
                    language=language,
                    title=summary.title,
                    abstract=summary.abstract,
                    tex=pdf_source_tex,
                    pdf=pdf,
                    pdf_warning=transliteration_warning,
                )
            except LatexCompilationError as ascii_retry_error:
                await logger.file(
                    "latex_compile_ascii_fallback_failure",
                    ascii_retry_error.diagnostics,
                    Logger.FileType.TEXT,
                )
                error_msg = f"ASCII-safe PDF retry also failed: {ascii_retry_error}"
                await logger.partial_result(error_msg)

    return ProcessResult(
        transcript=transcript,
        language=language,
        title=summary.title,
        abstract=summary.abstract,
        tex=pdf_source_tex,
        pdf=None,
        pdf_warning=error_msg,
    )
