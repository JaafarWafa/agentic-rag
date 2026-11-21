"""Turn messy extracted text into clean paragraphs.

Handles what real corpora throw at you: repeated PDF headers/footers and page
numbers, hyphenation across line breaks, hard-wrapped lines, smart quotes, runs of
whitespace, 'Page 3 of 7', and exact-duplicate paragraphs (copy-pasted boilerplate).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_PAGE_NO = re.compile(r"^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$", re.I)
_TRANS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                        "—": " - ", " ": " ", "•": "-", "": "-", "­": ""})


def _norm_line(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def strip_running_lines(pages: list[str]) -> list[str]:
    """Remove lines that repeat on most pages (headers/footers), ignoring digits."""
    if len(pages) < 2:
        return pages
    def edges(p: str) -> list[str]:  # headers/footers live in the first/last few lines
        lines = [l for l in p.splitlines() if l.strip()]
        return lines[:3] + lines[-3:]

    counts: Counter[str] = Counter()
    for p in pages:
        counts.update({_norm_line(l) for l in edges(p)})
    need = len(pages) if len(pages) < 5 else int(0.6 * len(pages))
    running = {l for l, n in counts.items() if n >= need}
    out = []
    for p in pages:
        edge = set(edges(p))
        out.append("\n".join(l for l in p.splitlines() if not (l in edge and _norm_line(l) in running)))
    return out


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRANS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate across line breaks
    lines = [l.rstrip() for l in text.split("\n") if not _PAGE_NO.match(l)]
    out: list[str] = []
    for line in lines:
        s = re.sub(r"[ \t]{2,}", " ", line).strip()
        prev = out[-1] if out else ""
        is_structural = (not s or s.startswith(("#", "-", "*")) or re.match(r"^\d+(\.\d+)*[.)]?\s", s)
                         or (s.isupper() and len(s.split()) <= 10))
        # re-join hard-wrapped lines: previous line didn't end a sentence and this
        # one continues in lowercase
        if (prev and not is_structural and prev[-1] not in ".:;!?" and not prev.startswith("#")
                and not (prev.isupper() and len(prev.split()) <= 10) and s[:1].islower()):
            out[-1] = prev + " " + s
        else:
            out.append(s)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def dedupe_paragraphs(text: str) -> str:
    seen: set[str] = set()
    keep = []
    for para in re.split(r"\n\s*\n", text):
        key = re.sub(r"\W+", "", para.lower())
        if len(key) > 40 and key in seen:
            continue
        seen.add(key)
        keep.append(para)
    return "\n\n".join(keep)


def clean_pages(pages: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    bodies = strip_running_lines([t for _, t in pages])
    return [(pno, dedupe_paragraphs(clean_text(b))) for (pno, _), b in zip(pages, bodies)]
