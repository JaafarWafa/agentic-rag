"""Structure-aware chunking.

1. Detect headings (markdown `#`, numbered `3.2 Title`, ALL-CAPS lines) and keep a
   heading path, e.g. "Leave > Sick leave". Every chunk carries its path: it is
   indexed (BM25 weights it 2x), shown in citations, and gives the LLM context.
2. Pack whole sentences up to a target size; never split mid-sentence.
3. Carry one sentence of overlap between consecutive chunks of the same section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.text import sentences

_MD_HEAD = re.compile(r"^(#{1,6})\s+(.*)$")
_NUM_HEAD = re.compile(r"^((?:\d+\.)*\d+)[.)]?\s+([A-Z][^.!?]{2,80})$")


@dataclass
class Block:
    heading_path: list[str]
    text: str
    page: int | None


def _heading(line: str) -> tuple[int, str] | None:
    s = line.strip()
    if m := _MD_HEAD.match(s):
        return len(m.group(1)), m.group(2).strip(" #")
    if m := _NUM_HEAD.match(s):
        if len(s.split()) <= 12:
            return m.group(1).count(".") + 2, s
    if s.isupper() and 1 <= len(s.split()) <= 10 and re.search(r"[A-Z]{3}", s):
        return 2, s.title()
    return None


def to_blocks(pages: list[tuple[int | None, str]], title: str) -> list[Block]:
    path: list[tuple[int, str]] = []
    blocks: list[Block] = []
    for page, text in pages:
        buf: list[str] = []

        def flush():
            if buf:
                blocks.append(Block([h for _, h in path] or [title], "\n".join(buf).strip(), page))
                buf.clear()

        for para in re.split(r"\n\s*\n", text):
            lines = para.strip().split("\n")
            for line in lines:
                h = _heading(line)
                if h:
                    flush()
                    level, name = h
                    path = [(lv, n) for lv, n in path if lv < level] + [(level, name)]
                elif line.strip():
                    buf.append(line.strip())
            flush()
    return [b for b in blocks if b.text]


def chunk_blocks(blocks: list[Block], target_words: int = 140, max_words: int = 220) -> list[dict]:
    # merge consecutive blocks of the same section first, so short list items stay together
    merged: list[Block] = []
    for b in blocks:
        if merged and merged[-1].heading_path == b.heading_path:
            merged[-1].text += "\n\n" + b.text
        else:
            merged.append(Block(list(b.heading_path), b.text, b.page))

    chunks: list[dict] = []
    for b in merged:
        sents = sentences(b.text)
        cur: list[str] = []
        n = 0

        def emit():
            if cur:
                text = "".join((("\n" if x.startswith("- ") else " ") if j else "") + x for j, x in enumerate(cur))
                chunks.append({"section": " > ".join(b.heading_path), "text": text, "page": b.page})

        for s in sents:
            w = len(s.split())
            if cur and (n + w > max_words or n >= target_words):
                emit()
                overlap = cur[-1] if len(cur[-1].split()) < 40 else None
                cur, n = ([overlap], len(overlap.split())) if overlap else ([], 0)
            cur.append(s)
            n += w
        emit()
    # drop chunks that are pure overlap / too tiny to carry meaning
    return [c for c in chunks if len(c["text"].split()) >= 6]
