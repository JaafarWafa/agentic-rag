"""File loaders -> list of (page_number|None, raw_text). Cleaning happens afterwards."""
from __future__ import annotations

import io
import re
from html.parser import HTMLParser


def load_pdf(data: bytes) -> list[tuple[int | None, str]]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


class _HTMLText(HTMLParser):
    SKIP = {"head", "script", "style", "nav", "footer", "header", "aside", "noscript", "form", "button"}
    SKIP_ATTR = re.compile(r"cookie|banner|consent|menu|breadcrumb|sidebar|advert", re.I)
    BLOCK = {"p", "div", "li", "tr", "br", "section", "article", "table", "ul", "ol"}

    def __init__(self):
        super().__init__()
        self.out: list[str] = []
        self.skip = 0
        self.heading: str | None = None

    def handle_starttag(self, tag, attrs):
        boiler = any(k in ("id", "class") and v and self.SKIP_ATTR.search(v) for k, v in attrs)
        if tag in self.SKIP or boiler or self.skip:
            if tag not in self.VOID:
                self.skip += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in self.BLOCK:
            self.out.append("\n\n")

    VOID = {"br", "img", "hr", "meta", "link", "input"}

    def handle_endtag(self, tag):
        if self.skip:
            if tag not in self.VOID:
                self.skip -= 1
        elif re.fullmatch(r"h[1-6]", tag) or tag in self.BLOCK:
            self.out.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def load_html(data: bytes) -> list[tuple[int | None, str]]:
    p = _HTMLText()
    p.feed(data.decode("utf-8", errors="replace"))
    text = "".join(p.out)
    text = re.sub(r"[ \t]+", " ", text)
    return [(None, text)]


def load_text(data: bytes) -> list[tuple[int | None, str]]:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return [(None, data.decode(enc))]
        except UnicodeDecodeError:
            continue
    return [(None, data.decode("utf-8", errors="replace"))]


def load(filename: str, data: bytes) -> tuple[str, list[tuple[int | None, str]]]:
    name = filename.lower()
    if name.endswith(".pdf"):
        return "application/pdf", load_pdf(data)
    if name.endswith((".html", ".htm")):
        return "text/html", load_html(data)
    if name.endswith((".md", ".markdown")):
        return "text/markdown", load_text(data)
    if name.endswith(".txt"):
        return "text/plain", load_text(data)
    raise ValueError(f"unsupported file type: {filename} (pdf, html, md, txt)")
