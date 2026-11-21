"""Render scripts/pdf_sources/*.txt into deliberately messy PDFs in corpus/raw/.

The PDFs have what real policy PDFs have: running headers and footers with page
numbers, hard line wrapping, and words hyphenated across line breaks - so the
ingestion cleaner has real work to do. Run: python scripts/build_corpus.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "pdf_sources"
OUT = ROOT / "corpus" / "raw"
W, H = A4
FONT, SIZE, LEAD = "Helvetica", 10, 14
MARGIN = 60
MAXW = W - 2 * MARGIN


def wrap(text: str, font=FONT, size=SIZE) -> list[str]:
    """Greedy wrap that hyphenates long words at the line end (like a typesetter)."""
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if stringWidth(trial, font, size) <= MAXW:
            cur = trial
            continue
        if len(word) >= 9:  # split long words with a hyphen
            for cut in range(len(word) - 3, 3, -1):
                head = (cur + " " + word[:cut] + "-").strip()
                if stringWidth(head, font, size) <= MAXW:
                    lines.append(head)
                    cur = word[cut:]
                    break
            else:
                lines.append(cur)
                cur = word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def render(src: Path):
    raw = src.read_text().strip().splitlines()
    title = raw[0].lstrip("# ").strip()
    body: list[tuple[str, str]] = []  # (style, line)
    for line in raw[1:]:
        if not line.strip():
            body.append(("gap", ""))
        elif line.startswith("## "):
            body += [("gap", ""), ("h2", line[3:])]
        else:
            body += [("p", l) for l in wrap(line)]
    # paginate (small pages => several running headers/footers)
    per_page = 17
    pages = [body[i:i + per_page] for i in range(0, len(body), per_page)]
    c = canvas.Canvas(str(OUT / (src.stem + ".pdf")), pagesize=A4)
    c.setTitle(title)
    for pno, page in enumerate(pages, 1):
        c.setFont("Helvetica", 8)
        c.drawString(MARGIN, H - 35, f"Atlas Freight SARL  |  {title}  |  Internal")
        c.drawRightString(W - MARGIN, 30, f"Page {pno} of {len(pages)}")
        c.drawString(MARGIN, 30, "Printed copies are uncontrolled. Check the intranet for the latest version.")
        y = H - 70
        if pno == 1:
            c.setFont("Helvetica-Bold", 16)
            c.drawString(MARGIN, y, title)
            y -= 28
        for style, line in page:
            if style == "gap":
                y -= LEAD // 2
                continue
            c.setFont("Helvetica-Bold" if style == "h2" else FONT, 12 if style == "h2" else SIZE)
            c.drawString(MARGIN, y, line)
            y -= LEAD + (4 if style == "h2" else 0)
        c.showPage()
    c.save()
    print("wrote", OUT / (src.stem + ".pdf"), f"({len(pages)} pages)")


if __name__ == "__main__":
    for s in sorted(SRC.glob("*.txt")):
        render(s)
