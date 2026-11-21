from app.ingest.chunking import chunk_blocks, to_blocks
from app.ingest.cleaning import clean_pages, clean_text, dedupe_paragraphs, strip_running_lines
from app.ingest.loaders import load_html


def test_running_headers_and_page_numbers_removed():
    body = ["Intro line.", "Middle {i} line one.", "Middle {i} line two.", "Middle {i} line three.", "Outro."]
    pages = ["ACME Corp | Policy | Internal\n" + "\n".join(b.format(i=i) for b in body) + f"\nPage {i} of 3"
             for i in range(1, 4)]
    out = strip_running_lines(pages)
    assert all("ACME Corp" not in p and "Page" not in p for p in out)
    # body lines that differ only by a number are NOT headers when they're mid-page
    assert "Middle 2 line two." in out[1]


def test_dehyphenation_and_rewrap():
    assert clean_text("the reimburse-\nment is paid\nwithin 30 days.") == "the reimbursement is paid within 30 days."


def test_duplicate_paragraphs_dropped():
    para = "Report suspicious emails with the Report Phishing button in Outlook right away."
    assert dedupe_paragraphs(f"{para}\n\n{para}\n\nOther.").count("Report Phishing") == 1


def test_html_boilerplate_stripped():
    html = (b"<html><head><title>T</title><script>x=1</script></head><body><nav>Home | HR</nav>"
            b"<div id='cookie-banner'>We use cookies</div><h2>Leave</h2><p>You get 18 days.</p>"
            b"<footer>(c) 2026</footer></body></html>")
    text = load_html(html)[0][1]
    assert "18 days" in text and "cookies" not in text and "Home" not in text and "x=1" not in text


def test_chunks_carry_heading_path_and_respect_size():
    text = "# Handbook\n\n## Leave\n\n" + " ".join(f"Sentence number {i} is here." for i in range(80))
    chunks = chunk_blocks(to_blocks(clean_pages([(None, text)]), "Doc"), target_words=60, max_words=90)
    assert len(chunks) > 1
    assert all(c["section"] == "Handbook > Leave" for c in chunks)
    assert all(len(c["text"].split()) <= 90 for c in chunks)
    # one sentence of overlap between consecutive chunks
    assert chunks[0]["text"].split(". ")[-1].rstrip(".") in chunks[1]["text"]


def test_seed_corpus_is_idempotent(svc):
    before = svc.store.count_chunks()
    results = svc.seed()
    assert all(r["status"] == "duplicate" for r in results)
    assert svc.store.count_chunks() == before


def test_pdf_ingestion_end_to_end(svc):
    docs = {d["source"]: d for d in svc.store.list_documents(["admin", "hr", "legal", "employee", "public"])}
    pdf = docs["travel_expense_policy.pdf"]
    chunks = svc.store.get_chunks(pdf["id"], ["employee"])
    joined = " ".join(c.text for c in chunks)
    assert "Printed copies are uncontrolled" not in joined  # footer removed
    assert "Page 1 of" not in joined
    assert any(c.page == 2 for c in chunks)  # page numbers kept as metadata
    assert any(c.section.startswith("4. Accommodation") for c in chunks)
