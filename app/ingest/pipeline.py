"""Ingestion pipeline: load -> clean -> chunk -> embed -> store (idempotent by content hash)."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from app.auth import ALL_ROLES
from app.db.base import Store
from app.ingest.chunking import chunk_blocks, to_blocks
from app.ingest.cleaning import clean_pages
from app.ingest.loaders import load
from app.retrieval.embeddings import Embedder

log = logging.getLogger("rag.ingest")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def ingest_bytes(store: Store, embedder: Embedder, filename: str, data: bytes, allowed_roles: list[str],
                 title: str | None = None, *, target_words: int = 140, max_words: int = 220) -> dict:
    bad = set(allowed_roles) - ALL_ROLES
    if bad or not allowed_roles:
        raise ValueError(f"invalid allowed_roles {sorted(bad) or allowed_roles}; choose from {sorted(ALL_ROLES)}")
    sha = hashlib.sha256(data).hexdigest()
    if existing := store.find_document_by_hash(sha):
        return {**existing, "status": "duplicate"}

    mime, pages = load(filename, data)
    pages = clean_pages(pages)
    title = title or _guess_title(pages) or Path(filename).stem.replace("_", " ").title()
    blocks = to_blocks(pages, title)
    chunks = chunk_blocks(blocks, target_words, max_words)
    if not chunks:
        raise ValueError(f"{filename}: no extractable text")

    doc_id = f"{_slug(Path(filename).stem)}-{sha[:8]}"
    for i, c in enumerate(chunks):
        c["id"], c["ord"] = f"{doc_id}#{i:03d}", i
    # Embed title + section + text so short chunks still carry their context.
    vecs = embedder.embed([f"{title}. {c['section']}. {c['text']}" for c in chunks])
    doc = {"id": doc_id, "title": title, "source": filename, "mime": mime, "sha256": sha,
           "allowed_roles": sorted(set(allowed_roles)), "n_chars": sum(len(t) for _, t in pages)}
    store.add_document(doc, chunks, vecs)
    log.info("ingested %s: %d chunks roles=%s", filename, len(chunks), doc["allowed_roles"])
    return {**doc, "n_chunks": len(chunks), "status": "ingested"}


def _guess_title(pages) -> str | None:
    for _, text in pages:
        for line in text.splitlines():
            s = line.strip().lstrip("#").strip()
            if 3 <= len(s) <= 90:
                return s
    return None


def ingest_directory(store: Store, embedder: Embedder, directory: Path, **kw) -> list[dict]:
    """Ingest a folder using manifest.json: {"file.pdf": {"roles": [...], "title": "..."}}."""
    manifest = json.loads((directory / "manifest.json").read_text())
    out = []
    for fname, meta in manifest.items():
        out.append(ingest_bytes(store, embedder, fname, (directory / fname).read_bytes(),
                                meta["roles"], meta.get("title"), **kw))
    return out
