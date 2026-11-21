import dataclasses
import os

import pytest

os.environ.setdefault("LLM_PROVIDER", "offline")

from app.config import Settings  # noqa: E402
from app.service import RAGService  # noqa: E402


@pytest.fixture(scope="session")
def svc():
    s = RAGService(dataclasses.replace(Settings(), database_url="sqlite:///:memory:", llm="offline"))
    s.seed()
    return s


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main
    monkeypatch.setattr(main, "svc", RAGService(dataclasses.replace(
        Settings(), database_url=f"sqlite:///{tmp_path / 't.db'}", llm="offline")))
    with TestClient(main.app) as c:
        yield c
