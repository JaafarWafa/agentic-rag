"""Role-based access control.

Every document carries an `allowed_roles` list. A caller's roles are *expanded*
through the hierarchy below, and retrieval filters on the expanded set inside the
database query itself (pre-filtering) so restricted chunks are never scored, never
reach the LLM, and can never leak through a citation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from fastapi import Header, HTTPException

# role -> roles it inherits
ROLE_HIERARCHY: dict[str, set[str]] = {
    "public": set(),
    "employee": {"public"},
    "hr": {"employee", "public"},
    "legal": {"employee", "public"},
    "admin": {"hr", "legal", "employee", "public"},
}
ALL_ROLES = set(ROLE_HIERARCHY)


def expand_roles(roles: list[str] | set[str]) -> list[str]:
    out: set[str] = set()
    for r in roles:
        if r not in ROLE_HIERARCHY:
            continue
        out.add(r)
        out |= ROLE_HIERARCHY[r]
    return sorted(out or {"public"})


@dataclass
class Principal:
    user: str
    roles: list[str]

    @property
    def effective_roles(self) -> list[str]:
        return expand_roles(self.roles)

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles


# Demo keys. In production set API_KEYS='{"<key>": {"user": "...", "roles": [...]}}'
# (or swap this dependency for your OIDC/JWT validator - the rest of the app only
# sees a Principal).
_DEFAULT_KEYS = {
    "public-demo-key": {"user": "visitor", "roles": ["public"]},
    "employee-demo-key": {"user": "sara.employee", "roles": ["employee"]},
    "hr-demo-key": {"user": "youssef.hr", "roles": ["hr"]},
    "legal-demo-key": {"user": "amina.legal", "roles": ["legal"]},
    "admin-demo-key": {"user": "root", "roles": ["admin"]},
}
API_KEYS: dict[str, dict] = json.loads(os.getenv("API_KEYS", "null")) or _DEFAULT_KEYS


def get_principal(x_api_key: str | None = Header(default=None)) -> Principal:
    if x_api_key is None:
        return Principal(user="anonymous", roles=["public"])
    entry = API_KEYS.get(x_api_key)
    if not entry:
        raise HTTPException(status_code=401, detail="invalid API key")
    return Principal(user=entry["user"], roles=list(entry["roles"]))


def require_admin(p: Principal) -> None:
    if not p.is_admin:
        raise HTTPException(status_code=403, detail="admin role required")
