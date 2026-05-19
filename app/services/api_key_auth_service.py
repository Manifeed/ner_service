from __future__ import annotations

from fastapi import Header, HTTPException

from app.domain.config import resolve_ner_service_api_key


def require_ner_service_api_key_auth(authorization: str | None = Header(default=None)) -> None:
    expected_api_key = resolve_ner_service_api_key()
    if not expected_api_key:
        return
    if authorization != f"Bearer {expected_api_key}":
        raise HTTPException(status_code=401, detail="Invalid service API key")
