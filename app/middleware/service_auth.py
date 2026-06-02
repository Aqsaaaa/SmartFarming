import os

from fastapi import Header, HTTPException, status

RAG_SERVICE_TOKEN: str = os.getenv("RAG_SERVICE_TOKEN", "")


async def verify_service_token(authorization: str | None = Header(None)) -> None:
    """FastAPI dependency that validates the RAG service bearer token.

    Every service-to-service call from Laravel to Python MUST include this
    token. No user credentials or user bearer tokens are accepted.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed Authorization header. Expected 'Bearer <token>'.",
        )

    token = authorization.removeprefix("Bearer ")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is empty.",
        )

    if token != RAG_SERVICE_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service token.",
        )
