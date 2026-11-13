"""
File: settings.py
Project: Smart Health Monitoring System (REVA University - R24EJ806)
Author: Vasanth A N
Description: Shared configuration and bearer-token authentication for the cloud
             backend (Module 3) and caregiver dashboard (Module 4). Keeping these
             here removes the circular import between the two modules and keeps
             secrets out of source code.
"""
import os
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

DB_PATH = os.environ.get("HEALTH_DB_PATH", "telemetry_audit.db")
MODEL_PATH = os.environ.get("HEALTH_MODEL_PATH", "hybrid_model.joblib")
DEFAULT_DEVICE_ID = os.environ.get("HEALTH_DEFAULT_DEVICE", "DEV_ESP32_NODE_03")

# Two roles: edge devices may only write telemetry, caregivers may only read it.
DEVICE_TOKEN_ENV = "HEALTH_DEVICE_TOKEN"
CAREGIVER_TOKEN_ENV = "HEALTH_CAREGIVER_TOKEN"

_bearer = HTTPBearer(auto_error=False)


def _token_from_env(env_name: str) -> str:
    token = os.environ.get(env_name)
    if not token:
        raise RuntimeError(f"Environment variable {env_name} is not set")
    return token


def check_tokens_configured() -> None:
    """Fail fast at startup instead of accepting requests with no secret configured."""
    _token_from_env(DEVICE_TOKEN_ENV)
    _token_from_env(CAREGIVER_TOKEN_ENV)


def _verify(creds: Optional[HTTPAuthorizationCredentials], env_name: str) -> None:
    expected = _token_from_env(env_name)
    if (
        creds is None
        or creds.scheme.lower() != "bearer"
        or not secrets.compare_digest(creds.credentials.encode(), expected.encode())
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_device_token(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> None:
    _verify(creds, DEVICE_TOKEN_ENV)


def require_caregiver_token(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> None:
    _verify(creds, CAREGIVER_TOKEN_ENV)
