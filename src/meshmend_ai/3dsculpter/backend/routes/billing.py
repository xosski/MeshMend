from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from services.subscription_service import REQUIRE_SUBSCRIPTION, subscription_service


router = APIRouter()


class ProvisionRequest(BaseModel):
    email: str
    plan: str = "starter"
    credits: int = 25
    active: bool = True


def _require_admin(admin_token: str | None) -> None:
    configured = os.environ.get("MESHMEND_ADMIN_TOKEN", "")
    if not configured or admin_token != configured:
        raise HTTPException(status_code=403, detail="Admin token required.")


@router.get("/me")
async def billing_me(x_api_key: str | None = Header(default=None)):
    return subscription_service.usage_summary(x_api_key)


@router.get("/status")
async def billing_status():
    return {
        "subscription_required": REQUIRE_SUBSCRIPTION,
        "admin_provisioning_enabled": bool(os.environ.get("MESHMEND_ADMIN_TOKEN")),
    }


@router.post("/provision")
async def provision_account(request: ProvisionRequest, x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    if request.credits < 0:
        raise HTTPException(status_code=400, detail="credits must be non-negative")
    return subscription_service.provision_account(
        email=request.email,
        plan=request.plan,
        credits=request.credits,
        active=request.active,
    )
