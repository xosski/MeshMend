from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.config import BACKEND_DIR


SUBSCRIPTION_DB_PATH = Path(os.environ.get("MESHMEND_SUBSCRIPTION_DB", BACKEND_DIR / "subscriptions.sqlite3"))
REQUIRE_SUBSCRIPTION = os.environ.get("MESHMEND_REQUIRE_SUBSCRIPTION", "").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Account:
    id: int
    email: str
    plan: str
    active: bool
    credits_remaining: int


class SubscriptionService:
    """Small local account/credit gate for downloadable installs.

    This is intentionally provider-agnostic and does not require a hosted
    backend. A future license portal, emailed keys, or offline activation flow
    can call `provision_account`/`add_credits`; generation routes only need to
    validate a token and consume credits atomically.
    """

    def __init__(self, db_path: Path = SUBSCRIPTION_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with closing(self._connect()) as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    token_hash TEXT NOT NULL UNIQUE,
                    plan TEXT NOT NULL DEFAULT 'starter',
                    active INTEGER NOT NULL DEFAULT 1,
                    credits_remaining INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            db.commit()
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER,
                    operation TEXT NOT NULL,
                    credits INTEGER NOT NULL,
                    provider TEXT,
                    provider_task_id TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(account_id) REFERENCES accounts(id)
                )
                """
            )
            db.commit()

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256((token or "").encode("utf-8")).hexdigest()

    def provision_account(self, email: str, plan: str = "starter", credits: int = 25, active: bool = True) -> dict[str, Any]:
        token = "mm_" + secrets.token_urlsafe(32)
        token_hash = self.hash_token(token)
        with closing(self._connect()) as db:
            db.execute(
                """
                INSERT INTO accounts(email, token_hash, plan, active, credits_remaining)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    token_hash=excluded.token_hash,
                    plan=excluded.plan,
                    active=excluded.active,
                    credits_remaining=excluded.credits_remaining,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (email.strip().lower(), token_hash, plan, 1 if active else 0, int(credits)),
            )
            db.commit()
        account = self.account_from_token(token)
        return {"token": token, "account": account.__dict__ if account else None}

    def account_from_token(self, token: str | None) -> Account | None:
        if not token:
            return None
        token_hash = self.hash_token(token.strip())
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT id, email, plan, active, credits_remaining FROM accounts WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
        if not row:
            return None
        return Account(
            id=int(row["id"]),
            email=str(row["email"]),
            plan=str(row["plan"]),
            active=bool(row["active"]),
            credits_remaining=int(row["credits_remaining"]),
        )

    def authorize_generation(self, token: str | None, estimated_credits: int, operation: str) -> Account | None:
        if not REQUIRE_SUBSCRIPTION:
            return self.account_from_token(token) if token else None
        account = self.account_from_token(token)
        if account is None:
            raise PermissionError("A paid/local license key is required for premium 3D generation.")
        if not account.active:
            raise PermissionError("Subscription is inactive.")
        if account.credits_remaining < int(estimated_credits):
            raise RuntimeError("Insufficient generation credits.")
        return account

    def consume_credits(
        self,
        account: Account | None,
        credits: int,
        operation: str,
        provider: str = "",
        provider_task_id: str = "",
        metadata: str = "",
    ) -> None:
        if account is None:
            return
        with closing(self._connect()) as db:
            updated = db.execute(
                """
                UPDATE accounts
                SET credits_remaining = credits_remaining - ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND active = 1 AND credits_remaining >= ?
                """,
                (int(credits), int(account.id), int(credits)),
            ).rowcount
            if updated != 1:
                raise RuntimeError("Insufficient generation credits.")
            db.execute(
                """
                INSERT INTO usage_events(account_id, operation, credits, provider, provider_task_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (account.id, operation, int(credits), provider, provider_task_id, metadata),
            )
            db.commit()

    def usage_summary(self, token: str | None) -> dict[str, Any]:
        account = self.account_from_token(token)
        if account is None:
            return {"authenticated": False, "subscription_required": REQUIRE_SUBSCRIPTION}
        with closing(self._connect()) as db:
            rows = db.execute(
                """
                SELECT operation, SUM(credits) AS credits, COUNT(*) AS count
                FROM usage_events WHERE account_id = ? GROUP BY operation
                """,
                (account.id,),
            ).fetchall()
        return {
            "authenticated": True,
            "subscription_required": REQUIRE_SUBSCRIPTION,
            "account": account.__dict__,
            "usage": [dict(row) for row in rows],
        }


subscription_service = SubscriptionService()
