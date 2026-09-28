"""SQLite ledger for delayed five-trading-day policy feedback."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

DEFAULT_LEDGER_PATH = Path(
    os.getenv("RL_LEDGER_PATH", Path(__file__).with_name("predictions.db"))
)


class PredictionLedger:
    def __init__(self, path: Path = DEFAULT_LEDGER_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_predictions (
                    id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    features_json TEXT NOT NULL,
                    action TEXT NOT NULL,
                    allocation REAL NOT NULL,
                    predicted_return REAL NOT NULL,
                    confidence REAL NOT NULL,
                    model_version INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    settled_date TEXT,
                    exit_price REAL,
                    forward_return REAL,
                    future_volatility REAL,
                    reward REAL,
                    trained_at TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(ticker, prediction_date, model_version)
                )
                """
            )

    def record(
        self,
        *,
        ticker: str,
        prediction_date: str,
        entry_price: float,
        features: dict,
        signal: dict,
        horizon_days: int = 5,
    ) -> str:
        """Stores one shadow prediction, idempotently per day/model version."""
        prediction_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        values = (
            prediction_id,
            ticker.upper(),
            prediction_date,
            float(entry_price),
            int(horizon_days),
            json.dumps(features, sort_keys=True),
            signal["action"],
            float(signal["allocation"]),
            float(signal["predicted_5d_return"]),
            float(signal["confidence"]),
            int(signal["model_version"]),
            created_at,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO policy_predictions (
                    id, ticker, prediction_date, entry_price, horizon_days,
                    features_json, action, allocation, predicted_return,
                    confidence, model_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            row = connection.execute(
                """
                SELECT id FROM policy_predictions
                WHERE ticker = ? AND prediction_date = ? AND model_version = ?
                """,
                (ticker.upper(), prediction_date, int(signal["model_version"])),
            ).fetchone()
        return str(row["id"])

    def pending(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM policy_predictions
                WHERE status = 'pending'
                ORDER BY prediction_date, ticker
                """
            ).fetchall()
        return [self._decode(row) for row in rows]

    def settle(
        self,
        prediction_id: str,
        *,
        settled_date: str,
        exit_price: float,
        forward_return: float,
        future_volatility: float,
        reward: float,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE policy_predictions
                SET status = 'settled', settled_date = ?, exit_price = ?,
                    forward_return = ?, future_volatility = ?, reward = ?
                WHERE id = ? AND status = 'pending'
                """,
                (
                    settled_date,
                    float(exit_price),
                    float(forward_return),
                    float(future_volatility),
                    float(reward),
                    prediction_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(
                    f"Prediction {prediction_id} is missing or already settled"
                )

    def untrained_settled(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM policy_predictions
                WHERE status = 'settled' AND trained_at IS NULL
                ORDER BY settled_date, ticker
                """
            ).fetchall()
        return [self._decode(row) for row in rows]

    def mark_trained(self, prediction_ids: list[str]) -> None:
        if not prediction_ids:
            return
        trained_at = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in prediction_ids)
        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE policy_predictions
                SET trained_at = ?
                WHERE id IN ({placeholders}) AND status = 'settled'
                """,
                (trained_at, *prediction_ids),
            )

    def get(self, prediction_id: str) -> Optional[dict]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM policy_predictions WHERE id = ?",
                (prediction_id,),
            ).fetchone()
        return self._decode(row) if row else None

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["features"] = json.loads(result.pop("features_json"))
        return result
