"""SQLite ledger for delayed five-trading-day LightGBM feedback."""

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
                CREATE TABLE IF NOT EXISTS quant_predictions (
                    id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    features_json TEXT NOT NULL,
                    probability_up REAL NOT NULL,
                    decision TEXT NOT NULL,
                    allocation REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    settled_date TEXT,
                    exit_price REAL,
                    forward_return REAL,
                    label_up INTEGER,
                    direction_correct INTEGER,
                    future_volatility REAL,
                    reward REAL,
                    trained_at TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(ticker, prediction_date)
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
        probability_up: float,
        decision: str,
        allocation: float,
        horizon_days: int = 5,
    ) -> str:
        """Stores the first quant prediction for a ticker/trading date."""
        prediction_id = str(uuid4())
        values = (
            prediction_id,
            ticker.upper(),
            prediction_date,
            float(entry_price),
            int(horizon_days),
            json.dumps(features, sort_keys=True),
            float(probability_up),
            decision,
            float(allocation),
            datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO quant_predictions (
                    id, ticker, prediction_date, entry_price, horizon_days,
                    features_json, probability_up, decision, allocation,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            row = connection.execute(
                """
                SELECT id FROM quant_predictions
                WHERE ticker = ? AND prediction_date = ?
                """,
                (ticker.upper(), prediction_date),
            ).fetchone()
        return str(row["id"])

    def pending(self) -> list[dict]:
        return self._select(
            """
            SELECT * FROM quant_predictions
            WHERE status = 'pending'
            ORDER BY prediction_date, ticker
            """
        )

    def settle(
        self,
        prediction_id: str,
        *,
        settled_date: str,
        exit_price: float,
        forward_return: float,
        label_up: int,
        direction_correct: bool,
        future_volatility: float,
        reward: float,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE quant_predictions
                SET status = 'settled', settled_date = ?, exit_price = ?,
                    forward_return = ?, label_up = ?, direction_correct = ?,
                    future_volatility = ?, reward = ?
                WHERE id = ? AND status = 'pending'
                """,
                (
                    settled_date,
                    float(exit_price),
                    float(forward_return),
                    int(label_up),
                    int(direction_correct),
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
        return self._select(
            """
            SELECT * FROM quant_predictions
            WHERE status = 'settled' AND trained_at IS NULL
            ORDER BY settled_date, ticker
            """
        )

    def mark_trained(self, prediction_ids: list[str]) -> None:
        if not prediction_ids:
            return
        placeholders = ",".join("?" for _ in prediction_ids)
        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE quant_predictions
                SET trained_at = ?
                WHERE id IN ({placeholders}) AND status = 'settled'
                """,
                (datetime.now(timezone.utc).isoformat(), *prediction_ids),
            )

    def get(self, prediction_id: str) -> Optional[dict]:
        rows = self._select(
            "SELECT * FROM quant_predictions WHERE id = ?",
            (prediction_id,),
        )
        return rows[0] if rows else None

    def _select(self, query: str, parameters: tuple = ()) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["features"] = json.loads(result.pop("features_json"))
        return result
