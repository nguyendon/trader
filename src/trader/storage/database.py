"""SQLite storage for trades and backtest results."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from loguru import logger

if TYPE_CHECKING:
    from trader.engine.backtest import BacktestResult, Trade


# Default database path
DEFAULT_DB_PATH = Path.home() / ".trader" / "trader.db"


class TradeStore:
    """SQLite-based storage for trades and backtest results."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize the trade store.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.trader/trader.db
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Backtest runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    final_capital REAL NOT NULL,
                    total_return_pct REAL NOT NULL,
                    num_trades INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    max_drawdown_pct REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    reason_entry TEXT,
                    reason_exit TEXT,
                    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
                )
            """)

            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy ON backtest_runs(strategy_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_runs_symbol ON backtest_runs(symbol)
            """)

            conn.commit()
            logger.debug(f"Database initialized at {self.db_path}")

    def save_backtest(self, result: BacktestResult, run_id: str | None = None) -> str:
        """Save a backtest result and its trades.

        Args:
            result: BacktestResult to save
            run_id: Optional custom run ID. Auto-generated if not provided.

        Returns:
            The run_id used for this backtest
        """
        if run_id is None:
            run_id = f"{result.strategy_name}_{result.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert backtest run
            summary = result.summary()
            cursor.execute("""
                INSERT INTO backtest_runs (
                    run_id, strategy_name, symbol, start_date, end_date,
                    initial_capital, final_capital, total_return_pct,
                    num_trades, win_rate, profit_factor, max_drawdown_pct,
                    sharpe_ratio, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                result.strategy_name,
                result.symbol,
                summary["start_date"],
                summary["end_date"],
                summary["initial_capital"],
                summary["final_capital"],
                summary["total_return_pct"],
                summary["num_trades"],
                summary["win_rate"],
                summary["profit_factor"],
                summary["max_drawdown_pct"],
                summary["sharpe_ratio"],
                datetime.now().isoformat(),
            ))

            # Insert trades
            for trade in result.trades:
                cursor.execute("""
                    INSERT INTO trades (
                        run_id, symbol, entry_time, exit_time, side,
                        quantity, entry_price, exit_price, pnl, pnl_pct,
                        reason_entry, reason_exit
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    trade.symbol,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat(),
                    trade.side.value,
                    trade.quantity,
                    float(trade.entry_price),
                    float(trade.exit_price),
                    float(trade.pnl),
                    trade.pnl_pct,
                    trade.reason_entry,
                    trade.reason_exit,
                ))

            conn.commit()
            logger.info(f"Saved backtest run {run_id} with {len(result.trades)} trades")

        return run_id

    def get_backtest_runs(
        self,
        strategy: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get backtest runs with optional filtering.

        Args:
            strategy: Filter by strategy name
            symbol: Filter by symbol
            limit: Maximum number of results

        Returns:
            List of backtest run summaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM backtest_runs WHERE 1=1"
            params: list = []

            if strategy:
                query += " AND strategy_name = ?"
                params.append(strategy)
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_trades(
        self,
        run_id: str | None = None,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get trades with optional filtering.

        Args:
            run_id: Filter by backtest run ID
            symbol: Filter by symbol
            limit: Maximum number of results

        Returns:
            List of trade records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM trades WHERE 1=1"
            params: list = []

            if run_id:
                query += " AND run_id = ?"
                params.append(run_id)
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            query += " ORDER BY entry_time DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_trade_stats(self, symbol: str | None = None) -> dict:
        """Get aggregate trade statistics.

        Args:
            symbol: Optional symbol filter

        Returns:
            Dictionary with aggregate stats
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            where_clause = "WHERE symbol = ?" if symbol else ""
            params = [symbol] if symbol else []

            cursor.execute(f"""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade,
                    AVG(pnl_pct) as avg_pnl_pct
                FROM trades
                {where_clause}
            """, params)

            row = cursor.fetchone()
            if row is None or row["total_trades"] == 0:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "best_trade": 0.0,
                    "worst_trade": 0.0,
                    "avg_pnl_pct": 0.0,
                }

            total = row["total_trades"]
            winning = row["winning_trades"] or 0

            return {
                "total_trades": total,
                "winning_trades": winning,
                "losing_trades": row["losing_trades"] or 0,
                "win_rate": (winning / total * 100) if total > 0 else 0.0,
                "total_pnl": row["total_pnl"] or 0.0,
                "avg_pnl": row["avg_pnl"] or 0.0,
                "best_trade": row["best_trade"] or 0.0,
                "worst_trade": row["worst_trade"] or 0.0,
                "avg_pnl_pct": (row["avg_pnl_pct"] or 0.0) * 100,
            }

    def delete_run(self, run_id: str) -> bool:
        """Delete a backtest run and its trades.

        Args:
            run_id: The run ID to delete

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM trades WHERE run_id = ?", (run_id,))
            cursor.execute("DELETE FROM backtest_runs WHERE run_id = ?", (run_id,))

            deleted = cursor.rowcount > 0
            conn.commit()

            if deleted:
                logger.info(f"Deleted backtest run {run_id}")

            return deleted


# Global store instance
_store: TradeStore | None = None


def get_trade_store(db_path: Path | str | None = None) -> TradeStore:
    """Get the global trade store instance.

    Args:
        db_path: Optional custom database path

    Returns:
        TradeStore instance
    """
    global _store
    if _store is None or db_path is not None:
        _store = TradeStore(db_path)
    return _store
