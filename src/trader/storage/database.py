"""SQLite storage for trades and backtest results."""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from trader.engine.backtest import BacktestResult


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

            # Live trades table (for paper/live trading)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS live_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_price REAL,
                    exit_time TEXT,
                    pnl REAL,
                    pnl_pct REAL,
                    status TEXT NOT NULL DEFAULT 'open',
                    strategy TEXT,
                    broker_order_id TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Daily P&L summary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    starting_equity REAL NOT NULL,
                    ending_equity REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0,
                    unrealized_pnl REAL NOT NULL DEFAULT 0,
                    num_trades INTEGER NOT NULL DEFAULT 0,
                    winning_trades INTEGER NOT NULL DEFAULT 0,
                    losing_trades INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
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
                CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy
                ON backtest_runs(strategy_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_runs_symbol
                ON backtest_runs(symbol)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_live_trades_symbol
                ON live_trades(symbol)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_live_trades_status
                ON live_trades(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_pnl_date ON daily_pnl(date)
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
            cursor.execute(
                """
                INSERT INTO backtest_runs (
                    run_id, strategy_name, symbol, start_date, end_date,
                    initial_capital, final_capital, total_return_pct,
                    num_trades, win_rate, profit_factor, max_drawdown_pct,
                    sharpe_ratio, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
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
                ),
            )

            # Insert trades
            for trade in result.trades:
                cursor.execute(
                    """
                    INSERT INTO trades (
                        run_id, symbol, entry_time, exit_time, side,
                        quantity, entry_price, exit_price, pnl, pnl_pct,
                        reason_entry, reason_exit
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
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
                    ),
                )

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

            cursor.execute(
                f"""
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
            """,
                params,
            )

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

    # ==================== Live Trading Methods ====================

    def record_entry(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        strategy: str | None = None,
        broker_order_id: str | None = None,
        notes: str | None = None,
    ) -> int:
        """Record a trade entry (opening a position).

        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            entry_price: Entry price per share
            strategy: Strategy name
            broker_order_id: Broker's order ID
            notes: Optional notes

        Returns:
            Trade ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute(
                """
                INSERT INTO live_trades (
                    symbol, side, quantity, entry_price, entry_time,
                    status, strategy, broker_order_id, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)
            """,
                (
                    symbol,
                    side,
                    quantity,
                    entry_price,
                    now,
                    strategy,
                    broker_order_id,
                    notes,
                    now,
                ),
            )

            conn.commit()
            trade_id = cursor.lastrowid
            logger.info(
                f"Recorded entry: {side} {quantity} {symbol} @ ${entry_price:.2f}"
            )
            return trade_id

    def record_exit(
        self,
        trade_id: int,
        exit_price: float,
        notes: str | None = None,
    ) -> dict:
        """Record a trade exit (closing a position).

        Args:
            trade_id: ID of the trade to close
            exit_price: Exit price per share
            notes: Optional notes

        Returns:
            Updated trade record with P&L
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get the original trade
            cursor.execute("SELECT * FROM live_trades WHERE id = ?", (trade_id,))
            trade = cursor.fetchone()

            if trade is None:
                raise ValueError(f"Trade {trade_id} not found")

            if trade["status"] != "open":
                raise ValueError(f"Trade {trade_id} is already {trade['status']}")

            # Calculate P&L
            entry_price = trade["entry_price"]
            quantity = trade["quantity"]
            side = trade["side"]

            if side == "buy":
                pnl = (exit_price - entry_price) * quantity
            else:  # sell/short
                pnl = (entry_price - exit_price) * quantity

            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            if side == "sell":
                pnl_pct = -pnl_pct

            now = datetime.now().isoformat()
            existing_notes = trade["notes"] or ""
            combined_notes = (
                f"{existing_notes}\n{notes}".strip() if notes else existing_notes
            )

            cursor.execute(
                """
                UPDATE live_trades
                SET exit_price = ?, exit_time = ?, pnl = ?, pnl_pct = ?,
                    status = 'closed', notes = ?
                WHERE id = ?
            """,
                (exit_price, now, pnl, pnl_pct, combined_notes, trade_id),
            )

            conn.commit()
            logger.info(
                f"Recorded exit: {trade['symbol']} @ ${exit_price:.2f}, "
                f"P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)"
            )

            return {
                "trade_id": trade_id,
                "symbol": trade["symbol"],
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }

    def get_open_trades(self, symbol: str | None = None) -> list[dict]:
        """Get all open live trades.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open trade records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if symbol:
                cursor.execute(
                    "SELECT * FROM live_trades WHERE status = 'open' AND symbol = ? "
                    "ORDER BY entry_time DESC",
                    (symbol,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM live_trades WHERE status = 'open' "
                    "ORDER BY entry_time DESC"
                )

            return [dict(row) for row in cursor.fetchall()]

    def get_live_trades(
        self,
        symbol: str | None = None,
        status: str | None = None,
        days: int | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get live trades with optional filtering.

        Args:
            symbol: Filter by symbol
            status: Filter by status ('open', 'closed')
            days: Filter to last N days
            limit: Maximum results

        Returns:
            List of trade records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM live_trades WHERE 1=1"
            params: list = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if status:
                query += " AND status = ?"
                params.append(status)
            if days:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                query += " AND entry_time >= ?"
                params.append(cutoff)

            query += " ORDER BY entry_time DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_live_trade_stats(
        self,
        symbol: str | None = None,
        days: int | None = None,
    ) -> dict:
        """Get aggregate statistics for live trades.

        Args:
            symbol: Filter by symbol
            days: Filter to last N days

        Returns:
            Dictionary with aggregate stats
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            where_clauses = ["status = 'closed'"]
            params: list = []

            if symbol:
                where_clauses.append("symbol = ?")
                params.append(symbol)
            if days:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                where_clauses.append("exit_time >= ?")
                params.append(cutoff)

            where = " AND ".join(where_clauses)

            cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade,
                    AVG(pnl_pct) as avg_pnl_pct,
                    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss
                FROM live_trades
                WHERE {where}
            """,
                params,
            )

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
                    "profit_factor": 0.0,
                }

            total = row["total_trades"]
            winning = row["winning_trades"] or 0
            gross_profit = row["gross_profit"] or 0
            gross_loss = row["gross_loss"] or 0

            return {
                "total_trades": total,
                "winning_trades": winning,
                "losing_trades": row["losing_trades"] or 0,
                "win_rate": (winning / total * 100) if total > 0 else 0.0,
                "total_pnl": row["total_pnl"] or 0.0,
                "avg_pnl": row["avg_pnl"] or 0.0,
                "best_trade": row["best_trade"] or 0.0,
                "worst_trade": row["worst_trade"] or 0.0,
                "avg_pnl_pct": row["avg_pnl_pct"] or 0.0,
                "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else 0.0,
            }

    # ==================== Daily P&L Methods ====================

    def record_daily_pnl(
        self,
        date: str,
        starting_equity: float,
        ending_equity: float,
        realized_pnl: float = 0,
        unrealized_pnl: float = 0,
        num_trades: int = 0,
        winning_trades: int = 0,
        losing_trades: int = 0,
    ) -> None:
        """Record daily P&L summary.

        Args:
            date: Date string (YYYY-MM-DD)
            starting_equity: Equity at market open
            ending_equity: Equity at market close
            realized_pnl: Realized P&L for the day
            unrealized_pnl: Unrealized P&L at close
            num_trades: Number of trades executed
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO daily_pnl (
                    date, starting_equity, ending_equity, realized_pnl,
                    unrealized_pnl, num_trades, winning_trades, losing_trades,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    date,
                    starting_equity,
                    ending_equity,
                    realized_pnl,
                    unrealized_pnl,
                    num_trades,
                    winning_trades,
                    losing_trades,
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            daily_return = ((ending_equity - starting_equity) / starting_equity) * 100
            logger.info(f"Recorded daily P&L for {date}: {daily_return:+.2f}%")

    def get_daily_pnl(self, days: int = 30) -> list[dict]:
        """Get daily P&L history.

        Args:
            days: Number of days to retrieve

        Returns:
            List of daily P&L records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM daily_pnl
                ORDER BY date DESC
                LIMIT ?
            """,
                (days,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_performance_report(self, days: int = 30) -> dict:
        """Generate a performance report for the specified period.

        Args:
            days: Number of days to include

        Returns:
            Dictionary with performance metrics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get daily P&L data
            cursor.execute(
                """
                SELECT * FROM daily_pnl
                ORDER BY date DESC
                LIMIT ?
            """,
                (days,),
            )

            daily_records = [dict(row) for row in cursor.fetchall()]

            if not daily_records:
                return {
                    "period_days": days,
                    "trading_days": 0,
                    "total_pnl": 0.0,
                    "avg_daily_pnl": 0.0,
                    "best_day": 0.0,
                    "worst_day": 0.0,
                    "win_days": 0,
                    "lose_days": 0,
                    "total_trades": 0,
                    "avg_trades_per_day": 0.0,
                }

            # Calculate metrics
            daily_pnls = []
            total_trades = 0

            for record in daily_records:
                daily_pnl = record["ending_equity"] - record["starting_equity"]
                daily_pnls.append(daily_pnl)
                total_trades += record["num_trades"]

            total_pnl = sum(daily_pnls)
            win_days = sum(1 for p in daily_pnls if p > 0)
            lose_days = sum(1 for p in daily_pnls if p < 0)

            return {
                "period_days": days,
                "trading_days": len(daily_records),
                "total_pnl": total_pnl,
                "avg_daily_pnl": total_pnl / len(daily_records) if daily_records else 0,
                "best_day": max(daily_pnls) if daily_pnls else 0,
                "worst_day": min(daily_pnls) if daily_pnls else 0,
                "win_days": win_days,
                "lose_days": lose_days,
                "win_rate": (win_days / len(daily_records) * 100)
                if daily_records
                else 0,
                "total_trades": total_trades,
                "avg_trades_per_day": total_trades / len(daily_records)
                if daily_records
                else 0,
            }

    # ==================== Export Methods ====================

    def export_trades_csv(
        self,
        filepath: Path | str,
        source: str = "all",
        symbol: str | None = None,
        days: int | None = None,
    ) -> int:
        """Export trades to CSV file.

        Args:
            filepath: Output file path
            source: 'backtest', 'live', or 'all'
            symbol: Filter by symbol
            days: Filter to last N days

        Returns:
            Number of trades exported
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        trades = []

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get backtest trades
            if source in ("backtest", "all"):
                query = "SELECT *, 'backtest' as source FROM trades WHERE 1=1"
                params: list = []

                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                if days:
                    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                    query += " AND entry_time >= ?"
                    params.append(cutoff)

                cursor.execute(query, params)
                trades.extend([dict(row) for row in cursor.fetchall()])

            # Get live trades
            if source in ("live", "all"):
                query = """
                    SELECT *, 'live' as source FROM live_trades
                    WHERE status = 'closed'
                """
                params = []

                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                if days:
                    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                    query += " AND exit_time >= ?"
                    params.append(cutoff)

                cursor.execute(query, params)
                trades.extend([dict(row) for row in cursor.fetchall()])

        if not trades:
            logger.warning("No trades to export")
            return 0

        # Write CSV
        fieldnames = [
            "source",
            "symbol",
            "side",
            "quantity",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "pnl",
            "pnl_pct",
            "strategy",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(trades)

        logger.info(f"Exported {len(trades)} trades to {filepath}")
        return len(trades)

    def export_daily_pnl_csv(
        self,
        filepath: Path | str,
        days: int = 365,
    ) -> int:
        """Export daily P&L to CSV file.

        Args:
            filepath: Output file path
            days: Number of days to export

        Returns:
            Number of records exported
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        records = self.get_daily_pnl(days)

        if not records:
            logger.warning("No daily P&L records to export")
            return 0

        fieldnames = [
            "date",
            "starting_equity",
            "ending_equity",
            "realized_pnl",
            "unrealized_pnl",
            "num_trades",
            "winning_trades",
            "losing_trades",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"Exported {len(records)} daily P&L records to {filepath}")
        return len(records)


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
