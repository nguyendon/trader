"""Trading scheduler with cron-like scheduling and market triggers."""

from __future__ import annotations

import asyncio
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from loguru import logger

if TYPE_CHECKING:
    from trader.broker.base import BaseBroker
    from trader.data.fetcher import BaseDataFetcher
    from trader.notifications.discord import DiscordNotifier
    from trader.risk.manager import RiskManager
    from trader.strategies.base import BaseStrategy


class ScheduleStatus(str, Enum):
    """Status of a schedule."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"  # For one-time schedules
    ERROR = "error"


class MarketTrigger(str, Enum):
    """Market-based triggers for schedules."""

    MARKET_OPEN = "market_open"  # 9:30 AM ET
    MARKET_CLOSE = "market_close"  # 4:00 PM ET
    PRE_MARKET = "pre_market"  # 4:00 AM ET
    AFTER_HOURS = "after_hours"  # 4:00 PM ET
    MINUTES_AFTER_OPEN = "minutes_after_open"  # N minutes after open
    MINUTES_BEFORE_CLOSE = "minutes_before_close"  # N minutes before close


@dataclass
class ScheduleConfig:
    """Configuration for a trading schedule."""

    # When to run
    cron_expression: str | None = None  # e.g., "30 9 * * 1-5" (9:30 AM weekdays)
    time_of_day: time | None = None  # Simple time trigger
    market_trigger: MarketTrigger | None = None  # Market-based trigger
    trigger_offset_minutes: int = 0  # Offset for market triggers

    # What to run
    symbols: list[str] = field(default_factory=list)
    strategy_name: str = "sma"
    action: str = "run"  # "run" (generate signals), "backtest", "scan"

    # Run options
    run_once: bool = False  # One-time schedule
    enabled_days: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4]
    )  # Mon-Fri (0=Monday)
    skip_holidays: bool = True
    paper_mode: bool = True  # Use paper trading

    # Execution settings
    max_retries: int = 3
    retry_delay_seconds: int = 60

    @property
    def trigger_description(self) -> str:
        """Human-readable trigger description."""
        if self.cron_expression:
            return f"Cron: {self.cron_expression}"
        if self.time_of_day:
            return f"Daily at {self.time_of_day.strftime('%H:%M')}"
        if self.market_trigger:
            desc = self.market_trigger.value.replace("_", " ").title()
            if self.trigger_offset_minutes:
                if self.trigger_offset_minutes > 0:
                    desc += f" +{self.trigger_offset_minutes}m"
                else:
                    desc += f" {self.trigger_offset_minutes}m"
            return desc
        return "Manual"


@dataclass
class Schedule:
    """A scheduled trading task."""

    id: str
    name: str
    config: ScheduleConfig
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0
    error_count: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "symbols": ",".join(self.config.symbols),
            "strategy_name": self.config.strategy_name,
            "action": self.config.action,
            "cron_expression": self.config.cron_expression,
            "time_of_day": (
                self.config.time_of_day.isoformat() if self.config.time_of_day else None
            ),
            "market_trigger": (
                self.config.market_trigger.value if self.config.market_trigger else None
            ),
            "trigger_offset_minutes": self.config.trigger_offset_minutes,
            "run_once": self.config.run_once,
            "enabled_days": ",".join(map(str, self.config.enabled_days)),
            "paper_mode": self.config.paper_mode,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Schedule:
        """Create from dictionary."""
        config = ScheduleConfig(
            symbols=data["symbols"].split(",") if data["symbols"] else [],
            strategy_name=data["strategy_name"],
            action=data["action"],
            cron_expression=data.get("cron_expression"),
            time_of_day=(
                time.fromisoformat(data["time_of_day"]) if data.get("time_of_day") else None
            ),
            market_trigger=(
                MarketTrigger(data["market_trigger"])
                if data.get("market_trigger")
                else None
            ),
            trigger_offset_minutes=data.get("trigger_offset_minutes", 0),
            run_once=data.get("run_once", False),
            enabled_days=(
                [int(d) for d in data["enabled_days"].split(",")]
                if data.get("enabled_days")
                else [0, 1, 2, 3, 4]
            ),
            paper_mode=data.get("paper_mode", True),
        )

        return cls(
            id=data["id"],
            name=data["name"],
            config=config,
            status=ScheduleStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_run=(
                datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None
            ),
            next_run=(
                datetime.fromisoformat(data["next_run"]) if data.get("next_run") else None
            ),
            run_count=data.get("run_count", 0),
            error_count=data.get("error_count", 0),
            last_error=data.get("last_error"),
        )


class CronParser:
    """Simple cron expression parser for trading schedules.

    Supports: minute hour day month weekday
    - minute: 0-59
    - hour: 0-23
    - day: 1-31
    - month: 1-12
    - weekday: 0-6 (0=Monday) or 1-5 range

    Special characters: * (any), , (list), - (range), / (step)
    """

    @staticmethod
    def parse_field(field: str, min_val: int, max_val: int) -> set[int]:
        """Parse a single cron field into a set of valid values."""
        values: set[int] = set()

        for part in field.split(","):
            # Handle step values
            step = 1
            if "/" in part:
                part, step_str = part.split("/")
                step = int(step_str)

            # Handle range or wildcard
            if part == "*":
                values.update(range(min_val, max_val + 1, step))
            elif "-" in part:
                start, end = map(int, part.split("-"))
                values.update(range(start, end + 1, step))
            else:
                values.add(int(part))

        return values

    @classmethod
    def parse(cls, expression: str) -> dict[str, set[int]]:
        """Parse a cron expression into field sets."""
        parts = expression.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression: {expression}. "
                "Expected 5 fields: minute hour day month weekday"
            )

        return {
            "minute": cls.parse_field(parts[0], 0, 59),
            "hour": cls.parse_field(parts[1], 0, 23),
            "day": cls.parse_field(parts[2], 1, 31),
            "month": cls.parse_field(parts[3], 1, 12),
            "weekday": cls.parse_field(parts[4], 0, 6),
        }

    @classmethod
    def get_next_run(
        cls,
        expression: str,
        after: datetime | None = None,
        tz: ZoneInfo | None = None,
    ) -> datetime:
        """Calculate the next run time for a cron expression."""
        if after is None:
            after = datetime.now(tz)
        elif tz and after.tzinfo is None:
            after = after.replace(tzinfo=tz)

        fields = cls.parse(expression)

        # Start from next minute
        current = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Find next matching time (limit iterations to prevent infinite loop)
        for _ in range(366 * 24 * 60):  # Max 1 year of minutes
            if (
                current.minute in fields["minute"]
                and current.hour in fields["hour"]
                and current.day in fields["day"]
                and current.month in fields["month"]
                and current.weekday() in fields["weekday"]
            ):
                return current

            current += timedelta(minutes=1)

        raise ValueError(f"Could not find next run time for: {expression}")


class ScheduleStore:
    """SQLite storage for schedules."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize schedule store."""
        if db_path is None:
            db_path = Path.home() / ".trader" / "schedules.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection."""
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

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schedules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    symbols TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    action TEXT NOT NULL DEFAULT 'run',
                    cron_expression TEXT,
                    time_of_day TEXT,
                    market_trigger TEXT,
                    trigger_offset_minutes INTEGER DEFAULT 0,
                    run_once INTEGER DEFAULT 0,
                    enabled_days TEXT DEFAULT '0,1,2,3,4',
                    paper_mode INTEGER DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    last_run TEXT,
                    next_run TEXT,
                    run_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    last_error TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schedule_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    schedule_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    signals_generated INTEGER DEFAULT 0,
                    trades_executed INTEGER DEFAULT 0,
                    error_message TEXT,
                    FOREIGN KEY (schedule_id) REFERENCES schedules(id)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_schedule_runs_schedule
                ON schedule_runs(schedule_id)
            """)

            conn.commit()

    def save_schedule(self, schedule: Schedule) -> None:
        """Save or update a schedule."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data = schedule.to_dict()

            cursor.execute(
                """
                INSERT OR REPLACE INTO schedules (
                    id, name, symbols, strategy_name, action,
                    cron_expression, time_of_day, market_trigger,
                    trigger_offset_minutes, run_once, enabled_days,
                    paper_mode, status, created_at, last_run, next_run,
                    run_count, error_count, last_error
                ) VALUES (
                    :id, :name, :symbols, :strategy_name, :action,
                    :cron_expression, :time_of_day, :market_trigger,
                    :trigger_offset_minutes, :run_once, :enabled_days,
                    :paper_mode, :status, :created_at, :last_run, :next_run,
                    :run_count, :error_count, :last_error
                )
            """,
                data,
            )
            conn.commit()

    def get_schedule(self, schedule_id: str) -> Schedule | None:
        """Get a schedule by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,))
            row = cursor.fetchone()
            return Schedule.from_dict(dict(row)) if row else None

    def get_all_schedules(
        self, status: ScheduleStatus | None = None
    ) -> list[Schedule]:
        """Get all schedules, optionally filtered by status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM schedules WHERE status = ? ORDER BY created_at",
                    (status.value,),
                )
            else:
                cursor.execute("SELECT * FROM schedules ORDER BY created_at")
            return [Schedule.from_dict(dict(row)) for row in cursor.fetchall()]

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM schedule_runs WHERE schedule_id = ?", (schedule_id,))
            cursor.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def record_run(
        self,
        schedule_id: str,
        status: str,
        signals: int = 0,
        trades: int = 0,
        error: str | None = None,
    ) -> None:
        """Record a schedule run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute(
                """
                INSERT INTO schedule_runs (
                    schedule_id, started_at, completed_at, status,
                    signals_generated, trades_executed, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (schedule_id, now, now, status, signals, trades, error),
            )
            conn.commit()

    def get_run_history(self, schedule_id: str, limit: int = 20) -> list[dict]:
        """Get run history for a schedule."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM schedule_runs
                WHERE schedule_id = ?
                ORDER BY started_at DESC
                LIMIT ?
            """,
                (schedule_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]


class TradingScheduler:
    """Main scheduler for running trading tasks on schedule.

    Features:
    - Cron-like scheduling (e.g., "30 9 * * 1-5" for 9:30 AM weekdays)
    - Market-based triggers (open, close, pre-market, after-hours)
    - Multiple schedules running concurrently
    - Persistent schedule storage
    - Retry on failure
    - Holiday awareness (US market holidays)
    """

    # US market holidays for 2024-2025
    HOLIDAYS = {
        # 2024
        datetime(2024, 1, 1).date(),  # New Year's Day
        datetime(2024, 1, 15).date(),  # MLK Day
        datetime(2024, 2, 19).date(),  # Presidents Day
        datetime(2024, 3, 29).date(),  # Good Friday
        datetime(2024, 5, 27).date(),  # Memorial Day
        datetime(2024, 6, 19).date(),  # Juneteenth
        datetime(2024, 7, 4).date(),  # Independence Day
        datetime(2024, 9, 2).date(),  # Labor Day
        datetime(2024, 11, 28).date(),  # Thanksgiving
        datetime(2024, 12, 25).date(),  # Christmas
        # 2025
        datetime(2025, 1, 1).date(),  # New Year's Day
        datetime(2025, 1, 20).date(),  # MLK Day
        datetime(2025, 2, 17).date(),  # Presidents Day
        datetime(2025, 4, 18).date(),  # Good Friday
        datetime(2025, 5, 26).date(),  # Memorial Day
        datetime(2025, 6, 19).date(),  # Juneteenth
        datetime(2025, 7, 4).date(),  # Independence Day
        datetime(2025, 9, 1).date(),  # Labor Day
        datetime(2025, 11, 27).date(),  # Thanksgiving
        datetime(2025, 12, 25).date(),  # Christmas
    }

    # Market hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    PRE_MARKET_OPEN = time(4, 0)
    AFTER_HOURS_CLOSE = time(20, 0)

    def __init__(
        self,
        store: ScheduleStore | None = None,
        broker: BaseBroker | None = None,
        data_fetcher: BaseDataFetcher | None = None,
        risk_manager: RiskManager | None = None,
        notifier: DiscordNotifier | None = None,
        timezone: str = "America/New_York",
    ) -> None:
        """Initialize the scheduler."""
        self.store = store or ScheduleStore()
        self.broker = broker
        self.data_fetcher = data_fetcher
        self.risk_manager = risk_manager
        self.notifier = notifier
        self.tz = ZoneInfo(timezone)

        self._running = False
        self._stop_event = asyncio.Event()
        self._active_tasks: dict[str, asyncio.Task] = {}

    def is_holiday(self, date: datetime) -> bool:
        """Check if a date is a US market holiday."""
        return date.date() in self.HOLIDAYS

    def is_trading_day(self, date: datetime) -> bool:
        """Check if a date is a trading day (weekday, not holiday)."""
        return date.weekday() < 5 and not self.is_holiday(date)

    def get_market_time(
        self,
        trigger: MarketTrigger,
        date: datetime,
        offset_minutes: int = 0,
    ) -> datetime:
        """Get the market trigger time for a given date."""
        base_time = {
            MarketTrigger.MARKET_OPEN: self.MARKET_OPEN,
            MarketTrigger.MARKET_CLOSE: self.MARKET_CLOSE,
            MarketTrigger.PRE_MARKET: self.PRE_MARKET_OPEN,
            MarketTrigger.AFTER_HOURS: self.MARKET_CLOSE,
            MarketTrigger.MINUTES_AFTER_OPEN: self.MARKET_OPEN,
            MarketTrigger.MINUTES_BEFORE_CLOSE: self.MARKET_CLOSE,
        }[trigger]

        result = datetime.combine(date.date(), base_time, tzinfo=self.tz)

        # Apply offset
        if trigger == MarketTrigger.MINUTES_BEFORE_CLOSE:
            result -= timedelta(minutes=offset_minutes)
        else:
            result += timedelta(minutes=offset_minutes)

        return result

    def calculate_next_run(self, schedule: Schedule) -> datetime | None:
        """Calculate the next run time for a schedule."""
        config = schedule.config
        now = datetime.now(self.tz)

        if config.run_once and schedule.run_count > 0:
            return None  # One-time schedule already ran

        # Find next valid date
        check_date = now
        for _ in range(366):  # Check up to a year
            # Skip if not an enabled day
            if check_date.weekday() not in config.enabled_days:
                check_date = (check_date + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                continue

            # Skip holidays if configured
            if config.skip_holidays and self.is_holiday(check_date):
                check_date = (check_date + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                continue

            # Calculate time for this date
            if config.cron_expression:
                try:
                    next_run = CronParser.get_next_run(
                        config.cron_expression, after=now, tz=self.tz
                    )
                    # Verify it's a valid trading day
                    if next_run.weekday() in config.enabled_days:
                        if not config.skip_holidays or not self.is_holiday(next_run):
                            return next_run
                except ValueError:
                    pass

            elif config.time_of_day:
                run_time = datetime.combine(
                    check_date.date(), config.time_of_day, tzinfo=self.tz
                )
                if run_time > now:
                    return run_time

            elif config.market_trigger:
                run_time = self.get_market_time(
                    config.market_trigger,
                    check_date,
                    config.trigger_offset_minutes,
                )
                if run_time > now:
                    return run_time

            # Move to next day
            check_date = (check_date + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        return None

    def add_schedule(
        self,
        name: str,
        symbols: list[str],
        strategy_name: str = "sma",
        cron: str | None = None,
        time_of_day: str | None = None,
        market_trigger: str | None = None,
        offset_minutes: int = 0,
        run_once: bool = False,
        paper_mode: bool = True,
    ) -> Schedule:
        """Add a new schedule."""
        import uuid

        # Parse time if provided
        parsed_time = None
        if time_of_day:
            # Support formats: "9:30", "09:30", "9:30:00"
            match = re.match(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", time_of_day)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                second = int(match.group(3) or 0)
                parsed_time = time(hour, minute, second)

        # Parse market trigger if provided
        parsed_trigger = None
        if market_trigger:
            try:
                parsed_trigger = MarketTrigger(market_trigger)
            except ValueError:
                # Try common aliases
                aliases = {
                    "open": MarketTrigger.MARKET_OPEN,
                    "close": MarketTrigger.MARKET_CLOSE,
                    "premarket": MarketTrigger.PRE_MARKET,
                    "afterhours": MarketTrigger.AFTER_HOURS,
                }
                parsed_trigger = aliases.get(market_trigger.lower())

        config = ScheduleConfig(
            symbols=symbols,
            strategy_name=strategy_name,
            cron_expression=cron,
            time_of_day=parsed_time,
            market_trigger=parsed_trigger,
            trigger_offset_minutes=offset_minutes,
            run_once=run_once,
            paper_mode=paper_mode,
        )

        schedule = Schedule(
            id=str(uuid.uuid4())[:8],
            name=name,
            config=config,
        )

        # Calculate next run
        schedule.next_run = self.calculate_next_run(schedule)

        # Save to store
        self.store.save_schedule(schedule)
        logger.info(f"Added schedule '{name}' (ID: {schedule.id})")

        return schedule

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule."""
        deleted = self.store.delete_schedule(schedule_id)
        if deleted:
            logger.info(f"Removed schedule {schedule_id}")
        return deleted

    def pause_schedule(self, schedule_id: str) -> bool:
        """Pause a schedule."""
        schedule = self.store.get_schedule(schedule_id)
        if schedule:
            schedule.status = ScheduleStatus.PAUSED
            self.store.save_schedule(schedule)
            logger.info(f"Paused schedule {schedule_id}")
            return True
        return False

    def resume_schedule(self, schedule_id: str) -> bool:
        """Resume a paused schedule."""
        schedule = self.store.get_schedule(schedule_id)
        if schedule and schedule.status == ScheduleStatus.PAUSED:
            schedule.status = ScheduleStatus.ACTIVE
            schedule.next_run = self.calculate_next_run(schedule)
            self.store.save_schedule(schedule)
            logger.info(f"Resumed schedule {schedule_id}")
            return True
        return False

    def list_schedules(self, include_inactive: bool = False) -> list[Schedule]:
        """List all schedules."""
        if include_inactive:
            return self.store.get_all_schedules()
        return self.store.get_all_schedules(ScheduleStatus.ACTIVE)

    async def _execute_schedule(self, schedule: Schedule) -> None:
        """Execute a scheduled trading task."""
        logger.info(f"Executing schedule '{schedule.name}' ({schedule.id})")

        signals_generated = 0
        trades_executed = 0
        error_message = None

        try:
            if not self.broker or not self.data_fetcher:
                raise RuntimeError("Broker and data fetcher required for execution")

            # Import here to avoid circular imports
            from trader.strategies.registry import get_strategy

            strategy = get_strategy(schedule.config.strategy_name)

            # Connect broker if needed
            if not self.broker._client:  # type: ignore
                await self.broker.connect()

            # Process each symbol
            for symbol in schedule.config.symbols:
                try:
                    # Fetch data
                    end = datetime.now()
                    start = end - timedelta(days=100)

                    from trader.core.models import TimeFrame

                    data = await self.data_fetcher.fetch_bars_df(
                        symbol=symbol,
                        timeframe=TimeFrame.DAY,
                        start=start,
                        end=end,
                    )

                    if len(data) < strategy.min_bars_required:
                        logger.debug(f"{symbol}: Not enough data")
                        continue

                    # Calculate indicators and generate signal
                    data_with_indicators = strategy.calculate_indicators(data)
                    position = await self.broker.get_position(symbol)

                    signal = strategy.generate_signal(
                        data=data_with_indicators,
                        symbol=symbol,
                        position=position,
                    )

                    signals_generated += 1
                    logger.info(f"{symbol}: {signal.action.value} - {signal.reason}")

                    # TODO: Execute trade if configured
                    # For now, just log the signal

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # Update schedule state
            schedule.last_run = datetime.now(self.tz)
            schedule.run_count += 1
            schedule.next_run = self.calculate_next_run(schedule)

            if schedule.config.run_once:
                schedule.status = ScheduleStatus.COMPLETED

            self.store.save_schedule(schedule)
            self.store.record_run(
                schedule.id, "success", signals_generated, trades_executed
            )

        except Exception as e:
            error_message = str(e)
            schedule.error_count += 1
            schedule.last_error = error_message
            schedule.last_run = datetime.now(self.tz)

            if schedule.error_count >= schedule.config.max_retries:
                schedule.status = ScheduleStatus.ERROR

            self.store.save_schedule(schedule)
            self.store.record_run(schedule.id, "error", error=error_message)
            logger.error(f"Schedule '{schedule.name}' failed: {e}")

    async def run_daemon(self) -> None:
        """Run the scheduler daemon."""
        logger.info("Starting trading scheduler daemon")
        self._running = True
        self._stop_event.clear()

        try:
            while not self._stop_event.is_set():
                now = datetime.now(self.tz)

                # Get active schedules
                schedules = self.store.get_all_schedules(ScheduleStatus.ACTIVE)

                for schedule in schedules:
                    # Check if due to run
                    if schedule.next_run and now >= schedule.next_run:
                        # Don't run if already running
                        if schedule.id not in self._active_tasks:
                            task = asyncio.create_task(self._execute_schedule(schedule))
                            self._active_tasks[schedule.id] = task

                            # Clean up completed tasks
                            task.add_done_callback(
                                lambda t, sid=schedule.id: self._active_tasks.pop(
                                    sid, None
                                )
                            )

                # Sleep for a minute before checking again
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=60)
                except TimeoutError:
                    pass

        except Exception as e:
            logger.error(f"Scheduler daemon error: {e}")
            raise
        finally:
            self._running = False
            logger.info("Trading scheduler daemon stopped")

    async def stop(self) -> None:
        """Stop the scheduler daemon."""
        logger.info("Stopping scheduler daemon")
        self._stop_event.set()

        # Wait for active tasks to complete
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)


# Global store instance
_schedule_store: ScheduleStore | None = None


def get_schedule_store(db_path: Path | str | None = None) -> ScheduleStore:
    """Get the global schedule store instance."""
    global _schedule_store
    if _schedule_store is None or db_path is not None:
        _schedule_store = ScheduleStore(db_path)
    return _schedule_store
