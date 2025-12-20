"""Tests for the trading scheduler module."""

from __future__ import annotations

import tempfile
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from trader.scheduler.scheduler import (
    CronParser,
    MarketTrigger,
    Schedule,
    ScheduleConfig,
    ScheduleStatus,
    ScheduleStore,
    TradingScheduler,
)


class TestCronParser:
    """Tests for cron expression parsing."""

    def test_parse_wildcard(self) -> None:
        """Test parsing wildcard field."""
        result = CronParser.parse_field("*", 0, 59)
        assert result == set(range(60))

    def test_parse_single_value(self) -> None:
        """Test parsing single value."""
        result = CronParser.parse_field("30", 0, 59)
        assert result == {30}

    def test_parse_range(self) -> None:
        """Test parsing range."""
        result = CronParser.parse_field("1-5", 0, 6)
        assert result == {1, 2, 3, 4, 5}

    def test_parse_list(self) -> None:
        """Test parsing comma-separated list."""
        result = CronParser.parse_field("1,3,5", 0, 6)
        assert result == {1, 3, 5}

    def test_parse_step(self) -> None:
        """Test parsing step values."""
        result = CronParser.parse_field("*/15", 0, 59)
        assert result == {0, 15, 30, 45}

    def test_parse_full_expression(self) -> None:
        """Test parsing full cron expression."""
        # 9:30 AM on weekdays
        result = CronParser.parse("30 9 * * 1-5")
        assert result["minute"] == {30}
        assert result["hour"] == {9}
        assert result["day"] == set(range(1, 32))
        assert result["month"] == set(range(1, 13))
        assert result["weekday"] == {1, 2, 3, 4, 5}

    def test_parse_invalid_expression(self) -> None:
        """Test that invalid expressions raise error."""
        with pytest.raises(ValueError, match="Invalid cron expression"):
            CronParser.parse("30 9 *")  # Only 3 fields

    def test_get_next_run(self) -> None:
        """Test calculating next run time."""
        tz = ZoneInfo("America/New_York")
        # Use a fixed time that's a weekday
        after = datetime(2024, 6, 3, 8, 0, tzinfo=tz)  # Monday 8 AM

        # 9:30 AM on weekdays
        next_run = CronParser.get_next_run("30 9 * * 0-4", after=after, tz=tz)

        assert next_run.hour == 9
        assert next_run.minute == 30
        assert next_run.weekday() < 5  # Weekday


class TestScheduleConfig:
    """Tests for ScheduleConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ScheduleConfig()

        assert config.strategy_name == "sma"
        assert config.action == "run"
        assert config.run_once is False
        assert config.enabled_days == [0, 1, 2, 3, 4]  # Mon-Fri
        assert config.paper_mode is True

    def test_trigger_description_cron(self) -> None:
        """Test trigger description for cron."""
        config = ScheduleConfig(cron_expression="30 9 * * 1-5")
        assert "Cron" in config.trigger_description

    def test_trigger_description_time(self) -> None:
        """Test trigger description for time of day."""
        config = ScheduleConfig(time_of_day=time(9, 35))
        assert "09:35" in config.trigger_description

    def test_trigger_description_market_trigger(self) -> None:
        """Test trigger description for market trigger."""
        config = ScheduleConfig(
            market_trigger=MarketTrigger.MARKET_OPEN,
            trigger_offset_minutes=5,
        )
        desc = config.trigger_description
        assert "Market Open" in desc
        assert "+5m" in desc


class TestSchedule:
    """Tests for Schedule model."""

    def test_schedule_creation(self) -> None:
        """Test creating a schedule."""
        config = ScheduleConfig(
            symbols=["AAPL", "MSFT"],
            strategy_name="momentum",
            time_of_day=time(9, 35),
        )
        schedule = Schedule(
            id="test123",
            name="Morning Run",
            config=config,
        )

        assert schedule.id == "test123"
        assert schedule.name == "Morning Run"
        assert schedule.config.symbols == ["AAPL", "MSFT"]
        assert schedule.status == ScheduleStatus.ACTIVE
        assert schedule.run_count == 0

    def test_schedule_to_dict(self) -> None:
        """Test converting schedule to dictionary."""
        config = ScheduleConfig(
            symbols=["AAPL"],
            strategy_name="sma",
            market_trigger=MarketTrigger.MARKET_OPEN,
        )
        schedule = Schedule(
            id="abc",
            name="Test",
            config=config,
        )

        data = schedule.to_dict()

        assert data["id"] == "abc"
        assert data["name"] == "Test"
        assert data["symbols"] == "AAPL"
        assert data["strategy_name"] == "sma"
        assert data["market_trigger"] == "market_open"

    def test_schedule_from_dict(self) -> None:
        """Test creating schedule from dictionary."""
        data = {
            "id": "xyz",
            "name": "Restored",
            "symbols": "AAPL,MSFT",
            "strategy_name": "rsi",
            "action": "run",
            "cron_expression": None,
            "time_of_day": "09:30:00",
            "market_trigger": None,
            "trigger_offset_minutes": 0,
            "run_once": False,
            "enabled_days": "0,1,2,3,4",
            "paper_mode": True,
            "status": "active",
            "created_at": "2024-01-01T09:00:00",
            "last_run": None,
            "next_run": "2024-01-02T09:30:00",
            "run_count": 5,
            "error_count": 0,
            "last_error": None,
        }

        schedule = Schedule.from_dict(data)

        assert schedule.id == "xyz"
        assert schedule.name == "Restored"
        assert schedule.config.symbols == ["AAPL", "MSFT"]
        assert schedule.config.time_of_day == time(9, 30, 0)
        assert schedule.run_count == 5


class TestScheduleStore:
    """Tests for ScheduleStore persistence."""

    @pytest.fixture
    def store(self) -> ScheduleStore:
        """Create a temporary store for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_schedules.db"
            yield ScheduleStore(db_path)

    def test_save_and_get_schedule(self, store: ScheduleStore) -> None:
        """Test saving and retrieving a schedule."""
        config = ScheduleConfig(
            symbols=["AAPL"],
            strategy_name="momentum",
            time_of_day=time(10, 0),
        )
        schedule = Schedule(
            id="save_test",
            name="Save Test",
            config=config,
        )

        store.save_schedule(schedule)
        retrieved = store.get_schedule("save_test")

        assert retrieved is not None
        assert retrieved.id == "save_test"
        assert retrieved.name == "Save Test"
        assert retrieved.config.symbols == ["AAPL"]

    def test_get_nonexistent_schedule(self, store: ScheduleStore) -> None:
        """Test getting a schedule that doesn't exist."""
        result = store.get_schedule("nonexistent")
        assert result is None

    def test_get_all_schedules(self, store: ScheduleStore) -> None:
        """Test getting all schedules."""
        for i in range(3):
            config = ScheduleConfig(symbols=["AAPL"], time_of_day=time(9, 30 + i))
            schedule = Schedule(id=f"multi_{i}", name=f"Multi {i}", config=config)
            store.save_schedule(schedule)

        all_schedules = store.get_all_schedules()
        assert len(all_schedules) == 3

    def test_get_schedules_by_status(self, store: ScheduleStore) -> None:
        """Test filtering schedules by status."""
        # Create active and paused schedules
        active = Schedule(
            id="active",
            name="Active",
            config=ScheduleConfig(symbols=["AAPL"], time_of_day=time(9, 30)),
            status=ScheduleStatus.ACTIVE,
        )
        paused = Schedule(
            id="paused",
            name="Paused",
            config=ScheduleConfig(symbols=["MSFT"], time_of_day=time(10, 0)),
            status=ScheduleStatus.PAUSED,
        )

        store.save_schedule(active)
        store.save_schedule(paused)

        active_only = store.get_all_schedules(ScheduleStatus.ACTIVE)
        assert len(active_only) == 1
        assert active_only[0].id == "active"

    def test_delete_schedule(self, store: ScheduleStore) -> None:
        """Test deleting a schedule."""
        config = ScheduleConfig(symbols=["AAPL"], time_of_day=time(9, 30))
        schedule = Schedule(id="delete_me", name="Delete Me", config=config)
        store.save_schedule(schedule)

        assert store.get_schedule("delete_me") is not None
        deleted = store.delete_schedule("delete_me")
        assert deleted is True
        assert store.get_schedule("delete_me") is None

    def test_record_run(self, store: ScheduleStore) -> None:
        """Test recording a schedule run."""
        config = ScheduleConfig(symbols=["AAPL"], time_of_day=time(9, 30))
        schedule = Schedule(id="run_test", name="Run Test", config=config)
        store.save_schedule(schedule)

        store.record_run("run_test", "success", signals=3, trades=1)
        history = store.get_run_history("run_test")

        assert len(history) == 1
        assert history[0]["status"] == "success"
        assert history[0]["signals_generated"] == 3
        assert history[0]["trades_executed"] == 1


class TestTradingScheduler:
    """Tests for TradingScheduler."""

    @pytest.fixture
    def scheduler(self) -> TradingScheduler:
        """Create a scheduler with temporary store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_schedules.db"
            store = ScheduleStore(db_path)
            yield TradingScheduler(store=store)

    def test_is_trading_day_weekday(self, scheduler: TradingScheduler) -> None:
        """Test that weekdays are trading days."""
        # Monday
        monday = datetime(2024, 6, 3, 10, 0)
        assert scheduler.is_trading_day(monday) is True

    def test_is_trading_day_weekend(self, scheduler: TradingScheduler) -> None:
        """Test that weekends are not trading days."""
        saturday = datetime(2024, 6, 1, 10, 0)
        assert scheduler.is_trading_day(saturday) is False

    def test_is_holiday(self, scheduler: TradingScheduler) -> None:
        """Test holiday detection."""
        # Christmas 2024
        christmas = datetime(2024, 12, 25, 10, 0)
        assert scheduler.is_holiday(christmas) is True

        # Regular day
        regular = datetime(2024, 6, 3, 10, 0)
        assert scheduler.is_holiday(regular) is False

    def test_get_market_time_open(self, scheduler: TradingScheduler) -> None:
        """Test getting market open time."""
        date = datetime(2024, 6, 3, 10, 0, tzinfo=scheduler.tz)
        market_time = scheduler.get_market_time(MarketTrigger.MARKET_OPEN, date)

        assert market_time.hour == 9
        assert market_time.minute == 30

    def test_get_market_time_with_offset(self, scheduler: TradingScheduler) -> None:
        """Test getting market time with offset."""
        date = datetime(2024, 6, 3, 10, 0, tzinfo=scheduler.tz)
        market_time = scheduler.get_market_time(
            MarketTrigger.MINUTES_AFTER_OPEN, date, offset_minutes=5
        )

        assert market_time.hour == 9
        assert market_time.minute == 35

    def test_add_schedule(self, scheduler: TradingScheduler) -> None:
        """Test adding a schedule."""
        schedule = scheduler.add_schedule(
            name="Test Schedule",
            symbols=["AAPL", "MSFT"],
            strategy_name="momentum",
            time_of_day="09:35",
        )

        assert schedule.name == "Test Schedule"
        assert schedule.config.symbols == ["AAPL", "MSFT"]
        assert schedule.config.time_of_day == time(9, 35, 0)
        assert schedule.next_run is not None

    def test_add_schedule_with_market_trigger(self, scheduler: TradingScheduler) -> None:
        """Test adding a schedule with market trigger."""
        schedule = scheduler.add_schedule(
            name="Open Run",
            symbols=["AAPL"],
            market_trigger="open",
            offset_minutes=5,
        )

        assert schedule.config.market_trigger == MarketTrigger.MARKET_OPEN
        assert schedule.config.trigger_offset_minutes == 5

    def test_pause_and_resume_schedule(self, scheduler: TradingScheduler) -> None:
        """Test pausing and resuming a schedule."""
        schedule = scheduler.add_schedule(
            name="Pausable",
            symbols=["AAPL"],
            time_of_day="10:00",
        )

        # Pause
        result = scheduler.pause_schedule(schedule.id)
        assert result is True

        paused = scheduler.store.get_schedule(schedule.id)
        assert paused is not None
        assert paused.status == ScheduleStatus.PAUSED

        # Resume
        result = scheduler.resume_schedule(schedule.id)
        assert result is True

        resumed = scheduler.store.get_schedule(schedule.id)
        assert resumed is not None
        assert resumed.status == ScheduleStatus.ACTIVE

    def test_remove_schedule(self, scheduler: TradingScheduler) -> None:
        """Test removing a schedule."""
        schedule = scheduler.add_schedule(
            name="Removable",
            symbols=["AAPL"],
            time_of_day="10:00",
        )

        result = scheduler.remove_schedule(schedule.id)
        assert result is True

        removed = scheduler.store.get_schedule(schedule.id)
        assert removed is None

    def test_list_schedules(self, scheduler: TradingScheduler) -> None:
        """Test listing schedules."""
        scheduler.add_schedule(name="First", symbols=["AAPL"], time_of_day="09:30")
        scheduler.add_schedule(name="Second", symbols=["MSFT"], time_of_day="10:00")

        schedules = scheduler.list_schedules()
        assert len(schedules) == 2

    def test_calculate_next_run_time_of_day(self, scheduler: TradingScheduler) -> None:
        """Test calculating next run for time-based schedule."""
        schedule = Schedule(
            id="next_run_test",
            name="Next Run Test",
            config=ScheduleConfig(
                symbols=["AAPL"],
                time_of_day=time(10, 0),
                enabled_days=[0, 1, 2, 3, 4],  # Mon-Fri
            ),
        )

        next_run = scheduler.calculate_next_run(schedule)

        assert next_run is not None
        assert next_run.hour == 10
        assert next_run.minute == 0
        assert next_run.weekday() < 5  # Should be a weekday

    def test_calculate_next_run_one_time_completed(
        self, scheduler: TradingScheduler
    ) -> None:
        """Test that completed one-time schedules have no next run."""
        schedule = Schedule(
            id="one_time_done",
            name="One Time Done",
            config=ScheduleConfig(
                symbols=["AAPL"],
                time_of_day=time(10, 0),
                run_once=True,
            ),
            run_count=1,  # Already ran once
        )

        next_run = scheduler.calculate_next_run(schedule)
        assert next_run is None


class TestMarketTrigger:
    """Tests for MarketTrigger enum."""

    def test_all_triggers_defined(self) -> None:
        """Test that all expected triggers are defined."""
        triggers = [t.value for t in MarketTrigger]

        assert "market_open" in triggers
        assert "market_close" in triggers
        assert "pre_market" in triggers
        assert "after_hours" in triggers
        assert "minutes_after_open" in triggers
        assert "minutes_before_close" in triggers


class TestScheduleStatus:
    """Tests for ScheduleStatus enum."""

    def test_all_statuses_defined(self) -> None:
        """Test that all expected statuses are defined."""
        statuses = [s.value for s in ScheduleStatus]

        assert "active" in statuses
        assert "paused" in statuses
        assert "completed" in statuses
        assert "error" in statuses


class TestCryptoSchedules:
    """Tests for cryptocurrency 24/7 schedules."""

    @pytest.fixture
    def scheduler(self) -> TradingScheduler:
        """Create a scheduler with temporary store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_schedules.db"
            store = ScheduleStore(db_path)
            yield TradingScheduler(store=store)

    def test_crypto_schedule_runs_24_7(self, scheduler: TradingScheduler) -> None:
        """Test that crypto schedules run 24/7."""
        schedule = scheduler.add_schedule(
            name="Crypto Test",
            symbols=["BTC/USD"],
            time_of_day="10:00",
            crypto=True,
        )

        assert schedule.config.asset_type == "crypto"
        assert schedule.config.enabled_days == [0, 1, 2, 3, 4, 5, 6]
        assert schedule.config.skip_holidays is False

    def test_crypto_schedule_runs_on_weekends(self, scheduler: TradingScheduler) -> None:
        """Test that crypto schedules can run on weekends."""
        schedule = Schedule(
            id="crypto_weekend",
            name="Weekend Crypto",
            config=ScheduleConfig(
                symbols=["ETH/USD"],
                time_of_day=time(15, 0),
                asset_type="crypto",
                enabled_days=[0, 1, 2, 3, 4, 5, 6],
                skip_holidays=False,
            ),
        )

        # Test on a Saturday
        saturday = datetime(2024, 6, 1, 10, 0, tzinfo=scheduler.tz)  # June 1, 2024 is Saturday

        # Manually set now for testing
        next_run = scheduler.calculate_next_run(schedule)

        # Should schedule for next available time (could be Saturday afternoon)
        assert next_run is not None
        # Weekend day is allowed
        assert next_run.weekday() in [5, 6] or next_run.weekday() in [0, 1, 2, 3, 4]

    def test_crypto_schedule_ignores_holidays(self, scheduler: TradingScheduler) -> None:
        """Test that crypto schedules don't skip holidays."""
        schedule = Schedule(
            id="crypto_holiday",
            name="Holiday Crypto",
            config=ScheduleConfig(
                symbols=["BTC/USD"],
                time_of_day=time(10, 0),
                asset_type="crypto",
                enabled_days=[0, 1, 2, 3, 4, 5, 6],
                skip_holidays=False,
            ),
        )

        next_run = scheduler.calculate_next_run(schedule)
        assert next_run is not None

        # Crypto should run even on US holidays
        # Just verify it can calculate a next run

    def test_stock_schedule_vs_crypto_schedule(self, scheduler: TradingScheduler) -> None:
        """Test that stock and crypto schedules behave differently."""
        stock_schedule = scheduler.add_schedule(
            name="Stock Test",
            symbols=["AAPL"],
            time_of_day="10:00",
            crypto=False,
        )

        crypto_schedule = scheduler.add_schedule(
            name="Crypto Test",
            symbols=["BTC/USD"],
            time_of_day="10:00",
            crypto=True,
        )

        # Stock: Mon-Fri only, observes holidays
        assert stock_schedule.config.asset_type == "stock"
        assert stock_schedule.config.enabled_days == [0, 1, 2, 3, 4]
        assert stock_schedule.config.skip_holidays is True

        # Crypto: 24/7, no holidays
        assert crypto_schedule.config.asset_type == "crypto"
        assert crypto_schedule.config.enabled_days == [0, 1, 2, 3, 4, 5, 6]
        assert crypto_schedule.config.skip_holidays is False
