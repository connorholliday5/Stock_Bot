# ============================================================
# tests/test_phase1.py
# Validate Phase 1 is wired correctly before moving on
# Run: pytest tests/test_phase1.py -v
# ============================================================

import pytest
from unittest.mock import patch, MagicMock


class TestSettings:

    def test_settings_load(self):
        """Settings module imports without error"""
        from config import settings
        assert settings is not None

    def test_required_fields_exist(self):
        """All required setting fields are defined"""
        from config import settings
        assert hasattr(settings, "alpaca_api_key")
        assert hasattr(settings, "alpaca_secret_key")
        assert hasattr(settings, "polygon_api_key")
        assert hasattr(settings, "starting_capital")
        assert hasattr(settings, "max_stock_positions")
        assert hasattr(settings, "max_crypto_positions")

    def test_risk_params_sane(self):
        """Risk parameters are within safe bounds"""
        from config import settings
        assert 0 < settings.stock_risk_per_trade <= 0.05
        assert 0 < settings.crypto_risk_per_trade <= 0.05
        assert settings.min_position_size >= 10
        assert settings.weekly_drawdown_halt_stock < 0
        assert settings.weekly_drawdown_halt_crypto < 0

    def test_market_timing(self):
        """Market timing strings are defined"""
        from config import settings
        assert settings.stock_buy_time == "09:45"
        assert settings.stock_sell_time == "15:45"
        assert settings.stock_buy_day == "monday"
        assert settings.stock_sell_day == "friday"

    def test_crypto_universe_not_empty(self):
        """Crypto universe has at least one symbol"""
        from config import settings
        assert len(settings.crypto_universe) >= 1

    def test_database_url_format(self):
        """Database URL is properly formatted"""
        from config import settings
        url = settings.database_url
        assert url.startswith("postgresql://")


class TestDatabaseSchema:

    def test_all_models_import(self):
        """All database models import correctly"""
        from database.db import (
            Trade, Signal, Position,
            WeeklyPerformance, ModelPerformance, BotLog
        )
        assert Trade is not None
        assert Signal is not None
        assert Position is not None
        assert WeeklyPerformance is not None
        assert ModelPerformance is not None
        assert BotLog is not None

    def test_enums_defined(self):
        """All enums are defined"""
        from database.db import AssetType, OrderSide, OrderStatus, SignalDirection, MarketRegime
        assert AssetType.STOCK
        assert AssetType.CRYPTO
        assert OrderSide.BUY
        assert OrderSide.SELL
        assert MarketRegime.TRENDING


class TestScheduler:

    def test_scheduler_imports(self):
        """Scheduler imports without error"""
        import scheduler
        assert hasattr(scheduler, "build_scheduler")
        assert hasattr(scheduler, "register_jobs")

    def test_all_jobs_defined(self):
        """All required job functions exist"""
        import scheduler
        assert callable(scheduler.sunday_stock_scan)
        assert callable(scheduler.monday_stock_buys)
        assert callable(scheduler.friday_stock_sells)
        assert callable(scheduler.crypto_cycle)
        assert callable(scheduler.weekly_performance_report)


class TestAlerts:

    def test_alert_manager_imports(self):
        """Alert manager imports without error"""
        from monitoring import alert_manager
        assert alert_manager is not None

    def test_alert_methods_exist(self):
        """All alert methods are defined"""
        from monitoring import alert_manager
        assert callable(alert_manager.send_trade_opened)
        assert callable(alert_manager.send_trade_closed)
        assert callable(alert_manager.send_weekly_summary)
        assert callable(alert_manager.send_drawdown_halt)
        assert callable(alert_manager.send_error)
