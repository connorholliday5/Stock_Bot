# ============================================================
# config/settings.py
# Central configuration — loads from .env
# All other modules import from here, never from .env directly
#
# Production notes:
#   - Only the Alpaca keys are required. Polygon / Binance / Telegram /
#     Postgres are optional; features that need them degrade gracefully.
#   - Database: set DATABASE_URL directly, or the DB_* fields for Postgres.
#     With neither configured the bot falls back to a local SQLite file
#     (data_dir/stockbot.db) so a single container "just runs".
#   - Capital: capital_source="broker" (default) reads equity/cash from the
#     live Alpaca account, so whatever money is in the account is the
#     starting amount. "static" uses STARTING_CAPITAL instead.
# ============================================================

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # --- Alpaca (required) ---
    alpaca_api_key: str = Field("", validation_alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field("", validation_alias="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field("https://paper-api.alpaca.markets", validation_alias="ALPACA_BASE_URL")
    alpaca_paper: bool = Field(True, validation_alias="ALPACA_PAPER")

    # --- Polygon (optional; stock universe data) ---
    polygon_api_key: str = Field("", validation_alias="POLYGON_API_KEY")

    # --- Binance (optional; only if crypto_exchange="binance") ---
    binance_api_key: str = Field("", validation_alias="BINANCE_API_KEY")
    binance_secret_key: str = Field("", validation_alias="BINANCE_SECRET_KEY")

    # --- Database (optional; SQLite fallback) ---
    database_url_override: str = Field("", validation_alias="DATABASE_URL")
    db_host: str = Field("localhost", validation_alias="DB_HOST")
    db_port: int = Field(5432, validation_alias="DB_PORT")
    db_name: str = Field("stockbot", validation_alias="DB_NAME")
    db_user: str = Field("stockbot_user", validation_alias="DB_USER")
    db_password: str = Field("", validation_alias="DB_PASSWORD")
    data_dir: str = Field("data_store", validation_alias="DATA_DIR")

    # --- Telegram (optional) ---
    telegram_bot_token: str = Field("", validation_alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field("", validation_alias="TELEGRAM_CHAT_ID")

    # --- Web UI ---
    web_host: str = Field("0.0.0.0", validation_alias="WEB_HOST")
    web_port: int = Field(8000, validation_alias="WEB_PORT")
    # When set, mutating endpoints (halt/resume/run-job) require
    # "Authorization: Bearer <token>". Read endpoints stay open (LAN use).
    web_auth_token: str = Field("", validation_alias="WEB_AUTH_TOKEN")

    # --- Bot Behavior ---
    environment: str = Field("paper", validation_alias="ENVIRONMENT")
    # "broker": live Alpaca equity/cash is the capital base (recommended).
    # "static": STARTING_CAPITAL + DB-reconstructed P&L (offline/backtest).
    capital_source: str = Field("broker", validation_alias="CAPITAL_SOURCE")
    starting_capital: float = Field(1000.0, validation_alias="STARTING_CAPITAL")
    max_stock_positions: int = Field(8, validation_alias="MAX_STOCK_POSITIONS")
    max_crypto_positions: int = Field(3, validation_alias="MAX_CRYPTO_POSITIONS")
    stock_risk_per_trade: float = Field(0.02, validation_alias="STOCK_RISK_PER_TRADE")
    crypto_risk_per_trade: float = Field(0.015, validation_alias="CRYPTO_RISK_PER_TRADE")
    weekly_drawdown_halt_stock: float = Field(-0.08, validation_alias="WEEKLY_DRAWDOWN_HALT_STOCK")
    weekly_drawdown_halt_crypto: float = Field(-0.10, validation_alias="WEEKLY_DRAWDOWN_HALT_CRYPTO")
    min_position_size: float = Field(50.0, validation_alias="MIN_POSITION_SIZE")

    # --- Market Timing (EST) ---
    stock_buy_time: str = "09:45"
    stock_sell_time: str = "15:45"
    stock_scan_time: str = "20:00"
    stock_buy_day: str = "monday"
    stock_sell_day: str = "friday"

    # --- Crypto Universe ---
    # "alpaca" trades BTC/USD-style pairs 24/7 with the money already in the
    # Alpaca account; "binance" keeps the ccxt path (needs Binance keys).
    crypto_exchange: str = Field("alpaca", validation_alias="CRYPTO_EXCHANGE")
    # USDT-quoted (Binance style); the Alpaca path maps these to /USD pairs.
    # All four trade on both venues (BNB deliberately absent - not on Alpaca).
    crypto_universe: list = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LTC/USDT"]
    # Locked strategy: entries stay BTC-only until 60 profitable days, then
    # flip this to false to trade the whole scanned universe.
    crypto_btc_only: bool = Field(True, validation_alias="CRYPTO_BTC_ONLY")

    # --- Stock Universe ---
    # Optional comma-separated ticker override (e.g. "AAPL,MSFT,NVDA").
    # Blank = full built-in S&P 500 list (data/sp500.py) on the Alpaca path.
    stock_universe: str = Field("", validation_alias="STOCK_UNIVERSE")

    # --- Stock Strategy Params ---
    stock_stop_loss: float = 0.05
    stock_take_profit: float = 0.10
    rsi_period: int = 14
    rsi_overbought: float = 65.0
    sma_fast: int = 50
    sma_slow: int = 200
    min_adx: float = 20.0
    max_short_interest: float = 0.15

    # --- Crypto Strategy Params ---
    crypto_stop_loss: float = 0.03
    crypto_ema_fast: int = 9
    crypto_ema_slow: int = 21
    crypto_timeframe: str = "4h"

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        if self.db_password:
            return (
                f"postgresql://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        # SQLite fallback: zero-config single-node deployment.
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path / 'stockbot.db'}"

    @property
    def is_paper(self) -> bool:
        return self.environment == "paper" or self.alpaca_paper

    @property
    def execution_mode(self) -> str:
        """Three-tier ladder:
        sim   - ENVIRONMENT=paper: simulate fills locally, no orders sent.
        paper - ENVIRONMENT=live + ALPACA_PAPER=true: real orders to the
                Alpaca *paper* API (full rehearsal of the order path).
        live  - ENVIRONMENT=live + ALPACA_PAPER=false: real money.
        """
        if self.environment != "live":
            return "sim"
        return "paper" if self.alpaca_paper else "live"

    @property
    def use_broker_capital(self) -> bool:
        return self.capital_source == "broker"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
