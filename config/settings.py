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
    # Reject a stock entry whose trailing returns track an already-held name
    # more closely than this. Momentum rankings cluster by sector, so without
    # it "8 positions" can be 8 versions of one bet. 1.0 disables the guard.
    max_position_correlation: float = Field(
        0.85, validation_alias="MAX_POSITION_CORRELATION")
    # Cap on a single stock position as a fraction of equity. 0 = auto
    # (1 / MAX_STOCK_POSITIONS), which is what makes MAX_STOCK_POSITIONS
    # achievable at all: risk-based sizing computes
    #   notional = equity * risk_per_trade / stop_pct
    # so 2% risk with a 5% stop is a 40% position - cash runs out after ~2.5
    # of them and the configured 8 never happens.
    stock_max_position_pct: float = Field(
        0.0, validation_alias="STOCK_MAX_POSITION_PCT")

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
    # All trade on both venues (BNB deliberately absent - not on Alpaca).
    # NOTE: entries stay BTC-only while CRYPTO_BTC_ONLY=true; the rest are
    # scanned every cycle and become tradable the moment you flip the flag.
    crypto_universe: list = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "LTC/USDT",
        "DOGE/USDT", "LINK/USDT", "AVAX/USDT",
    ]
    # Locked strategy: entries stay BTC-only until 60 profitable days, then
    # flip this to false to trade the whole scanned universe.
    crypto_btc_only: bool = Field(True, validation_alias="CRYPTO_BTC_ONLY")

    # --- Stock Universe ---
    # Optional comma-separated ticker override (e.g. "AAPL,MSFT,NVDA").
    # Blank = full built-in S&P 500 list (data/sp500.py) on the Alpaca path.
    stock_universe: str = Field("", validation_alias="STOCK_UNIVERSE")

    # --- Strategy modes ---
    # rotate    - Friday sells only positions that fell out of the fresh
    #             top rankings; winners keep riding (holds over weekends).
    # liquidate - legacy: sell everything Friday 3:45, flat all weekend.
    # Which stock strategy trades live.
    #   momentum - 12-1 cross-sectional momentum, monthly rebalance, no
    #              stops. Backtested +202% vs SPY +67% over 5y (survivorship
    #              caveat: MTUM, the real ETF, did 10.3% CAGR vs this 24.9%,
    #              so expect low-to-mid teens live, not the headline).
    #   rotation - the legacy weekly rotation. FALSIFIED: -1.6% over 3.3y
    #              against SPY +79%. Kept only for comparison.
    stock_strategy: str = Field("momentum", validation_alias="STOCK_STRATEGY")
    # Concentration dial. Fewer names = more return AND more drawdown; the
    # backtest was smooth and monotonic across 5/10/20:
    #   top 5  -> +325%, max drawdown -52%
    #   top 10 -> +183%, max drawdown -41%
    #   top 20 -> +131%, max drawdown -27%
    momentum_top_n: int = Field(10, validation_alias="MOMENTUM_TOP_N")
    momentum_lookback: int = Field(252, validation_alias="MOMENTUM_LOOKBACK")
    momentum_skip: int = Field(21, validation_alias="MOMENTUM_SKIP")

    stock_exit_mode: str = Field("rotate", validation_alias="STOCK_EXIT_MODE")
    # A held position survives Friday rotation while it ranks inside this
    # many names of the fresh scoring (top_n buys, keep_rank holds).
    rotation_keep_rank: int = Field(20, validation_alias="ROTATION_KEEP_RANK")
    # regime - long while trend is intact (EMA9>EMA21, price>EMA200, ADX
    #          trending); enters mid-trend. cross - legacy: only enters on a
    #          fresh EMA9/21 cross within 3 bars (misses running trends).
    crypto_entry_mode: str = Field("regime", validation_alias="CRYPTO_ENTRY_MODE")
    # Calendar-day lookback the Sunday ML retrain pulls (the weekly scan uses
    # a shorter window). More history = more training examples; ~3 years is a
    # good default for daily-bar cross-sectional models.
    ml_lookback_days: int = Field(1095, validation_alias="ML_LOOKBACK_DAYS")
    # Fraction of equity reserved for the crypto book so Monday stock buys
    # can never starve BTC of cash. 0 disables the reservation (shared pot).
    crypto_allocation_pct: float = Field(0.25, validation_alias="CRYPTO_ALLOCATION_PCT")

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
