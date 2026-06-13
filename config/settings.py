# ============================================================
# config/settings.py
# Central configuration — loads from .env
# All other modules import from here, never from .env directly
# ============================================================

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):

    # --- Alpaca ---
    alpaca_api_key: str = Field(..., env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field("https://paper-api.alpaca.markets", env="ALPACA_BASE_URL")
    alpaca_paper: bool = Field(True, env="ALPACA_PAPER")

    # --- Polygon ---
    polygon_api_key: str = Field(..., env="POLYGON_API_KEY")

    # --- Binance ---
    binance_api_key: str = Field(..., env="BINANCE_API_KEY")
    binance_secret_key: str = Field(..., env="BINANCE_SECRET_KEY")

    # --- Database ---
    db_host: str = Field("localhost", env="DB_HOST")
    db_port: int = Field(5432, env="DB_PORT")
    db_name: str = Field("stockbot", env="DB_NAME")
    db_user: str = Field("stockbot_user", env="DB_USER")
    db_password: str = Field(..., env="DB_PASSWORD")

    # --- Telegram ---
    telegram_bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(..., env="TELEGRAM_CHAT_ID")

    # --- Bot Behavior ---
    environment: str = Field("paper", env="ENVIRONMENT")
    starting_capital: float = Field(1000.0, env="STARTING_CAPITAL")
    max_stock_positions: int = Field(8, env="MAX_STOCK_POSITIONS")
    max_crypto_positions: int = Field(3, env="MAX_CRYPTO_POSITIONS")
    stock_risk_per_trade: float = Field(0.02, env="STOCK_RISK_PER_TRADE")
    crypto_risk_per_trade: float = Field(0.015, env="CRYPTO_RISK_PER_TRADE")
    weekly_drawdown_halt_stock: float = Field(-0.08, env="WEEKLY_DRAWDOWN_HALT_STOCK")
    weekly_drawdown_halt_crypto: float = Field(-0.10, env="WEEKLY_DRAWDOWN_HALT_CRYPTO")
    min_position_size: float = Field(50.0, env="MIN_POSITION_SIZE")

    # --- Market Timing (EST) ---
    stock_buy_time: str = "09:45"
    stock_sell_time: str = "15:45"
    stock_scan_time: str = "20:00"
    stock_buy_day: str = "monday"
    stock_sell_day: str = "friday"

    # --- Crypto Universe ---
    crypto_universe: list = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

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
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def is_paper(self) -> bool:
        return self.environment == "paper" or self.alpaca_paper

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
