# ============================================================
# main.py
# Production entry point: one process runs everything.
#   - APScheduler (all trading jobs)     - background threads
#   - FastAPI web UI + JSON API          - uvicorn, this thread
# Ctrl-C / SIGTERM stops uvicorn, then the scheduler shuts down.
# ============================================================

import sys

from loguru import logger

from config import settings
from database import init_db, health_check


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
    )
    logger.add(
        "logs/bot_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}",
    )


def run_startup_checks() -> bool:
    """Verify required systems before starting. Alpaca keys are the only hard
    requirement; optional integrations log their status and degrade."""
    logger.info("Running startup checks...")
    ok = True

    if health_check():
        logger.info("✓ Database connected ({})",
                    settings.database_url.split("@")[-1])
    else:
        logger.error("STARTUP FAILED: database connection failed")
        ok = False

    for name, value in (("ALPACA_API_KEY", settings.alpaca_api_key),
                        ("ALPACA_SECRET_KEY", settings.alpaca_secret_key)):
        if not value or value.startswith("your_"):
            logger.error("STARTUP FAILED: {} not configured", name)
            ok = False
        else:
            logger.info("✓ {} loaded", name)

    logger.info("✓ Capital source: {}", settings.capital_source)
    logger.info("✓ Crypto exchange: {}", settings.crypto_exchange)
    if settings.polygon_api_key and not settings.polygon_api_key.startswith("your_"):
        logger.info("✓ Polygon configured (full S&P 500 scan)")
    else:
        logger.info("- Polygon not configured; stock data via Alpaca IEX seed universe")
    if settings.telegram_bot_token:
        logger.info("✓ Telegram alerts configured")
    else:
        logger.info("- Telegram not configured; alerts go to logs only")

    mode = settings.execution_mode
    if mode == "live":
        logger.warning("⚠️  EXECUTION MODE: LIVE — real money at risk")
    elif mode == "paper":
        logger.info("✓ EXECUTION MODE: paper (real orders to Alpaca's paper API)")
    else:
        logger.info("✓ EXECUTION MODE: sim (fills simulated locally, no orders sent)")

    return ok


def main() -> int:
    setup_logging()
    logger.info("=" * 50)
    logger.info("STOCK BOT STARTING")
    logger.info("=" * 50)

    init_db()
    if not run_startup_checks():
        logger.critical("Startup checks failed — bot will not start")
        return 1

    try:
        from monitoring import alert_manager
        alert_manager.send_bot_started()
    except Exception as exc:
        logger.warning("startup alert failed: {}", exc)

    from scheduler import build_scheduler
    scheduler = build_scheduler()
    scheduler.start()
    logger.info("Scheduler started with {} jobs", len(scheduler.get_jobs()))

    import uvicorn
    from webapp import create_app
    app = create_app(scheduler)
    logger.info("Web UI: http://{}:{}", settings.web_host, settings.web_port)
    if not settings.web_auth_token:
        logger.warning("WEB_AUTH_TOKEN not set — control endpoints are unauthenticated. "
                       "Fine on localhost; set a token before exposing the port.")

    try:
        uvicorn.run(app, host=settings.web_host, port=settings.web_port,
                    log_level="warning")
    finally:
        logger.info("Shutting down scheduler...")
        scheduler.shutdown(wait=False)
        logger.info("Bot stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
