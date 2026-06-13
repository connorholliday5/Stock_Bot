# ============================================================
# main.py
# Bot entry point — startup checks, watchdog, orchestration
# ============================================================

import sys
import time
from loguru import logger
from config import settings
from database import init_db, health_check
from monitoring import alert_manager


# ============================================================
# Logging Setup
# ============================================================

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    level="INFO"
)
logger.add(
    "logs/bot_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}"
)


# ============================================================
# Startup Checks
# ============================================================

def run_startup_checks() -> bool:
    """Verify all systems are ready before bot starts"""
    logger.info("Running startup checks...")
    checks_passed = True

    # Database
    if not health_check():
        logger.error("STARTUP FAILED: Database connection failed")
        checks_passed = False
    else:
        logger.info("✓ Database connected")

    # Required env vars
    required = [
        ("ALPACA_API_KEY", settings.alpaca_api_key),
        ("ALPACA_SECRET_KEY", settings.alpaca_secret_key),
        ("POLYGON_API_KEY", settings.polygon_api_key),
    ]
    for name, value in required:
        if not value or value.startswith("your_"):
            logger.error(f"STARTUP FAILED: {name} not configured")
            checks_passed = False
        else:
            logger.info(f"✓ {name} loaded")

    # Paper trading confirmation
    if settings.is_paper:
        logger.info("✓ Running in PAPER TRADING mode")
    else:
        logger.warning("⚠️  Running in LIVE TRADING mode — real money at risk")

    return checks_passed


# ============================================================
# Watchdog Loop
# ============================================================

def run_with_watchdog():
    """
    Wraps the scheduler in a watchdog loop.
    If the scheduler crashes, it restarts automatically
    and sends a Telegram alert.
    """
    from scheduler import start_scheduler

    consecutive_failures = 0
    max_failures = 5

    while True:
        try:
            logger.info("Starting scheduler...")
            start_scheduler()
        except KeyboardInterrupt:
            logger.info("Bot stopped manually")
            break
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Scheduler crashed (failure {consecutive_failures}/{max_failures}): {e}")
            alert_manager.send_bot_restarted(str(e))

            if consecutive_failures >= max_failures:
                logger.critical("Too many consecutive failures — halting bot")
                alert_manager.send_error("WATCHDOG", f"Bot halted after {max_failures} consecutive crashes")
                sys.exit(1)

            wait = min(60 * consecutive_failures, 300)
            logger.info(f"Restarting in {wait} seconds...")
            time.sleep(wait)
        else:
            consecutive_failures = 0


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("STOCK BOT STARTING")
    logger.info("=" * 50)

    # Initialize database tables
    init_db()

    # Run all startup checks
    if not run_startup_checks():
        logger.critical("Startup checks failed — bot will not start")
        sys.exit(1)

    logger.info("All startup checks passed")
    alert_manager.send_bot_started()

    # Start with watchdog
    run_with_watchdog()
