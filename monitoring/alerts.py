# ============================================================
# monitoring/alerts.py
# Telegram alert system — all critical bot events sent here
# ============================================================

import asyncio
from telegram import Bot
from telegram.error import TelegramError
from loguru import logger
from config import settings


class AlertManager:

    def __init__(self):
        self.bot = Bot(token=settings.telegram_bot_token)
        self.chat_id = settings.telegram_chat_id
        self.enabled = bool(settings.telegram_bot_token and settings.telegram_chat_id)

    async def _send(self, message: str):
        if not self.enabled:
            logger.warning("Telegram not configured — alert not sent")
            return
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="Markdown"
            )
        except TelegramError as e:
            logger.error(f"Telegram alert failed: {e}")

    def send(self, message: str):
        """Sync wrapper — call from anywhere"""
        try:
            asyncio.get_event_loop().run_until_complete(self._send(message))
        except RuntimeError:
            asyncio.run(self._send(message))

    def send_trade_opened(self, symbol: str, side: str, qty: float, price: float, asset_type: str):
        msg = (
            f"*TRADE OPENED* {'📈' if side == 'buy' else '📉'}\n"
            f"Type: {asset_type.upper()}\n"
            f"Symbol: `{symbol}`\n"
            f"Side: {side.upper()}\n"
            f"Qty: {qty}\n"
            f"Price: ${price:,.4f}"
        )
        self.send(msg)

    def send_trade_closed(self, symbol: str, pnl: float, pnl_pct: float, reason: str):
        emoji = "✅" if pnl >= 0 else "❌"
        msg = (
            f"*TRADE CLOSED* {emoji}\n"
            f"Symbol: `{symbol}`\n"
            f"PnL: ${pnl:,.2f} ({pnl_pct:.2f}%)\n"
            f"Reason: {reason}"
        )
        self.send(msg)

    def send_weekly_summary(self, week: int, net_pnl: float, win_rate: float, total_trades: int):
        emoji = "🟢" if net_pnl >= 0 else "🔴"
        msg = (
            f"*WEEKLY SUMMARY* {emoji} — Week {week}\n"
            f"Net PnL: ${net_pnl:,.2f}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Trades: {total_trades}"
        )
        self.send(msg)

    def send_monday_buys(self, buy_list: list):
        lines = "\n".join([f"  • `{s['symbol']}` — ${s['allocation']:,.2f}" for s in buy_list])
        msg = f"*MONDAY BUY LIST* 🛒\n{lines}"
        self.send(msg)

    def send_drawdown_halt(self, asset_type: str, drawdown: float):
        msg = (
            f"*⛔ DRAWDOWN HALT — {asset_type.upper()}*\n"
            f"Weekly drawdown hit {drawdown:.2f}%\n"
            f"Bot halted. Manual review required."
        )
        self.send(msg)

    def send_error(self, module: str, error: str):
        msg = (
            f"*🚨 ERROR — {module}*\n"
            f"`{error}`"
        )
        self.send(msg)

    def send_bot_started(self):
        env = "PAPER" if settings.is_paper else "LIVE"
        msg = f"*🤖 Bot Started*\nEnvironment: {env}\nCapital: ${settings.starting_capital:,.2f}"
        self.send(msg)

    def send_bot_restarted(self, reason: str):
        msg = f"*🔄 Bot Restarted*\nReason: {reason}"
        self.send(msg)


alert_manager = AlertManager()
