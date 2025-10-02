# src/notifier.py
import requests
from loguru import logger
from src.config import Cfg

class TelegramNotifier:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.bot_token = cfg.monitoring.telegram_bot_token
        self.chat_id = cfg.monitoring.telegram_chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram Notifier not fully configured (missing bot_token or chat_id). Notifications will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Telegram Notifier initialized.")

    def send_message(self, message: str, level: str = "INFO"):
        if not self.enabled:
            return

        full_message = f"[{level}] {message}"
        payload = {
            "chat_id": self.chat_id,
            "text": full_message,
            "parse_mode": "HTML" # Allows basic formatting like bold, italics
        }

        try:
            response = requests.post(self.base_url, data=payload)
            response.raise_for_status() # Raise an exception for HTTP errors
            logger.debug(f"Telegram message sent: {message}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while sending Telegram message: {e}")
