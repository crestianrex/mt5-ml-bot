### `src/mt5_client.py`

from __future__ import annotations
import os
from typing import Optional
import MetaTrader5 as mt5
from loguru import logger

class MT5Client:
    def __init__(self, login: int, password: str, server: str, path: Optional[str] = None):
        self.login = int(login)
        self.password = password
        self.server = server
        self.path = path

    def connect(self) -> bool:
        if not mt5.initialize(path=self.path or None):
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False
        authorized = mt5.login(self.login, password=self.password, server=self.server)
        if not authorized:
            logger.error(f"MT5 login failed: {mt5.last_error()}")
        else:
            logger.info("MT5 login OK")
        return authorized

    def shutdown(self):
        mt5.shutdown()
