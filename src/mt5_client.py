# src/mt5_client.py
from __future__ import annotations
import time
from typing import Optional
import MetaTrader5 as mt5  # type: ignore
from loguru import logger


class MT5Client:
    """
    Safe wrapper around MetaTrader5 initialization and login.
    Usage:
      m = MT5Client(login, password, server, path)
      ok = m.connect()
      if ok: ... m.shutdown()
    """

    def __init__(
        self,
        login: Optional[str] | Optional[int],
        password: Optional[str],
        server: Optional[str],
        path: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self._raw_login = login
        self.password = password
        self.server = server
        self.path = path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connected = False
        # attempt safe coercion to int; if fails we'll try login() later (mt5.login may accept int)
        try:
            self.login = int(login) if login is not None and str(login).strip() != "" else None
        except Exception:
            logger.warning("MT5Client: login cannot be converted to int; will try as string at login()")
            self.login = None

    def connect(self) -> bool:
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"MT5Client: initialize() attempt {attempt}/{self.max_retries} (path={self.path})")
                ok = mt5.initialize(path=self.path) if self.path else mt5.initialize()
                if not ok:
                    last_err = mt5.last_error()
                    logger.error(f"MT5 initialize() failed: {last_err}")
                    mt5.shutdown()
                    time.sleep(self.retry_delay)
                    continue

                # If credentials provided, attempt explicit login
                if self.login is not None and self.password and self.server:
                    logger.info("MT5Client: attempting explicit mt5.login()")
                    authorized = mt5.login(self.login, password=self.password, server=self.server)
                    if not authorized:
                        last_err = mt5.last_error()
                        logger.error(f"MT5 login failed: {last_err}")
                        mt5.shutdown()
                        time.sleep(self.retry_delay)
                        continue
                    logger.info("MT5 login OK")
                else:
                    # No creds: assume terminal already logged in; validate by checking account_info()
                    acct = mt5.account_info()
                    if acct is None:
                        last_err = mt5.last_error()
                        logger.error("MT5 terminal not logged in and no credentials were provided.")
                        mt5.shutdown()
                        time.sleep(self.retry_delay)
                        continue
                    logger.info(f"MT5 terminal already logged in (account={acct.login})")

                # verify account_info now
                account_info = mt5.account_info()
                if account_info is None:
                    last_err = mt5.last_error()
                    logger.error("MT5 connected but account_info() returned None.")
                    mt5.shutdown()
                    time.sleep(self.retry_delay)
                    continue

                logger.info(f"MT5 connected successfully (account={account_info.login})")
                self._connected = True
                return True

            except Exception as exc:
                last_err = exc
                logger.exception(f"MT5Client: unexpected error on connect: {exc}")
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                time.sleep(self.retry_delay)

        logger.critical(f"MT5Client: failed to connect after {self.max_retries} attempts. Last error: {last_err}")
        return False

    def is_connected(self) -> bool:
        return bool(self._connected)

    def shutdown(self) -> None:
        try:
            if self._connected:
                logger.info("MT5Client: shutting down connection.")
            else:
                logger.info("MT5Client: shutdown() called but client not connected.")
            mt5.shutdown()
        except Exception as e:
            logger.warning(f"MT5Client: exception during shutdown: {e}")
        finally:
            self._connected = False

    def account_info(self):
        try:
            return mt5.account_info()
        except Exception:
            return None
