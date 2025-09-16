# src/strategy_base.py
from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    """
    Abstract base class for a trading strategy.
    """
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the strategy/model to historical data.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Return the probability of the positive class (e.g., price going up).
        """
        raise NotImplementedError
