"""Base model class for all models."""

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd
from matplotlib.pyplot import Axes

from traced_v2.utils import create_hash

# pylint: disable=too-many-arguments, fixme, line-too-long, too-many-instance-attributes, invalid-name

TS_SUBSCRIPTIONS = defaultdict(
    lambda: []
)  # used for sharing timestamps between models in order to save memory


class BaseModel(ABC):
    """Abstract class for all models."""

    def __init__(self, src: str, dest: str, subscription: str | None = None) -> None:
        self.src: str = src
        self.dest: str = dest

        self.n: int = 0
        self.n_anomalies = 1

        self.is_subscribed: bool = subscription is not None
        self.subscription: str | None = subscription or self._generate_id()
        self.timestamps: list[int] = TS_SUBSCRIPTIONS[self.subscription]

    def _generate_id(self) -> str:
        """Generate a unique ID for the model index."""
        return create_hash(f"{self.src}{self.dest}{str(datetime.now())}")

    def get_n(self) -> int:
        """Get the number of observations."""
        return len(self.timestamps)

    def log_timestamp(self, ts: int) -> None:
        """Log a new timestamp."""
        self.n += 1

        if self.is_subscribed:
            return

        if self.timestamps:
            assert ts >= self.timestamps[-1], "Timestamps must be strictly increasing"

        self.timestamps.append(ts)

    @abstractmethod
    def to_dict(self) -> dict[str, list[Any]]:
        """Convert the model statistics to a dictionary."""

    def to_frame(self, omit_first: object = False) -> pd.DataFrame:
        """Convert the model statistics to a pandas DataFrame."""
        df = pd.DataFrame(
            self.to_dict(),
            index=pd.DatetimeIndex(pd.to_datetime(self.timestamps, unit="ms")),
        )
        df.attrs["src"] = self.src
        df.attrs["dest"] = self.dest

        if omit_first:
            df = df.iloc[1:]

        return df


class Visual(ABC):
    """Abstract class for models that can be plotted."""

    # pylint: disable=too-few-public-methods
    @abstractmethod
    def plot(self, ax: Axes | None = None) -> None:
        """Plot the model statistics."""
