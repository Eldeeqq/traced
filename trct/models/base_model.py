"""Base class for all models."""

from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


class BaseModel:
    """Base class for all models."""

    def __init__(self, src: str, dest: str, *args, **kwargs) -> None:
        """Initialize a new model.

        Args:
            src (str): source identificator
            dest (str): destination identificator

        """
        self.u: str = src
        self.v: str = dest

        self.tss: list[int] = [0]  # timestamps
        self.ns: list[int] = [0]  # number of observations

    def log(self, ts) -> Any:
        """Logs timestamps and increments counter.

        Args:
            ts (int): timestamp

        Raises:
            ValueError: if timestamps are not increasing
        """

        if not self.ts <= ts:
            raise ValueError(f"Timestamps must be increasing. {self.ts} !< {ts}.")

        self.n += 1
        self.ts = ts

    def to_frame(self, omit_first=True) -> pd.DataFrame:
        """Convert the model statistics to a pandas DataFrame."""
        df = pd.DataFrame(
            self.get_data(),
            index=pd.DatetimeIndex(pd.to_datetime(self.tss, unit="ms")),
        )

        df["u"] = self.u
        df["v"] = self.v
        df["total"] = self.ns

        if omit_first:
            df = df.iloc[1:]

        return df

    @abstractmethod
    def get_data(self) -> dict[str, list[Any]]:
        """Return the model data."""

    @abstractmethod
    def plot(self, axes: plt.Axes, *args, **kwargs) -> None:
        """Plot the model on specified axis."""

    @abstractmethod
    def score(self, *args, **kwargs) -> Any:
        """Score the traceroute data."""

    @property
    def ts(self) -> int:
        """Return the last (current) timestamps."""
        return self.tss[-1]

    @ts.setter
    def ts(self, value: int) -> None:
        """Adds value to the list of timestamps."""
        self.tss.append(value)

    @property
    def n(self) -> int:  # type: ignore
        """Return the last (current) number of observations."""
        return self.ns[-1]

    @n.setter
    def n(self, value: int) -> None:
        """Adds value to the last number of observations."""
        self.ns.append(value)
