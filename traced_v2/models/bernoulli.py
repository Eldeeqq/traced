"""Module for the MultinomialModel class."""

from collections import defaultdict
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from traced_v2.models.base_model import BaseModel, Visual

# pylint: disable=too-many-arguments, fixme, line-too-long, too-many-instance-attributes, invalid-name


class BernoulliModel(BaseModel, Visual):
    """Univariate Bernoulli model with bayesian updating."""

    def to_dict(self) -> dict[str, list[Any]]:
        pass

    def plot(self, ax: Figure | Axes | None = None):
        pass

    def __init__(
        self, src: str, dest: str, *args, gamma: float = 1.0, **kwargs
    ) -> None:
        super().__init__(src, dest, *args, **kwargs)


# TODO: Implement Bernoulli model
