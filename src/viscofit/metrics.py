from typing import Protocol

import numpy as np
import numpy.typing as npt

class Metric(Protocol):
    def __call__(self, ytrue: npt.ArrayLike, ypred: npt.ArrayLike, /) -> float: ...


def mean_squared_error(ytrue, ypred) -> float:
    ...