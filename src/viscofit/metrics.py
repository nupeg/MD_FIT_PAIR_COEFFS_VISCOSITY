from typing import Protocol

import numpy as np
import numpy.typing as npt

class Metric(Protocol):
    def __call__(self, ytrue: npt.ArrayLike, ycalc: npt.ArrayLike, /) -> float: ...


def mean_squared_error(ytrue: npt.ArrayLike, ycalc: npt.ArrayLike, /) -> float:
    ytrue = np.asarray(ytrue)
    ycalc = np.asarray(ycalc)

    deviations = ycalc - ytrue

    return np.mean(deviations ** 2)

def mean_absolute_error(ytrue: npt.ArrayLike, ycalc: npt.ArrayLike, /) -> float:
    ytrue = np.asarray(ytrue)
    ycalc = np.asarray(ycalc)

    deviations = ycalc - ytrue

    return np.mean(np.abs(deviations))

def mean_absolute_percentage_error(ytrue: npt.ArrayLike, ycalc: npt.ArrayLike, /) -> float:
    ytrue = np.asarray(ytrue)
    ycalc = np.asarray(ycalc)

    deviations = ycalc - ytrue
    deviations = np.abs(deviations / ytrue)

    return 100 * np.mean(deviations)

def r2_score(ytrue: npt.ArrayLike, ycalc: npt.ArrayLike, /) -> float:
    ytrue = np.asarray(ytrue)
    ycalc = np.asarray(ycalc)

    deviations = ycalc - ytrue
    deviations_squared = deviations ** 2

    ytrue_centered = ytrue - np.mean(ytrue)
    ytrue_centered_squared = ytrue_centered ** 2

    return 1 - np.sum(deviations_squared) / np.sum(ytrue_centered_squared)

def max_error(ytrue: npt.ArrayLike, ycalc: npt.ArrayLike, /) -> float:
    ytrue = np.asarray(ytrue)
    ycalc = np.asarray(ycalc)

    deviations = ycalc - ytrue

    return np.max(np.abs(deviations))