from typing import (
    Any,
    Self
)
from os import PathLike
from dataclasses import dataclass

from polars import DataFrame
from scipy.optimize import OptimizeResult


@dataclass(slots=True, frozen=True)
class Artifact:
    best_parameters: dict[str, Any]
    best_score: float
    parameter_names: list[str]
    search_history: DataFrame

    @classmethod
    def from_scipy(cls, data: OptimizeResult, /) -> Self: ...

    @classmethod
    def save(self, folder: PathLike, /) -> None: ...