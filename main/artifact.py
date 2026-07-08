from typing import (
    Any,
    Self,
    Union,
    Iterable,
    Mapping, 
    Optional, 
    TypeAlias,
    Protocol,
)
from os import PathLike
from pathlib import Path    
from enum import StrEnum
from shutil import copy2
from dataclasses import dataclass

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Dimension
from skopt.callbacks import CheckpointSaver

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