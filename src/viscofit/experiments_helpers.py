from typing import Iterator, TypeAlias, Optional, Sequence, ClassVar
from os import PathLike
from pathlib import Path
from dataclasses import dataclass   

import polars as pl

import numpy as np
import numpy.typing as npt

from optuna import Trial, create_study

from viscofit.utils import (
    dict_from_lists, 
    memorable_name, 
    run_parallel, 
    run_work_pool, 
    calculate_deviations
)
from viscofit.tables import PolarsLike, transform_dataframe, add_calculated_columns
from viscofit.metrics import (
    Metric, 
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
    max_error,
    r2_score
)
from viscofit.molecular_dynamics import (
    Simulation,
    setup_simulation
)
from viscofit.protocols import SettingsHandler
from viscofit.monitor import open_monitor

AVOGADRO:               float = 6.02214076e23
WATER_MOLAR_MASS_KG:    float = 0.01801528

@dataclass(slots=True, frozen=True)
class Experiment:
    func: SettingsHandler
    name: str
    description: str

class ExperimentRegister:

    experiments: ClassVar[list[Experiment] ] = []

    @classmethod
    def iter_experiments(cls) -> Iterator[Experiment]: 
        return iter(cls.experiments)

    @classmethod
    def names(cls) -> Iterator[str]:
        for experiment in cls.iter_experiments():
            yield str(experiment.name)

    @classmethod
    def descriptions(cls) -> Iterator[str]:
        for experiment in cls.iter_experiments():
            yield str(experiment.description)

    @classmethod
    def catalog(cls) -> dict[str, str]:
        return dict_from_lists(cls.names(), cls.descriptions())

    @classmethod
    def take(cls, name: str, /) -> SettingsHandler: 
        for experiment in cls.iter_experiments():
            if experiment.name != name:
                continue
            return experiment.func
        names = sorted(cls.names())
        raise ValueError(
            f'Invalid process name {name!r}. Available options are: {names!r}'
        )

@dataclass(slots=True, frozen=True)
class OptimizationResult:
    ...

LiteralDatabasePath:   TypeAlias = str
ScoresDataframe:       TypeAlias = pl.DataFrame
PredictionsDataframe:  TypeAlias = pl.DataFrame


def setup_simulations_parallel(simulations: Sequence[Simulation], /, njobs: Optional[int]=None, maintain_order: bool=False) -> None:
    with open_monitor(total=len(simulations), title='PLAYMOL', description='Prepare simulation inputs with PlayMol') as ui:
        for _ in run_parallel(simulations, process=setup_simulation, njobs=njobs, maintain_order=maintain_order):
            ui.communicate()