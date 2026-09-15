from typing import (
    Sequence, 
    Optional, 
    Iterable, 
    NamedTuple, 
    TypeAlias
)

from os import PathLike
from pathlib import Path
from functools import partial
from dataclasses import dataclass

from optuna import Trial, create_study

from viscofit.utils import memorable_name
from viscofit.metrics import Metric, mean_squared_error
from viscofit.molecular_dynamics import (
    PairCoeff, 
    MixtureData,
    run_viscosity_simulation_pipeline,
    calculate_viscosity_parallel
)

class RealDimension(NamedTuple):
    low: float
    high: float
    
@dataclass(slots=True, frozen=True)
class PairCoeffRange:
    atom_type_01: str
    atom_type_02: str
    sigma_dimension: RealDimension
    epsilon_dimension: RealDimension


@dataclass(slots=True, frozen=True)
class OptimizationResult:
    ...

@dataclass
class ExperimentCase:
    system: MixtureData
    reference_viscosity: float

LiteralDatabasePath:   TypeAlias = str

def resolve_optimization_storage(checkpoint_folder_path: Optional[PathLike]=None, checkpoint_name: Optional[str]=None, /) -> LiteralDatabasePath | None:
    if checkpoint_folder_path is None:
        return None
    
    if checkpoint_name is None:
        checkpoint_name = memorable_name(size=2)
    
    checkpoint_folder_path = Path(checkpoint_folder_path)

    if not checkpoint_folder_path.is_absolute():
        raise ValueError(
            f'checkpoint_folder_path must be an absolute path.'
        )
    
    if not checkpoint_folder_path.is_dir():
        raise NotADirectoryError(
            f'checkpoint_folder_path does not exist or is not a directory: {checkpoint_folder_path!s}.'
        )
    
    checkpoint_filepath = checkpoint_folder_path.joinpath(checkpoint_name)
    checkpoint_filepath = checkpoint_filepath.with_suffix('.db')

    return f'sqlite:///{checkpoint_filepath}'

def evaluate_viscosity_coeffs(
        trial: Trial,
        *, 
        experimental_data: Sequence[ExperimentCase],
        coeffs_ranges: Sequence[PairCoeffRange],
        folder_path: PathLike,
        playmol_cmd: str,
        lammps_cmd: str | list[str],
        njobs: Optional[int]=None,
        coeffs_values: Sequence[PairCoeff] | None=None,
        optimize_metric: Optional[Metric]=None
    ) -> float:
    
    trial_coeffs: list[PairCoeff] = []

    if coeffs_values is not None:
        trial_coeffs.extend(coeffs_values)

    for coeffs_range in coeffs_ranges:
        atom_type_01 = coeffs_range.atom_type_01
        atom_type_02 = coeffs_range.atom_type_02
        sigma_dimension = coeffs_range.sigma_dimension
        epsilon_dimension = coeffs_range.epsilon_dimension
        
        # NOTE: Get this labels in output
        sigma_label = f'sigma_{atom_type_01}_{atom_type_02}'
        epsilon_label = f'epsilon_{atom_type_01}_{atom_type_02}'
        
        sigma = trial.suggest_float(name=sigma_label, low=sigma_dimension.low, high=sigma_dimension.high)
        epsilon = trial.suggest_float(name=epsilon_label, low=epsilon_dimension.low, high=epsilon_dimension.high)
        
        suggested_coeffs = PairCoeff(
            atom_type_01=atom_type_01,
            atom_type_02=atom_type_02,
            sigma=sigma,
            epsilon=epsilon)
        
        trial_coeffs.append(suggested_coeffs)

    systems = [datapoint.system for datapoint in experimental_data]

    run_viscosity_simulation_pipeline(
        systems, 
        trial_coeffs,
        folder_path,
        playmol_cmd,
        lammps_cmd,
        njobs) 

    simulation_folder_paths = [
        ...
    ]
    predicted = calculate_viscosity_parallel(simulation_folder_paths, njobs=njobs)
    expected = [datapoint.reference_viscosity for datapoint in experimental_data]

    score = optimize_metric(expected, predicted)
    return float(score)

def fit_viscosity_coeffs(
        experimental_data: Sequence[ExperimentCase],
        coeffs_ranges: Sequence[PairCoeffRange],
        coeffs_values: Optional[Sequence[PairCoeff] ]=None,
        n_trials: Optional[int]=None,
        optimize_metric: Optional[Metric]=None,
        checkpoint_name: Optional[str]=None,
        checkpoint_folder_path: Optional[PathLike]=None,
        greater_is_better: bool=True,
        show_progress_bar: bool=True
    ) -> OptimizationResult:
    
    if n_trials is None:
        n_trials = 100
    
    if optimize_metric is None:
        optimize_metric = mean_squared_error
        greater_is_better = False

    direction = 'maximize' if greater_is_better else 'minimize'
    
    storage = resolve_optimization_storage(checkpoint_folder_path, checkpoint_name)
    
    objective_function = lambda trial: evaluate_viscosity_coeffs(
        trial=trial,
        experimental_data=experimental_data,
        coeffs_ranges=coeffs_ranges,
        coeffs_values=coeffs_values,
        optimize_metric=optimize_metric)
    
    fit_study = create_study(
        storage=storage,
        direction=direction,
        study_name=checkpoint_name,
        load_if_exists=True
    )
    fit_study.optimize(func=objective_function, n_trials=n_trials, show_progress_bar=show_progress_bar)

















