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
from viscofit.protocols import SettingsHandler
from viscofit.adapters import ExperimentalData, PairCoeffRange
from viscofit.molecular_dynamics import ( 
    Command,
    PairCoeff,
    BoxDimensions, 
    SimulationFileNames,
    SimulationSetup,
    ViscosityAssets,
    calculate_viscosity_assets,
    create_simulation_setups,
    validate_mixtures_coeffs,
    validate_simulation_setups,
    execute_simulation,
    setup_simulation_files,
    complete_pair_coeffs,
    lorentz_berthelot_sigma,
    lorentz_berthelot_epsilon
)
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

def viscosity_coeffs_objective(
        trial: Trial,
        *, 
        experimental_data: Sequence[ExperimentalData],
        coeffs_ranges: Sequence[PairCoeffRange],
        folder_path: PathLike,
        playmol_cmd: str,
        lammps_cmd: str | list[str],
        box: BoxDimensions,
        files: SimulationFileNames,
        njobs: Optional[int]=None,
        coeffs_values: Sequence[PairCoeff] | None=None,
        optimize_metric: Optional[Metric]=None
    ) -> float:
    
    coeffs: list[PairCoeff] = []

    if coeffs_values is not None:
        coeffs.extend(coeffs_values)

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
            epsilon=epsilon
        )
        coeffs.append(suggested_coeffs)

    # NOTE: Never trust user folder-path
    # Create a safe workspace
    simulation_workspace = Path(folder_path).joinpath('simulation_workspace')

    systems = [datapoint.system for datapoint in experimental_data]

    coeffs_full = complete_pair_coeffs(
        coeffs, sigma_rule=lorentz_berthelot_sigma, epsilon_rule=lorentz_berthelot_epsilon
    )
    validate_mixtures_coeffs(systems, coeffs_full)

    simulations = create_simulation_setups(
        systems, 
        coeffs_full, 
        box, 
        files, 
        simulation_workspace
    )
    validate_simulation_setups(simulations)

    setup_simulations_parallel(simulations, playmol_cmd, njobs)
    execute_simulations_parallel(simulations, lammps_cmd)

    viscosity_assets = calculate_viscosity_parallel(simulations, njobs=njobs)
    
    predicted = [data.viscosity_average for data in viscosity_assets]
    expected = [data.reference_viscosity for data in experimental_data]

    score = optimize_metric(expected, predicted)
    return float(score)

def fit_viscosity_coeffs(
        experimental_data: Sequence[ExperimentalData],
        coeffs_ranges: Sequence[PairCoeffRange],
        folder_path: PathLike,
        playmol_cmd: str,
        lammps_cmd: str | list[str],
        box: BoxDimensions,
        files: SimulationFileNames,
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
    
    objective_function = lambda trial: viscosity_coeffs_objective(
        trial=trial,
        experimental_data=experimental_data,
        coeffs_ranges=coeffs_ranges,
        folder_path=folder_path,
        playmol_cmd=playmol_cmd,
        lammps_cmd=lammps_cmd,
        box=box,
        files=files,
        coeffs_values=coeffs_values,
        optimize_metric=optimize_metric
    )
    fit_study = create_study(
        storage=storage,
        direction=direction,
        study_name=checkpoint_name,
        load_if_exists=True
    )
    fit_study.optimize(func=objective_function, n_trials=n_trials, show_progress_bar=show_progress_bar)
    ...
    
    return

def setup_simulations_parallel(simulations: Sequence[SimulationSetup], playmol_cmd: Command, /, njobs: Optional[int]=None) -> None:
    total = len(simulations)

    process = lambda setup: setup_simulation_files(setup, playmol_cmd=playmol_cmd, new_terminal=False)

    with open_monitor(total=total, title='SETUP', description='Creating simulation files with Playmol') as ui:
        for _ in run_parallel(simulations, process=process, njobs=njobs):
            ui.communicate()

def execute_simulations_parallel(simulations: Sequence[SimulationSetup], lammps_cmd: Command | Sequence[Command], /) -> None: 
    filepaths = [
        simulation.simulation_filepath for simulation in simulations
    ]
    total = len(filepaths)

    if isinstance(lammps_cmd, (str, Path, PathLike)):
        lammps_cmd = [lammps_cmd]

    processes = [
        lambda path: execute_simulation(path, lammps_cmd=cmd, new_terminal=True) for cmd in lammps_cmd
    ]

    with open_monitor(total=total, title='SIMULATION', description='Running simulations with LAMMPS') as ui:
        for _ in run_work_pool(filepaths, processes):
            ui.communicate()

def calculate_viscosity_parallel(simulations: Sequence[SimulationSetup], /, njobs: Optional[int]=None) -> list[ViscosityAssets]:
    simulation_folders = [
        simulation.folder_path for simulation in simulations
    ]
    total = len(simulation_folders)
    
    calculations = []
    
    # NOTE: Always maintain order to return values processes 
    with open_monitor(total=total, title='CALCULATING', description='Calculating viscosity with Green-Kubo approach') as ui:
        for calculated_viscosity in run_parallel(simulation_folders, process=calculate_viscosity_assets, njobs=njobs, maintain_order=True):
            calculations.append(calculated_viscosity)
            ui.communicate()
        
    return calculations

def workspace_folder(base_folder: PathLike, name: str, /) -> Path:
    return Path(base_folder).joinpath(name)

def evaluate_predictions(ytrue: npt.ArrayLike, ycalc: npt.ArrayLike, /, dataframe: Optional[PolarsLike]=None) -> tuple[ScoresDataframe, PredictionsDataframe]:
    ytrue  = np.asarray(ytrue)
    ycalc  = np.asarray(ycalc)
    
    deviations = calculate_deviations(ytrue, ycalc)
    deviations_absolute = calculate_deviations(ytrue, ycalc, absolute=True)

    relative_deviations = calculate_deviations(ytrue, ycalc, relative=True, percentage=True)
    relative_deviations_absolute = calculate_deviations(ytrue, ycalc, relative=True, percentage=True, absolute=True)

    if dataframe is None:
        dataframe = {}

    dataframe = add_calculated_columns(dataframe, {
        'Y_TRUE': ytrue,
        'Y_CALC': ycalc,
        'DIFF': deviations,
        'DIFF_ABSOLUTE': deviations_absolute,
        'DIFF_RELATIVE_%': relative_deviations,
        'DIFF_ABSOLUTE_RELATIVE_%': relative_deviations_absolute
    })

    score_r2        = r2_score(ytrue, ycalc)
    score_mse       = mean_squared_error(ytrue, ycalc)
    score_mae       = mean_absolute_error(ytrue, ycalc)
    score_mape      = mean_absolute_percentage_error(ytrue, ycalc)
    score_max_error = max_error(ytrue, ycalc)

    scores = transform_dataframe({
        'R2': score_r2,
        'MEAN_SQUARED_ERROR': score_mse,
        'MEAN_ABSOLUTE_ERROR': score_mae,
        'MEAN_ABSOLUTE_PERCENTAGE_ERROR': score_mape,
        'MAX_ERROR': score_max_error
    })
    return scores, dataframe
