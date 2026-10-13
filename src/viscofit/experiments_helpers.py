from typing import Iterator, Mapping, Optional, Sequence, ClassVar
from os import PathLike
from pathlib import Path
from dataclasses import dataclass   

import polars as pl

import numpy as np
import numpy.typing as npt

from optuna import Trial, create_study

from viscofit.utils import (
    coalesce,
    dict_from_lists,  
    run_parallel, 
    run_work_pool, 
    calculate_deviations
)
from viscofit.tables import (
    PolarsLike, 
    ColumnName,
    transform_dataframe, 
    add_calculated_columns,
    write_excel_sheets
)
from viscofit.metrics import ( 
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
    max_error,
    r2_score
)
from viscofit.molecular_dynamics import (
    Simulation,
    SimulationAssets,
    setup_simulation,
    run_simulation,
    calculate_simulation_assets
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


def export_regression_results_workbook(
        excel_filepath: PathLike, 
        ytrue: npt.ArrayLike, 
        ycalc: npt.ArrayLike, 
        metadata: Optional[PolarsLike]=None,
        context: Optional[Mapping[str, str]]=None,
        unit: Optional[str]=None,
        extra_sheets: Optional[Mapping[str, PolarsLike]]=None
    ) -> None:
    
    ytrue = np.asarray(ytrue)
    ycalc = np.asarray(ycalc)

    metadata = coalesce(metadata, {})
    context = coalesce(context, {})
    extra_sheets = coalesce(extra_sheets, {})
    unit = coalesce(unit, 'unknown  ')

    dataframe = transform_dataframe(metadata)
    dataframe = add_calculated_columns(dataframe, {
        f'Y_TRUE': ytrue,
        f'Y_CALC': ycalc,
        f'Y_DIFF':              calculate_deviations(ytrue, ycalc, absolute=False, relative=False),
        f'Y_DIFF_ABSOLUTE':     calculate_deviations(ytrue, ycalc, absolute=True,  relative=False),
        f'Y_DIFF_%':            calculate_deviations(ytrue, ycalc, absolute=False, relative=True, percentage=True),
        f'Y_DIFF_ABSOLUTE_%':   calculate_deviations(ytrue, ycalc, absolute=True,  relative=True, percentage=True),
    })

    r2 = r2_score(ytrue, ycalc)
    mae = mean_absolute_error(ytrue, ycalc)
    mape = mean_absolute_percentage_error(ytrue, ycalc)
    mse = mean_squared_error(ytrue, ycalc)
    max_dev = max_error(ytrue, ycalc)
    
    statistics = [
        {'METRIC': 'R2',        'UNIT': '1',        'VALUE': r2},
        {'METRIC': 'MAE',       'UNIT': f'{unit}',  'VALUE': mae},
        {'METRIC': 'MAPE',      'UNIT': f'%',       'VALUE': mape},
        {'METRIC': 'MSE',       'UNIT': f'{unit}²', 'VALUE': mse},
        {'METRIC': 'MAX_ERROR', 'UNIT': f'{unit}',  'VALUE': max_dev},
    ]
    sheets = {
        'DATA':         dataframe,
        'STATISTICS':   statistics,
        'CONTEXT':      context,
        **extra_sheets
    }
    write_excel_sheets(excel_filepath, sheets)

def workspace_path(root: PathLike, folder_name: str, /) -> Path:
    return Path(root).joinpath(str(folder_name) )

def setup_simulations_parallel(simulations: Sequence[Simulation], playmol_command: str, /, njobs: Optional[int]=None, maintain_order: bool=False) -> None:
    process = lambda simulation: (
        setup_simulation(simulation, playmol_command, new_terminal=False)
    )
    with open_monitor(total=len(simulations), title='PLAYMOL', description='Prepare simulation inputs with PlayMol') as ui:
        for _ in run_parallel(simulations, process=process, njobs=njobs, maintain_order=maintain_order):
            ui.communicate()

def run_simulations_parallel(simulations: Sequence[Simulation], lammps_commands: str | Sequence[str], /, estructure_outputs: bool=False) -> None:
    if isinstance(lammps_commands, str):
        lammps_commands = [lammps_commands]
    
    processes = (
        lambda simulation: (
            run_simulation(simulation, command, new_terminal=True, structure_outputs=estructure_outputs)
        ) 
        for command in lammps_commands
    )
    with open_monitor(total=len(simulations), title='LAMMPS', description='Running simulations with LAMMPS') as ui:
        for _ in run_work_pool(simulations, processes=processes):
            ui.communicate()

def calculate_simulations_result_parallel(simulations: Sequence[Simulation], /, njobs: Optional[int]=None, maintain_order: bool=False) -> list[SimulationAssets]:
    process = lambda simulation: (
        calculate_simulation_assets(simulation)
    )
    outputs = []
    with open_monitor(total=len(simulations), title='RESULTS', description='Calculating simulation assets') as ui:
        for output in run_parallel(simulations, process=process, njobs=njobs, maintain_order=maintain_order):
            outputs.append(output)
            ui.communicate()
    return outputs

def export_molecular_dynamics_assets(
        excel_filepath: PathLike,
        systems: PolarsLike,
        coeffs: PolarsLike,
        viscosity_experimental: ColumnName,
        simulations_result: Sequence[SimulationAssets]
    ) -> None:

    systems = transform_dataframe(systems)

    viscosity_calculated = np.array([
        data.viscosity_average for data in simulations_result
    ])
    viscosity_uncertainty = np.array([
        data.viscosity_uncertainty for data in simulations_result
    ])
    viscosity_experimental = systems.get_column(viscosity_experimental).to_numpy()

    systems = add_calculated_columns(systems, {
        'VISCOSITY_MD_CP': viscosity_calculated,
        'VISCOSITY_MD_UNCERTAINTY_CP': viscosity_uncertainty
    })
    export_regression_results_workbook(
        excel_filepath,
        ytrue=viscosity_experimental,
        ycalc=viscosity_calculated,
        unit='cP',
        metadata=systems,
        context={
            'Y_TRUE':   'Experimental viscosity',
            'Y_CALC':   'Viscosity calculated by Molecular Dynamics',
            'Y_UNIT':   'Centipoise [cP]'
        },
        extra_sheets={
            'PAIR_COEFFS': coeffs
        }
    )