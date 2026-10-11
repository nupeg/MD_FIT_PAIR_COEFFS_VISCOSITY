from viscofit.settings import Settings

from viscofit.metrics import mean_squared_error
from viscofit.tables import write_excel_sheets

from viscofit.datahub import (
    FILEPATH_PURE_WATER_RESULT,
    DataContext,
    load_systems,
    load_coeffs, 
    load_coeffs_range
)
from viscofit.experiments_helpers import setup_simulations_parallel


def pure_water_simulations(config: Settings, /) -> None:
    '''
    Run molecular dynamics simulations to calculate the viscosity of pure water
    using the Green-Kubo method.

    The simulation results are compared with reference viscosity data to evaluate
    how well the simulation reproduces the expected viscosity of pure water.
    '''
    systems = load_systems(DataContext.PURE_WATER_SIMULATIONS)
    coeffs = load_coeffs(DataContext.PURE_WATER_SIMULATIONS)
    
    simulations = ...
    setup_simulations_parallel(simulations, maintain_order=False)


    