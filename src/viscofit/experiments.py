from viscofit.settings import Settings

from viscofit.datahub import (
    FILEPATH_PURE_WATER_RESULT,
    DataContext,
    SystemSchema,
    load_systems,
    load_coeffs, 
    load_coeffs_range
)
from viscofit.experiments_helpers import (
    workspace_path,
    setup_simulations_parallel, 
    run_simulations_parallel,
    calculate_simulations_result_parallel,
    export_molecular_dynamics_assets
)
from viscofit.adapters import adapt_simulations_from_tables


def pure_water_simulations(config: Settings, /) -> None:
    '''
    Run molecular dynamics simulations to calculate the viscosity of pure water
    using the Green-Kubo method.

    The simulation results are compared with reference viscosity data to evaluate
    how well the simulation reproduces the expected viscosity of pure water.
    '''
    systems = load_systems(DataContext.PURE_WATER_SIMULATIONS, only_actives=True)
    coeffs = load_coeffs(DataContext.PURE_WATER_SIMULATIONS)

    root = workspace_path(config.root_folder, DataContext.PURE_WATER_SIMULATIONS)
    simulations = adapt_simulations_from_tables(root, systems, coeffs)

    setup_simulations_parallel(simulations, config.playmol_command, njobs=config.njobs, maintain_order=False)
    run_simulations_parallel(simulations, config.lammps_commands, estructure_outputs=True)

    # NOTE: Always maintain_order = TRUE for calculations
    simulations_result = calculate_simulations_result_parallel(simulations, njobs=config.njobs, maintain_order=True)
    
    export_molecular_dynamics_assets(
        FILEPATH_PURE_WATER_RESULT,
        systems=systems,
        coeffs=coeffs,
        viscosity_experimental=SystemSchema.VISCOSITY,
        simulations_result=simulations_result
    )


    


    