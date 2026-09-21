from viscofit.settings import Settings

from viscofit.metrics import mean_squared_error
from viscofit.tables import write_excel_sheets
from viscofit.experiments_helpers import (
    fit_viscosity_coeffs, 
    complete_pair_coeffs,
    lorentz_berthelot_epsilon,
    lorentz_berthelot_sigma,
    validate_mixtures_coeffs,
    create_simulation_setups,
    setup_simulations_parallel,
    execute_simulations_parallel,
    calculate_viscosity_parallel,
    workspace_folder,
    evaluate_predictions
)
from viscofit.adapters import (
    adapt_coeffs,
    adapt_coeffs_ranges,
    adapt_experimental_data,
    adapt_simulation_inputs
)
from viscofit.datahub import (
    FILEPATH_PURE_WATER_RESULT,
    DataContext,
    load_systems,
    load_coeffs, 
    load_coeffs_range
)


def pure_water_simulations(config: Settings, /) -> None:
    '''
    Run molecular dynamics simulations to calculate the viscosity of pure water
    using the Green-Kubo method.

    The simulation results are compared with reference viscosity data to evaluate
    how well the simulation reproduces the expected viscosity of pure water.
    '''
    systems_dataframe = load_systems(DataContext.PURE_WATER_SIMULATIONS)
    coeffs_dataframe = load_coeffs(DataContext.PURE_WATER_SIMULATIONS)
    
    coeffs = adapt_coeffs(coeffs_dataframe)
    simulation_inputs = adapt_simulation_inputs(config)
    experimental_data = adapt_experimental_data(systems_dataframe, simulation_inputs.n_particles_water)

    systems = [data.system for data in experimental_data]
    coeffs = complete_pair_coeffs(coeffs, sigma_rule=lorentz_berthelot_sigma, epsilon_rule=lorentz_berthelot_epsilon)

    validate_mixtures_coeffs(systems, coeffs)
    workspace = workspace_folder(simulation_inputs.root_folder_path, DataContext.PURE_WATER_SIMULATIONS)

    simulations = create_simulation_setups(
        systems=systems,
        coeffs=coeffs,
        box=simulation_inputs.box,
        files=simulation_inputs.files,
        base_folder=workspace,
        npt_steps=simulation_inputs.npt_steps,
        nvt_steps=simulation_inputs.nvt_steps,
        num_trajectories=simulation_inputs.num_trajectories
    )
    setup_simulations_parallel(simulations, simulation_inputs.playmol_cmd)
    execute_simulations_parallel(simulations, simulation_inputs.lammps_cmd)

    viscosity_assets = calculate_viscosity_parallel(simulations, njobs=config.njobs)

    calculations = [data.viscosity_average for data in viscosity_assets]
    references = [data.reference_viscosity for data in experimental_data]
    
    scores, evaluations = evaluate_predictions(calculations, references, dataframe=systems_dataframe)

    write_excel_sheets(FILEPATH_PURE_WATER_RESULT, {
        'DATABASE': evaluations,
        'SCORES': scores
    })
    
def preliminar_tests(config: Settings, /) -> None:
    '''
    DOCSTRING
    '''
    simulation_inputs = adapt_simulation_inputs(config)

    systems_dataframe = load_systems(DataContext.PRELIMINAR_TEST)
    coeffs_dataframe = load_coeffs(DataContext.PRELIMINAR_TEST)
    coeffs_range_dataframe = load_coeffs_range(DataContext.PRELIMINAR_TEST)
    
    coeffs_values = adapt_coeffs(coeffs_dataframe)
    coeffs_ranges = adapt_coeffs_ranges(coeffs_range_dataframe)
    experimental_data = adapt_experimental_data(systems_dataframe, simulation_inputs.n_particles_water)
    
    optimization_result = fit_viscosity_coeffs(
        experimental_data=experimental_data,
        coeffs_ranges=coeffs_ranges,
        folder_path=simulation_inputs.root_folder_path,
        playmol_cmd=simulation_inputs.playmol_cmd,
        lammps_cmd=simulation_inputs.lammps_cmd,
        box=simulation_inputs.box,
        files=simulation_inputs.files,
        n_trials=...,
        checkpoint_folder_path=...,
        coeffs_values=coeffs_values,
        optimize_metric=mean_squared_error,
        checkpoint_name='preliminar_tests',
        show_progress_bar=True,
        greater_is_better=False,
    )
