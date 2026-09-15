from typing import Iterator
from dataclasses import dataclass   

from viscofit.utils import dict_from_lists
from viscofit.metrics import mean_squared_error
from viscofit.tables import PolarsLike, transform_dataframe
from viscofit.settings import Settings
from viscofit.protocols import SettingsHandler
from viscofit.molecular_dynamics import PairCoeff, MixtureData
from viscofit.fit_params import PairCoeffRange, ExperimentCase, fit_viscosity_coeffs
from viscofit.experiments_helpers import (
    Experiment,
    ExperimentCase,
    ExperimentRegister,
    extract_coeffs,
    extract_coeffs_ranges,
    extract_experimental_cases
)
from viscofit.datahub import (
    DataContext,
    SystemSchema,
    CoeffSchema,
    CoeffRangeSchema,
    load_systems,
    load_coeffs, 
    load_coeffs_range
)

def preliminar_tests(config: Settings, /) -> None:
    '''
    DOCSTRING
    '''
    systems_dataframe = load_systems(DataContext.PRELIMINAR_TEST)
    coeffs_dataframe = load_coeffs(DataContext.PRELIMINAR_TEST)
    coeffs_range_dataframe = load_coeffs_range(DataContext.PRELIMINAR_TEST)
        
    coeffs_values = extract_coeffs(coeffs_dataframe)
    coeffs_ranges = extract_coeffs_ranges(coeffs_range_dataframe)
    experimental_data = extract_experimental_cases(systems_dataframe)
    
    optimize_result = fit_viscosity_coeffs(
        experimental_data=experimental_data,
        coeffs_ranges=coeffs_values,
        coeffs_values=coeffs_ranges,
        n_trials=...,
        checkpoint_folder_path=...,
        optimize_metric=mean_squared_error,
        checkpoint_name='preliminar_tests',
        show_progress_bar=True,
        greater_is_better=False,
    )


def fit_params(config: Settings, /) -> None:
    systems = load_systems(DataContext.FIT_PARAMS)
    coeffs = load_coeffs(DataContext.FIT_PARAMS)
    coeffs_range = load_coeffs_range(DataContext.FIT_PARAMS)

def evaluate_madrid_2019(config: Settings, /) -> None: 
    systems = load_systems(DataContext.EVALUATE_MADRID_2019)
    coeffs = load_coeffs(DataContext.EVALUATE_MADRID_2019)