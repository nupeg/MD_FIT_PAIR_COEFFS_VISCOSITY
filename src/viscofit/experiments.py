from typing import Iterator
from dataclasses import dataclass   

from viscofit.utils import dict_from_lists
from viscofit.metrics import mean_squared_error
from viscofit.tables import PolarsLike, transform_dataframe
from viscofit.settings import Settings
from viscofit.protocols import SettingsHandler
from viscofit.molecular_dynamics import PairCoeff, MixtureData
from viscofit.fit_params import PairCoeffRange, ExperimentCase, fit_viscosity_coeffs
from viscofit.datahub import (
    DataContext,
    SystemSchema,
    CoeffSchema,
    CoeffRangeSchema,
    load_systems,
    load_coeffs, 
    load_coeffs_range
)

@dataclass(slots=True, frozen=True)
class Experiment:
    func: SettingsHandler
    name: str
    description: str

class ExperimentRegister:

    @classmethod
    def iter_experiments(cls) -> Iterator[Experiment]: 
        values = vars(cls).values()
        selector = lambda data: isinstance(data, Experiment)
        return filter(selector, values)

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


def extract_experimental_cases(systems: PolarsLike, /) -> list[ExperimentCase]: 
    experimental_data = []

    for system in transform_dataframe(systems).iter_rows(named=True):
        electrolyte = system[SystemSchema.ELECTROLYTE]
        molality = system[SystemSchema.MOLALITY]
        pressure = system[SystemSchema.PRESSURE]
        temperature = system[SystemSchema.TEMPERATURE]
        reference_viscosity = system[SystemSchema.VISCOSITY]
        
        cation = system[SystemSchema.CATION]
        cation_esteq = system[SystemSchema.CATION_ESTEQ]
        cation_charge = system[SystemSchema.CATION_CHARGE]
        
        anion = system[SystemSchema.ANION]
        anion_esteq = system[SystemSchema.ANION_ESTEQ]
        anion_charge = system[SystemSchema.ANION_CHARGE]

        folder_name = f'Solvent=H2O_Solute={electrolyte}_M={molality}_T={temperature}_P={pressure}'

        compositions = {
            ...
        }

        instance = ExperimentCase(
            system=MixtureData(
                folder_name=folder_name,
                pressure=pressure,
                temperature=temperature,
                compositions=compositions
            ),
            reference_viscosity=reference_viscosity
        )
        experimental_data.append(instance)
    return experimental_data

def extract_coeffs(coeffs: PolarsLike, /) -> list[PairCoeff]: 
    ...

def extract_coeffs_ranges(coeffs_ranges: PolarsLike, /) -> list[PairCoeffRange]: 
    ... 


def preliminar_tests(config: Settings, /) -> None: 
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
        greater_is_better=True,
    )


def fit_params(config: Settings, /) -> None:
    systems = load_systems(DataContext.FIT_PARAMS)
    coeffs = load_coeffs(DataContext.FIT_PARAMS)
    coeffs_range = load_coeffs_range(DataContext.FIT_PARAMS)

def evaluate_madrid_2019(config: Settings, /) -> None: 
    systems = load_systems(DataContext.EVALUATE_MADRID_2019)
    coeffs = load_coeffs(DataContext.EVALUATE_MADRID_2019)