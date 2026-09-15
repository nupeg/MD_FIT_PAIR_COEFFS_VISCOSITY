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


