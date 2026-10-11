from typing import Iterator, NamedTuple, TypeAlias, ClassVar

from dataclasses import dataclass   
from pathlib import Path
from math import ceil

from viscofit.utils import dict_from_lists
from viscofit.tables import PolarsLike, transform_dataframe
from viscofit.settings import Settings
from viscofit.protocols import SettingsHandler
from viscofit.molecular_dynamics import ( 
    PairCoefficient, 
    BoxDimensions,
    SimulationFilenames,
    BondType,
    AngleType,
    Molecule,
    Simulation
)
from viscofit.datahub import (
    SystemSchema,
    CoeffSchema,
    CoeffRangeSchema,
    get_molecule_filepath,
    extract_atom_types,
    extract_bond_types,
    extract_angle_types
)


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


