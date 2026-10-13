from typing import Iterator, NamedTuple, TypeAlias, ClassVar

from os import PathLike
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
    SimulationCoefficients,
    SimulationConditions,
    SimulationSystem,
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


def build_molecule(name: str, /) -> Molecule:
    ...

def adapt_simulations_from_tables(root: PathLike, systems: PolarsLike, coeffs: PolarsLike, /) -> list[Simulation]:
    systems = transform_dataframe(systems)
    coeffs = transform_dataframe(coeffs)

    simulations = []
    for system in systems.iter_rows(named=True):


        solvent_molecule  = build_molecule()
        cation_molecule   = build_molecule()
        anion_molecule    = build_molecule()
        
        molecules = {...}
        
        box = BoxDimensions(
            x=...,
            y=...,
            z=...
        )
        filenames = SimulationFilenames(
            simulation='viscosity.lmp',
            coeffs='coeffs.lmp',
            start_box_lammps='start_box.lmp',
            start_box_playmol='start_box.playmol',
            start_box_xyz='start_box.xyz',
            nvt_output_data='nvt_lammps_data.parquet'
        )
        conditions = SimulationConditions(
            temperature=...,
            pressure=...,
            timestep=...,
            nvt_steps=...,
            npt_steps=...,
            num_trajectories=...
        )
        coeffs = SimulationCoefficients(
            pair=...,
            bond=...,
            angle=...
        )
        system = SimulationSystem(
            box=box,
            molecules=molecules
        )
        simulation = Simulation(
            root=root,
            filenames=filenames,
            conditions=conditions,
            coeffs=coeffs,
            system=system
        )
        simulations.append(simulation)

    return simulations
