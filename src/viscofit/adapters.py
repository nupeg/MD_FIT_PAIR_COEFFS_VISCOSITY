from typing import Iterator, NamedTuple, TypeAlias, ClassVar

from dataclasses import dataclass   
from pathlib import Path

from viscofit.utils import dict_from_lists
from viscofit.tables import PolarsLike, transform_dataframe
from viscofit.settings import Settings
from viscofit.protocols import SettingsHandler
from viscofit.molecular_dynamics import ( 
    PairCoeff, 
    MixtureData, 
    MoleculeData,
    BoxDimensions,
    SimulationFiles
)
from viscofit.datahub import (
    SystemSchema,
    CoeffSchema,
    CoeffRangeSchema,
    get_molecule_filepath,
    extract_atom_types
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

class RealDimension(NamedTuple):
    low: float
    high: float

@dataclass(slots=True, frozen=True)
class PairCoeffRange:
    atom_type_01: str
    atom_type_02: str
    sigma_dimension: RealDimension
    epsilon_dimension: RealDimension

@dataclass(slots=True, frozen=True)
class ExperimentalData:
    system: MixtureData
    reference_viscosity: float

@dataclass(slots=True, frozen=True)
class UserSimulationInputs:
    n_particles_water: int
    root_folder_path: Path
    playmol_cmd: str
    lammps_cmd: list[str]
    box: BoxDimensions
    files: SimulationFiles


LiteralDatabasePath:   TypeAlias = str


def build_molecule(molecule_name: str, /) -> MoleculeData:
    filepath = get_molecule_filepath(molecule_name)
    atom_types = extract_atom_types(filepath, raise_empty=True)

    instance = MoleculeData(
        name=molecule_name,
        filepath=filepath,
        atom_types=atom_types
    )
    return instance

def adapt_experimental_data(systems: PolarsLike, water_particles: int, /) -> list[ExperimentalData]: 
    experimental_data = []  

    for system in transform_dataframe(systems).iter_rows(named=True):
        electrolyte = system[SystemSchema.ELECTROLYTE]
        molality = system[SystemSchema.MOLALITY]
        pressure = system[SystemSchema.PRESSURE]
        temperature = system[SystemSchema.TEMPERATURE]
        reference_viscosity = system[SystemSchema.VISCOSITY]
        
        cation = system[SystemSchema.CATION]
        cation_esteq = system[SystemSchema.CATION_ESTEQ]

        anion = system[SystemSchema.ANION]
        anion_esteq = system[SystemSchema.ANION_ESTEQ]

        water_molecule = build_molecule('H2O')

        compositions = {
            water_molecule: water_particles  
        }   

        if molality > 0.0:
            anion_molecule  = build_molecule(anion)
            cation_molecule = build_molecule(cation)

            water_mass_kg = (water_particles / AVOGADRO) * WATER_MOLAR_MASS_KG

            anion_particles  = (water_mass_kg * molality * anion_esteq)  * AVOGADRO
            cation_particles = (water_mass_kg * molality * cation_esteq) * AVOGADRO

            compositions[anion_molecule] = anion_particles
            compositions[cation_molecule] = cation_particles
        
        folder_name = f'Solvent=H2O_Solute={electrolyte}_M={molality}_T={temperature}_P={pressure}'

        instance = ExperimentalData(
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

def adapt_coeffs(coeffs: PolarsLike, /) -> list[PairCoeff]: 
    coeffs = transform_dataframe(coeffs)
    output = [
        PairCoeff(
            atom_type_01=row[CoeffSchema.ATOM_TYPE_1],
            atom_type_02=row[CoeffSchema.ATOM_TYPE_2],
            sigma=row[CoeffSchema.SIGMA],
            epsilon=row[CoeffSchema.EPSILON]
        )
        for row in coeffs.iter_rows(named=True)
    ]
    return output

def adapt_coeffs_ranges(coeff_ranges: PolarsLike, /) -> list[PairCoeffRange]: 
    coeff_ranges = transform_dataframe(coeff_ranges) 
    output = [
        PairCoeffRange(
            atom_type_01=row[CoeffRangeSchema.ATOM_TYPE_1],
            atom_type_02=row[CoeffRangeSchema.ATOM_TYPE_2],
            sigma_dimension=RealDimension(
                low=row[CoeffRangeSchema.SIGMA_MIN], 
                high=row[CoeffRangeSchema.SIGMA_MAX]
            ),
            epsilon_dimension=RealDimension(
                low=row[CoeffRangeSchema.EPSILON_MIN], 
                high=row[CoeffRangeSchema.EPSILON_MAX]
            )
        )
        for row in coeff_ranges.iter_rows(named=True)
    ]
    return output

def adapt_simulation_inputs(config: Settings, /) -> UserSimulationInputs:
    n_particles_water = int(config.n_particles_water)
    root_folder_path = Path(config.root_folder_path)
    playmol_cmd = str(config.playmol_cmd)
    lammps_cmd = config.lammps_cmd

    if isinstance(lammps_cmd, str):
        lammps_cmd = [lammps_cmd]

    lammps_cmd = list(lammps_cmd)

    box = BoxDimensions(*config.box)

    files = SimulationFiles(
        simulation          = str(config.filename_simulation),
        coeffs              = str(config.filename_coeffs),
        start_box_playmol   = str(config.filename_start_box_playmol),
        start_box_lammps    = str(config.filename_start_box_lammps),
        start_box_xyz       = str(config.filename_start_box_xyz)
    )

    output = UserSimulationInputs(
        n_particles_water=n_particles_water,
        root_folder_path=root_folder_path,
        playmol_cmd=playmol_cmd,
        lammps_cmd=lammps_cmd,
        box=box,
        files=files
    )
    return output