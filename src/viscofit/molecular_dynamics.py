from typing import Mapping, Sequence, Hashable, TypeAlias, Final, Annotated

from os import PathLike
from pathlib import Path
from dataclasses import dataclass

import polars as pl

from viscofit.protocols import IdentityHashing
from viscofit.datahub import (
    get_molecule_filepath,
    write_text,
    parse_playmol_start_box_template,
    parse_coefficients_template,
    parse_lammps_viscosity_template
    
)
from viscofit.utils import (
    copy_files, 
    setup_dir, 
    ensure_file_exists, 
    execute_script, 
    filename,
    enumerate_unique,
    join_as_text,
    search_files
)


NEW_ROW:    Final[str] = '\n'
WHITESPACE: Final[str] = ' '

BOLTZMANN_CONSTANT: Annotated[float, 'J/K'] = 1.380649e-23

ATM_TO_PA:                  Final[float] = 101325
ANGSTROM_TO_METER:          Final[float] = 1e-10
PA_S_TO_CENTIPOISE:         Final[float] = 1000
FEMTOSECOND_TO_SECOND:      Final[float] = 1e-15
VISCOSITY_CUTOFF_TIME_FS:   Final[float] = 200_000

MoleculeID: TypeAlias = int

@dataclass(slots=True, frozen=True)
class BoxDimensions:
    x: float
    y: float
    z: float

@dataclass(slots=True, frozen=True)
class PairCoefficient:
    atom_01: str
    atom_02: str
    epsilon: float
    sigma: float

@dataclass(slots=True, frozen=True)
class BondType:
    @property
    def values(self) -> tuple[float, ...]: ...

@dataclass(slots=True, frozen=True)
class AngleType:
    @property
    def values(self) -> tuple[float, ...]: ...

@dataclass(slots=True, frozen=True, eq=False)
class Molecule(IdentityHashing):
    name: str

    @property
    def identity(self) -> Hashable:
        return id(self)

@dataclass(slots=True, frozen=True)
class SimulationFilenames:
    simulation: str
    coeffs: str
    start_box_lammps: str
    start_box_playmol: str
    start_box_xyz: str
    nvt_data: str

@dataclass(slots=True, frozen=True)
class SimulationConditions:
    temperature: float
    pressure: float
    timestep: float
    npt_steps: int
    nvt_steps: int
    num_trajectories: int

@dataclass(slots=True, frozen=True)
class SimulationSystem:
    box: BoxDimensions
    molecules: Mapping[Molecule, int]

    @property
    def molecule_names(self) -> list[str]:
        return [molecule.name for molecule in self.molecules.keys() ]
    
@dataclass(slots=True, frozen=True)
class SimulationCoefficients:
    pair: Sequence[PairCoefficient]
    bond: Sequence[BondType]
    angle: Sequence[AngleType]


@dataclass(slots=True, frozen=True)
class Simulation:
    root: PathLike
    filenames: SimulationFilenames
    conditions: SimulationConditions
    coeffs: SimulationCoefficients
    system: SimulationSystem

    @property
    def box(self) -> BoxDimensions:
        return self.system.box

    @property
    def input_folder(self) -> Path:
        return Path(self.root).joinpath('inputs')

    @property
    def lammps_folder(self) -> Path:
        return Path(self.root).joinpath('lammps')

    @property
    def data_folder(self) -> Path:
        return Path(self.root).joinpath('data')

    @property
    def playmol_input_filepath(self) -> Path:
        return self.input_folder.joinpath(self.filenames.start_box_playmol)

    @property
    def playmol_lammps_output_filepath(self) -> Path:
        return self.input_folder.joinpath(self.filenames.start_box_lammps)

    @property
    def playmol_xyz_output_filepath(self) -> Path:
        return self.input_folder.joinpath(self.filenames.start_box_xyz)

    @property
    def lammps_input_filepath(self) -> Path:
        return self.input_folder.joinpath(self.filenames.simulation)

    @property
    def coeffs_filepath(self) -> Path:
        return self.input_folder.joinpath(self.filenames.coeffs)

    @property
    def nvt_data_output_filepath(self) -> Path:
        return self.data_folder.joinpath(self.filenames.nvt_data)

    @property
    def playmol_molecules(self) -> dict[str, int]:
        return {molecule.name: count for molecule, count in self.system.molecules.items() }

    @property
    def molecule_names(self) -> list[str]:
        return self.system.molecule_names

    @property
    def temperature(self) -> float: ...

    @property
    def pressure(self) -> float: ...

    @property
    def timestep(self) -> float: ...

    @property
    def npt_steps(self) -> int: ...

    @property
    def nvt_steps(self) -> int: ...

    @property
    def num_trajectories(self) -> int: ...

    @property
    def atom_types(self) -> list[str]: ...

    @property
    def pair_coeffs(self) -> list[PairCoefficient]: ...

    @property
    def bond_types(self) -> list[BondType]: ...

    @property
    def angle_types(self) -> list[AngleType]: ...

    @property
    def simulation_done(self) -> bool: ...

    @property
    def outputs_structured(self) -> bool: ...




def write_playmol_input_file(
        filepath: PathLike,
        molecules: Mapping[PathLike, int],
        box: BoxDimensions,
        lammps_output_file: str,
        xyz_output_file: str
    ) -> None:

    includes = f'\n'.join(
        f'include {filepath}' for filepath in molecules.keys()
    )
    packs = f' '.join(
        f'{index} {count}' for index, count in enumerate(molecules.values(), start=1)
    )
    
    box_dimensions = f'{box.x} {box.y} {box.z}'

    content = parse_playmol_start_box_template(includes, box_dimensions, packs, lammps_output_file, xyz_output_file)
    write_text(filepath, content)

def write_simulation_input_file(
        filepath: PathLike,
        temperature: float,
        pressure: float,
        start_box_file: str,
        coeffs_file: str,
        timestep: float,
        npt_steps: int,
        nvt_steps: int,
        num_trajectories: int
    ) -> None:
    
    content = parse_lammps_viscosity_template(
        float(temperature),
        float(pressure),
        str(start_box_file),
        str(coeffs_file),
        float(timestep),
        int(npt_steps),
        int(nvt_steps),
        int(num_trajectories)
    )
    write_text(filepath, content)

def write_coefficients_file(
        filepath: PathLike,
        atom_types: Sequence[str],
        pair_coeffs: Sequence[PairCoefficient],
        bond_types: Sequence[BondType],
        angle_types: Sequence[AngleType],
    ) -> None:

    atom_type_indices = enumerate_unique(atom_types)
    bond_type_indices = enumerate_unique(bond_types)
    angle_type_indices = enumerate_unique(angle_types)

    bond_coeff_values = {
        index: join_as_text(bond.values, WHITESPACE)
        for index, bond in bond_type_indices.items()
    }
    angle_coeff_values = {
        index: join_as_text(angle.values, WHITESPACE)
        for index, angle in angle_type_indices.items()
    }
    atom_type_commands = NEW_ROW.join(
        f'labelmap atom {index} {atom}'
        for index, atom in atom_type_indices.items()
    )
    pair_coeff_commands = NEW_ROW.join(
        f'pair_coeff {coeff.atom_01} {coeff.atom_02} {coeff.epsilon} {coeff.sigma}'
        for coeff in pair_coeffs
    )
    bond_coeff_commands = NEW_ROW.join(
        f'bond_coeff {index} {values}'
        for index, values in bond_coeff_values.items()
    )
    angle_coeff_commands = NEW_ROW.join(
        f'angle_coeff {index} {values}'
        for index, values in angle_coeff_values.items()
    )

    content = parse_coefficients_template(
        atom_type_commands,
        pair_coeff_commands,
        bond_coeff_commands,
        angle_coeff_commands,
    )
    write_text(filepath, content)

def setup_simulation(data: Simulation, playmol_command: str, /, new_terminal: bool=False) -> None:
    setup_dir(data.input_folder, create=True, clear=True)

    molecule_files = map(get_molecule_filepath, data.molecule_names)
    molecule_filepaths = copy_files(*molecule_files, data.input_folder)

    write_playmol_input_file(
        box=data.box,
        filepath=data.playmol_input_filepath,
        molecules=data.playmol_molecules,
        lammps_output_file=filename(data.playmol_lammps_output_filepath),
        xyz_output_file=filename(data.playmol_xyz_output_filepath)
    )
    ensure_file_exists(
        *molecule_filepaths, data.playmol_input_filepath
    )
    execute_script(
        filepath=data.playmol_input_filepath,
        cmd=playmol_command,
        new_terminal=new_terminal
    )
    write_coefficients_file(
        filepath=data.coeffs_filepath,
        atom_types=data.atom_types,
        pair_coeffs=data.pair_coeffs,
        bond_types=data.bond_types,
        angle_types=data.angle_types
    )
    write_simulation_input_file(
        filepath=data.lammps_input_filepath,
        temperature=data.temperature,
        pressure=data.pressure,
        start_box_file=filename(data.playmol_lammps_output_filepath),
        coeffs_file=filename(data.coeffs_filepath),
        timestep=data.timestep,
        npt_steps=data.npt_steps,
        nvt_steps=data.nvt_steps,
        num_trajectories=data.num_trajectories
    )
    ensure_file_exists(
        data.playmol_lammps_output_filepath,
        data.playmol_xyz_output_filepath,
        data.lammps_input_filepath,
        data.coeffs_filepath
    )

def run_simulation(data: Simulation, lammps_command: str, /, new_terminal: bool=False) -> None:
    if data.simulation_done:
        return
    execute_script(
        data.lammps_input_filepath,
        lammps_command,
        new_terminal=new_terminal
    )

def structure_simulation_outputs(data: Simulation, /) -> None:
    if data.outputs_structured:
        return
    
    output_files = search_files(data.lammps_folder, r'*.RUN')

    schema = {
        'step':             pl.Int64,
        'timestep':         pl.Float64,
        'time':             pl.Float64,
        'temperature':      pl.Float64,
        'volume':           pl.Float64,
        'pxy':              pl.Float64,
        'pyz':              pl.Float64,
        'pxz':              pl.Float64,
        'pxx':              pl.Float64,
        'pyy':              pl.Float64,
        'pzz':              pl.Float64,
        'total_energy':     pl.Float64,
    }

    dataframes: list[pl.LazyFrame] = []

    for index, filepath in enumerate(output_files, start=1):
        dataframe = pl.scan_csv(filepath, separator='|', schema_overrides=schema)
        dataframe = dataframe.rename({
            'timestep':     'TIMESTEP_FS',
            'step':         'STEP',
            'time':         'TIME_FS',
            'temperature':  'TEMPERATURE_KELVIN',
            'volume':       'VOLUME_ANGSTROM3',
            'pxy':          'PXY_ATM',
            'pyz':          'PYZ_ATM',
            'pxz':          'PXZ_ATM',
            'pxx':          'PXX_ATM',
            'pyy':          'PYY_ATM',
            'pzz':          'PZZ_ATM',
            'total_energy': 'TOTAL_ENERGY_KCAL_MOL',
        })
        dataframe = dataframe.with_columns(
            (pl.lit(index)).alias('TRAJECTORY_ID'),
            (pl.lit(filepath.stem)).alias('TRAJECTORY_NAME'),
            (pl.col('TIME_FS') * FEMTOSECOND_TO_SECOND).alias('TIME_SECONDS'),
            (pl.col('VOLUME_ANGSTROM3') * ANGSTROM_TO_METER**3).alias('VOLUME_M3'),
            (pl.col('PXY_ATM') * ATM_TO_PA).alias('PXY_PASCAL'),
            (pl.col('PYZ_ATM') * ATM_TO_PA).alias('PYZ_PASCAL'),
            (pl.col('PXZ_ATM') * ATM_TO_PA).alias('PXZ_PASCAL'),
            (pl.col('PXX_ATM') * ATM_TO_PA).alias('PXX_PASCAL'),
            (pl.col('PYY_ATM') * ATM_TO_PA).alias('PYY_PASCAL'),
            (pl.col('PZZ_ATM') * ATM_TO_PA).alias('PZZ_PASCAL'),
        )
        dataframes.append(dataframe)

    if not dataframes:
        raise ValueError(f'No LAMMPS output files found in {data.lammps_folder}.')

    final_dataframe = pl.concat(dataframes, how='vertical')
    final_dataframe.sink_parquet(data.nvt_data_output_filepath)

def calculate_viscosity_assets(data: Simulation, /) -> None: 
    ...


