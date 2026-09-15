from typing import (
    Sequence, 
    TypeAlias, 
    NamedTuple, 
    Mapping, 
    Iterable,
    Optional
)

from os import PathLike
from pathlib import Path
from functools import partial
from dataclasses import dataclass

from viscofit.utils import (
    keys, 
    values,
    filename,
    setup_dir, 
    copy_files,
    parse_cmd, 
    join_as_text,
    run_parallel,
    run_work_pool,
    execute_script,
    enumerate_unique,
    ensure_file_exists,
    ensure_not_duplicates
)
from viscofit.datahub import (
    parse_coefficients_template,
    parse_lammps_viscosity_template,
    parse_playmol_start_box_template    
)
from viscofit.monitor import open_monitor


Count:              TypeAlias = int
Command:            TypeAlias = str
MoleculeFilepath:   TypeAlias = PathLike

class BoxDimensions(NamedTuple):
    x: float
    y: float
    z: float

class SimulationFiles(NamedTuple):
    simulation: str
    coeffs: str
    start_box_playmol: str
    start_box_lammps: str
    start_box_xyz: str


@dataclass(slots=True, frozen=True)
class PairCoeff:
    atom_type_01: str
    atom_type_02: str
    sigma: float
    epsilon: float

@dataclass(slots=True, frozen=True)
class MoleculeData:
    name: str
    filepath: Path
    atom_types: tuple[str, ...]

@dataclass(slots=True, frozen=True)
class MixtureData:
    folder_name: str
    pressure: float
    temperature: float
    compositions: dict[MoleculeData, int]
    
    @property
    def packs(self) -> dict[MoleculeFilepath, int]:
        return {
            Path(mol.filepath): count for mol, count in self.compositions.items() 
        }

    @property
    def atom_indices(self) -> dict[str, int]:
        full_atom_types = (
            atom_type for mol in keys(self.compositions) for atom_type in mol.atom_types
        )
        return enumerate_unique(full_atom_types)

    @property
    def molecule_filepaths(self) -> list[Path]:
        return [Path(mol.filepath) for mol in self.compositions]

@dataclass(slots=True, frozen=True)
class SimulationSetup:
    folder_path: Path
    files: SimulationFiles
    box: BoxDimensions
    mixture: MixtureData
    coeffs: list[PairCoeff]
    
    @property
    def temperature(self) -> float:
        return self.mixture.temperature 

    @property
    def pressure(self) -> float:
        return self.mixture.pressure 
    
    @property
    def start_box_playmol_filepath(self) -> Path:
        return Path(self.folder_path).joinpath(self.files.start_box_playmol)

    @property
    def start_box_lammps_filepath(self) -> Path:
        return Path(self.folder_path).joinpath(self.files.start_box_lammps)

    @property
    def start_box_xyz_filepath(self) -> Path:
        return Path(self.folder_path).joinpath(self.files.start_box_xyz)

    @property
    def simulation_filepath(self) -> Path:
        return Path(self.folder_path).joinpath(self.files.simulation)

    @property
    def coeffs_filepath(self) -> Path:
        return Path(self.folder_path).joinpath(self.files.coeffs)

    @property
    def molecule_filepaths(self) -> list[Path]: 
        return self.mixture.molecule_filepaths
    
    @property
    def packs(self) -> dict[MoleculeFilepath, int]:
        return self.mixture.packs
    
    @property
    def atom_indices(self) -> dict[str, int]:
        return self.mixture.atom_indices


def write_start_box_file(filepath: PathLike, packs: Mapping[MoleculeFilepath, Count], box: BoxDimensions, lammps_output_file: str, xyz_output_file: str) -> None:
    packs = dict(packs)

    molecule_filepaths = keys(packs)
    molecule_counts = values(packs)

    includes_text = (
        f'include {filepath}' for filepath in molecule_filepaths
    )
    compositions_text = (
        f'pack {index} {count}' for index, count in enumerate(molecule_counts, start=1)   
    )

    includes_text = join_as_text(includes_text, '\n')
    compositions_text = join_as_text(compositions_text, ' ')
    
    box_dimensions = f'{box.x} {box.y} {box.z}'

    content = parse_playmol_start_box_template(
        includes_text,
        box_dimensions,
        compositions_text,
        lammps_output_file,
        xyz_output_file)
    
    Path(filepath).write_text(content)

def write_coeffs_file(filepath: PathLike, coeffs: Sequence[PairCoeff], atom_types: Mapping[str, int] | Iterable[str], /) -> None:
    if not isinstance(atom_types, Mapping):
        atom_types = enumerate_unique(atom_types)
    
    atom_types_text = (
        f'labelmap atom {index} {atom}' for atom, index in atom_types.items()
    ) 
    pair_coeffs_text = (
        f'pair_coeff {data.atom_type_01} {data.atom_type_02} {data.epsilon} {data.sigma}' for data in coeffs
    )

    atom_types_text = join_as_text(atom_types_text, '\n')
    pair_coeffs_text = join_as_text(pair_coeffs_text, '\n')

    content = parse_coefficients_template(
        atom_types_text, 
        pair_coeffs_text)
    
    Path(filepath).write_text(content)

def write_simulation_file(filepath: PathLike, temperature: float, pressure: float, start_box_file: str, coeffs_file: str) -> None: 
    content = parse_lammps_viscosity_template(
        temperature,
        pressure,
        start_box_file,
        coeffs_file)
    
    Path(filepath).write_text(content)


def execute_simulation(filepath: PathLike, /, lammps_cmd: Command, new_terminal: bool=False) -> None:
    cmd_parts = parse_cmd(lammps_cmd)

    if not cmd_parts:
        raise ValueError('The LAMMPS command is empty or could not be parsed.')

    lammps_flags = ('-in', '<', '-i')

    if any(part in lammps_flags for part in cmd_parts):
        raise ValueError(
            f'Do not include input flags (-in, -i, or <) in the LAMMPS command. '
            f'The system automatically appends the input file at the end. '
            f'Final format: <command> -in <file>. '
            f'Command received: {lammps_cmd!r}'
        )

    cmd_parts.append('-in')
    execute_script(filepath, cmd_parts, new_terminal=new_terminal)

def setup_simulation_files(setup: SimulationSetup, /, playmol_cmd: Command, new_terminal: bool=False) -> None: 
    coeffs_file = filename(setup.coeffs_filepath)
    start_box_xyz_file = filename(setup.start_box_lammps_filepath)
    start_box_lammps_file = filename(setup.start_box_xyz_filepath)

    root = setup_dir(setup.folder_path, clear=True, create=True)
    copied_molecule_filepaths = copy_files(*setup.molecule_filepaths, destination_folder=root)
    
    write_coeffs_file(
        setup.coeffs_filepath,
        setup.coeffs,
        setup.atom_indices)

    write_start_box_file(
        setup.start_box_playmol_filepath,
        setup.packs,
        setup.box,
        start_box_lammps_file,
        start_box_xyz_file)

    write_simulation_file(
        setup.simulation_filepath,
        setup.temperature,
        setup.pressure,
        start_box_lammps_file,
        coeffs_file)

    ensure_file_exists(
        *copied_molecule_filepaths,
        setup.coeffs_filepath,
        setup.simulation_filepath,
        setup.start_box_playmol_filepath)
    
    execute_script(
        setup.start_box_playmol_filepath, 
        playmol_cmd, 
        new_terminal=new_terminal)

    ensure_file_exists(
        setup.start_box_lammps_filepath,
        setup.start_box_xyz_filepath)

def calculate_viscosity(simulation_folder: PathLike, /) -> float: 
    ...

def validate_simulation_setups(simulations: Sequence[SimulationSetup], /) -> None: 
    ensure_not_duplicates(setup.folder_path for setup in simulations)
    
def validate_simulation_coeffs(simulations: Sequence[SimulationSetup], coeffs: Sequence[PairCoeff], /) -> None: 
    ...

def get_workspace_path(folder_path: PathLike, /) -> Path:
    return Path(folder_path).joinpath('simulation_workspace')

def create_simulation_setups(
        systems: Sequence[MixtureData],
        coeffs: Sequence[PairCoeff],
        box: BoxDimensions,
        files: SimulationFiles, 
        base_folder: PathLike, 
        /
    ) -> list[SimulationSetup]: 
    
    base_folder = Path(base_folder)
    
    simulations = []

    for mixture in systems:
        folder_path = base_folder.joinpath(mixture.folder_name)
        setup = SimulationSetup(
            folder_path=folder_path,
            files=files,
            box=box,
            mixture=mixture,
            coeffs=coeffs
        )
        simulations.append(setup)
    return simulations

def setup_simulations_parallel(simulations: Sequence[SimulationSetup], playmol_cmd: Command, /, njobs: Optional[int]=None) -> None:
    total = len(simulations)

    process = lambda setup: setup_simulation_files(setup, playmol_cmd=playmol_cmd, new_terminal=False)

    with open_monitor(total=total, title='SETUP', description='Creating simulation files with Playmol') as ui:
        for _ in run_parallel(simulations, process=process, njobs=njobs):
            ui.communicate()

def execute_simulations_parallel(filepaths: Sequence[PathLike], lammps_cmd: Command | Sequence[Command], /) -> None: 
    total = len(filepaths)

    if isinstance(lammps_cmd, (str, Path, PathLike)):
        lammps_cmd = [lammps_cmd]

    processes = [
        lambda path: execute_simulation(path, lammps_cmd=cmd, new_terminal=True) for cmd in lammps_cmd
    ]

    with open_monitor(total=total, title='SIMULATION', description='Running simulations with LAMMPS') as ui:
        for _ in run_work_pool(filepaths, processes):
            ui.communicate()


def run_viscosity_simulation_pipeline(
        systems: Sequence[MixtureData], 
        coeffs: Sequence[PairCoeff],
        folder_path: PathLike,
        playmol_cmd: Command,
        lammps_cmd: Command | Sequence[Command], 
        box: BoxDimensions,
        files: SimulationFiles,
        njobs: Optional[int]=None
    ) -> None: 

    # NOTE: Never trust user folder-path
    simulation_workspace = get_workspace_path(folder_path)
    simulations = create_simulation_setups(
        systems, 
        coeffs, 
        box, 
        files, 
        simulation_workspace
    )
    validate_simulation_setups(simulations)
    validate_simulation_coeffs(simulations, coeffs)

    setup_simulations_parallel(simulations, playmol_cmd)
    execute_simulations_parallel(simulations, lammps_cmd)

def calculate_viscosity_parallel(simulation_folders: Sequence[PathLike], /, njobs: Optional[int]=None) -> list[float]:
    total = len(simulation_folders)
    
    calculations = []
    
    with open_monitor(total=total, title='CALCULATING', description='Calculating viscosity with Green-Kubo') as ui:
        for calculated_viscosity in run_parallel(simulation_folders, process=calculate_viscosity, njobs=njobs):
            calculations.append(calculated_viscosity)
            ui.communicate()

    return calculated_viscosity












