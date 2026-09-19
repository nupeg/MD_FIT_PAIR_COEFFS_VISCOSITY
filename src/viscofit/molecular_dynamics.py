from typing import (
    Sequence, 
    TypeAlias, 
    NamedTuple, 
    Mapping, 
    Iterable,
    Annotated,
    Final
)

from os import PathLike
from pathlib import Path
from itertools import combinations, combinations_with_replacement
from dataclasses import dataclass

import polars as pl

import scipy.optimize as opt

import numpy as np
import numpy.typing as npt

from viscofit.utils import (
    keys, 
    values,
    measure,
    filename,
    setup_dir, 
    copy_files,
    parse_cmd,
    search_files, 
    join_as_text,
    execute_script,
    sorted_tuple,
    enumerate_unique,
    ensure_file_exists,
    ensure_not_duplicates,
    sample_with_replacement
)
from viscofit.datahub import (
    parse_coefficients_template,
    parse_lammps_viscosity_template,
    parse_playmol_start_box_template    
)
from viscofit.tables import apply_table_schema
from viscofit.protocols import MixtureRule

BOLTZMAN_CONSTANT: Annotated[float, 'J/K'] = 1.380649e-23

ATM_TO_PA:              Final[float] = 101325
ANGSTROM_TO_METER:      Final[float] = 1e-10
FEMTOSECOND_TO_SECOND:  Final[float] = 1e-15
POISE_TO_CENTIPOISE:    Final[float] = 100

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

    @property
    def is_self(self) -> bool:
        return self.atom_type_01 == self.atom_type_02

    @property
    def is_cross(self) -> bool:
        return not self.is_self

    @property
    def atom_types(self) -> tuple[str, str]:
        return sorted_tuple((self.atom_type_01, self.atom_type_02))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PairCoeff) and self.atom_types == other.atom_types

    def __hash__(self)-> int:
        return hash(self.atom_types)

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

    @property
    def atom_types(self) -> list[str]:
        return sorted({atom_type for molecule in self.compositions for atom_type in molecule.atom_types})

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

@dataclass(slots=True, frozen=True)
class ViscosityAssets:
    folder_path:                Path
    trajectory_files:           list[Path]
    trajectory_count:           int
    viscosity_curves:           Annotated[npt.NDArray, 'cP']
    average_viscosity_curve:    Annotated[npt.NDArray, 'cP']
    standard_viscosity_curve:   Annotated[npt.NDArray, 'cP']
    viscosity_average:          Annotated[float, 'cP']
    viscosity_uncertainty:      Annotated[float, 'cP']

@dataclass(slots=True, frozen=True)
class TrajectoryData:
    '''
    Trajectory data expressed entirely in SI units.

    Attributes
    ----------
    pxy, pyz, pxz : array-like
        Shear stress components [Pa].
    pxx_yy, pyy_zz, pzz_xx : array-like
        Normal stress differences [Pa].
    timestep : float
        Time interval between consecutive observations [s].
    volume : float
        Simulation box volume [m³].
    temperature : float
        Temperature [K].
    '''
    pxy:         Annotated[npt.NDArray, 'Pa']
    pyz:         Annotated[npt.NDArray, 'Pa']
    pxz:         Annotated[npt.NDArray, 'Pa']
    pxx_yy:      Annotated[npt.NDArray, 'Pa']
    pyy_zz:      Annotated[npt.NDArray, 'Pa']
    pzz_xx:      Annotated[npt.NDArray, 'Pa']
    timestep:    Annotated[float, 's']
    volume:      Annotated[float, 'm³']
    temperature: Annotated[float, 'K']



def lorentz_berthelot_epsilon(atom_type_01_epsilon: float, atom_type_02_epsilon: float, /) -> float: 
    return pow(atom_type_01_epsilon * atom_type_02_epsilon, 0.5)

def lorentz_berthelot_sigma(atom_type_01_sigma: float, atom_type_02_sigma: float, /) -> float: 
    return (atom_type_01_sigma + atom_type_02_sigma) / 2

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
        pair_coeffs_text
    )
    Path(filepath).write_text(content)

def write_simulation_file(filepath: PathLike, temperature: float, pressure: float, start_box_file: str, coeffs_file: str) -> None: 
    content = parse_lammps_viscosity_template(
        temperature,
        pressure,
        start_box_file,
        coeffs_file
    )
    Path(filepath).write_text(content)

def autocorrelation_function(series: npt.ArrayLike, /) -> npt.NDArray:
    series = np.asarray(series)
    series = series - np.mean(series)

    series_fft = np.fft.fft(series)
    series_fft_pow = np.abs(series_fft) ** 2 / len(series_fft)
    series_ifft = np.fft.ifft(series_fft_pow)
    
    autocorrelation = series_ifft.real
    return autocorrelation.ravel()

def read_trajectory_data(filepath: PathLike, /) -> TrajectoryData:
    data = pl.read_csv(filepath)
    data = apply_table_schema(data, {
        'pxy':  ('pxy',  pl.Float64),
        'pyz':  ('pyz',  pl.Float64),
        'pxz':  ('pxz',  pl.Float64),
        'pxx':  ('pxx',  pl.Float64),
        'pyy':  ('pyy',  pl.Float64),
        'pzz':  ('pzz',  pl.Float64),
        'vol':  ('vol',  pl.Float64),
        'temp': ('temp', pl.Float64),
        'dt':   ('dt',   pl.Float64)
    }).with_columns(
        pl.col('pxy', 'pyz', 'pxz', 'pxx', 'pyy', 'pzz').mul(ATM_TO_PA),
        pl.col('vol').mul(ANGSTROM_TO_METER**3),
        pl.col('dt').mul(FEMTOSECOND_TO_SECOND)
    ).with_columns(
        ( (pl.col('pxx') - pl.col('pyy')) / 2 ).alias('pxx_yy'),
        ( (pl.col('pyy') - pl.col('pzz')) / 2 ).alias('pyy_zz'),
        ( (pl.col('pzz') - pl.col('pxx')) / 2 ).alias('pzz_xx')
    )

    pxy     = data['pxy']
    pyz     = data['pyz']
    pxz     = data['pxz']
    pxx_yy  = data['pxx_yy']
    pyy_zz  = data['pyy_zz']
    pzz_xx  = data['pzz_xx']

    volume = np.mean(data['vol'])
    timestep = np.mean(data['dt'])
    temperature = np.mean(data['temp'])

    data = TrajectoryData(
        pxy=pxy,
        pyz=pyz,
        pxz=pxz,
        pxx_yy=pxx_yy,
        pyy_zz=pyy_zz,
        pzz_xx=pzz_xx,
        timestep=timestep,
        volume=volume,
        temperature=temperature
    )
    return data

def log_b(x: npt.ArrayLike, a: float, b: float, /) -> npt.NDArray:
    x = np.asarray(x)
    return a + b * np.log(x)

def double_exponential(x: npt.ArrayLike, A: float, alpha: float, tal_01: float, tal_02: float, /) -> npt.NDArray:
    x = np.asarray(x)
    return A * alpha * tal_01 * ( 1 - np.exp(-x / tal_01) ) + A * (1 - alpha) * tal_02 * ( 1 - np.exp(-x / tal_02) )

def estimate_viscosity(curves: Iterable[npt.ArrayLike], /) -> float:
    curves = [
        np.asarray(curve) for curve in curves
    ]
    average_curve = np.mean(curves, axis=0)
    standard_curve = np.std(curves, axis=0, ddof=1)
    
    steps = np.arange(1, standard_curve.size + 1)

    (_, b), _ = opt.curve_fit(log_b, steps, standard_curve)
    
    sigma = steps**(b/2)

    bounds = [
        [0.0, 0.0, 0.0, 0.0], 
        [np.inf, 1.0, np.inf, np.inf]
    ]

    (A, alpha, tal_01, tal_02), _ = opt.curve_fit(
        double_exponential, 
        steps, 
        average_curve, 
        sigma=sigma, 
        bounds=bounds, 
        method='trf'
    )

    viscosity = A * alpha * tal_01 + A * (1 - alpha) * tal_02
    return float(viscosity)

def atom_type_pairs(atom_types: Iterable[str], /) -> Iterable[tuple[str, str]]:
    return combinations_with_replacement(sorted(atom_types), r=2)



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

def calculate_viscosity_assets(simulation_folder_path: PathLike, /) -> ViscosityAssets:
    simulation_folder_path = Path(simulation_folder_path)

    trajectory_files = search_files(simulation_folder_path, r'*.RUN')
    trajectory_count = len(trajectory_files)

    if trajectory_count < 1:
        raise ValueError(f'No trajectory files were found in the folder: {simulation_folder_path!s}')

    viscosity_curves: list[npt.NDArray] = []

    for trajectory in map(read_trajectory_data, trajectory_files):
        autocorrelations = np.array([
            autocorrelation_function(trajectory.pxy),
            autocorrelation_function(trajectory.pyz),
            autocorrelation_function(trajectory.pxz),
            autocorrelation_function(trajectory.pxx_yy),
            autocorrelation_function(trajectory.pyy_zz),
            autocorrelation_function(trajectory.pzz_xx),
        ])
        average_autocorrelation: Annotated[npt.NDArray, 'Pa²'] = np.mean(autocorrelations, axis=0)
        green_kubo_constant: Annotated[float, 'Pa⁻¹'] = trajectory.volume / (BOLTZMAN_CONSTANT * trajectory.temperature)

        viscosity_curve_cP = POISE_TO_CENTIPOISE * green_kubo_constant * np.trapezoid(average_autocorrelation) * trajectory.timestep
        viscosity_curves.append(viscosity_curve_cP)
    
    # NOTE: Bootstrap
    viscosity_estimates: list[float] = []
    
    for _ in range(1001):
        samples = sample_with_replacement(viscosity_curves)
        
        viscosity_estimate = estimate_viscosity(samples)
        viscosity_estimates.append(viscosity_estimate)

    viscosity_average, viscosity_uncertainty = measure(viscosity_estimates)

    average_viscosity_curve = np.mean(viscosity_curves, axis=0)
    standard_viscosity_curve = np.std(viscosity_curves, axis=0, ddof=1)
    
    assets = ViscosityAssets(
        folder_path=simulation_folder_path,
        trajectory_files=trajectory_files,
        trajectory_count=trajectory_count,
        viscosity_curves=viscosity_curves,
        average_viscosity_curve=average_viscosity_curve,
        standard_viscosity_curve=standard_viscosity_curve,
        viscosity_average=viscosity_average,
        viscosity_uncertainty=viscosity_uncertainty
    )
    return assets 

def validate_simulation_setups(simulations: Sequence[SimulationSetup], /) -> None: 
    ensure_not_duplicates(setup.folder_path for setup in simulations)

def validate_mixtures_coeffs(systems: Sequence[MixtureData], coeffs: Sequence[PairCoeff], /) -> None: 
    required_pairs = {
        pair
        for mixture in systems
        for pair in atom_type_pairs(mixture.atom_types)
    
    }
    available_pairs = {
        sorted_tuple(coeff.atom_types)
        for coeff in coeffs
    }
    missing_pairs = required_pairs.difference(available_pairs)

    if missing_pairs:
        missing_pairs = sorted(missing_pairs)
        raise ValueError(
            f'Missing pair coefficients: {missing_pairs}'
        )
    
def complete_pair_coeffs(coeffs: list[PairCoeff], /, sigma_rule: MixtureRule, epsilon_rule: MixtureRule) -> list[PairCoeff]: 
    full_atom_types:    set[str] = set()
    self_atom_types:    set[str] = set()
    self_coeffs:        set[PairCoeff] = set()
    cross_coeffs:       set[PairCoeff] = set()

    for coeff in coeffs:
        full_atom_types.update(coeff.atom_types)

        if coeff.is_self:
            self_atom_types.update(coeff.atom_types)
            self_coeffs.add(coeff)
            continue
        
        cross_coeffs.add(coeff)

    missing_self_atom_types = full_atom_types.difference(self_atom_types)

    if missing_self_atom_types:
        raise ValueError(
            f'Missing self coefficients for: {missing_self_atom_types}'
        )

    for coeff_01, coeff_02 in combinations(self_coeffs, r=2):
        cross_coeff = PairCoeff(
            atom_type_01=coeff_01.atom_type_01,
            atom_type_02=coeff_02.atom_type_01,
            sigma=sigma_rule(coeff_01.sigma, coeff_02.sigma),
            epsilon=epsilon_rule(coeff_01.epsilon, coeff_02.epsilon),
        )

        if cross_coeff not in cross_coeffs:
            cross_coeffs.add(cross_coeff)

    return [*self_coeffs, *cross_coeffs]

def create_simulation_setups(
        systems: Sequence[MixtureData],
        coeffs: Sequence[PairCoeff],
        box: BoxDimensions,
        files: SimulationFiles, 
        base_folder: PathLike
    ) -> list[SimulationSetup]: 
    
    base_folder = Path(base_folder)
    
    coeffs_mapping = {
        sorted_tuple(coeff.atom_types): coeff for coeff in coeffs
    }
    simulations = []
    
    for mixture in systems:
        folder_path = base_folder.joinpath(mixture.folder_name)

        mixture_coeffs = [
            coeffs_mapping[pair] for pair in atom_type_pairs(mixture.atom_types)
        ]

        setup = SimulationSetup(
            folder_path=folder_path,
            files=files,
            box=box,
            mixture=mixture,
            coeffs=mixture_coeffs
        )
        simulations.append(setup)
    return simulations











