from typing import (
    Mapping, 
    Sequence, 
    Hashable, 
    TypeAlias, 
    Final,
    Annotated, 
    Optional,
    Iterable
)

from os import PathLike
from pathlib import Path
from dataclasses import dataclass

import polars as pl
import scipy.optimize as opt
import scipy.integrate as spint
import numpy as np
import numpy.typing as npt

from viscofit.protocols import IdentityHashing
from viscofit.datahub import (
    write_text,
    get_molecule_filepath,
    parse_playmol_start_box_template,
    parse_coefficients_template,
    parse_lammps_viscosity_template
    
)
from viscofit.utils import (
    copy_files, 
    setup_dir,
    filename,
    ensure_file_exists, 
    execute_script, 
    enumerate_unique,
    join_as_text,
    search_files,
    measure,
    sample_with_replacement
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
    nvt_output_data: str

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
class SimulationAssets:
    viscosity_average: float
    viscosity_uncertainty: float
    viscosity_bootstrap_values: npt.NDArray

@dataclass(slots=True, frozen=True)
class SimulationDetails: 
    viscosity_average_curves: list[npt.NDArray]
    viscosity_average_curve: npt.NDArray
    viscosity_standard_curve: npt.NDArray


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
        return self.data_folder.joinpath(self.filenames.nvt_output_data)

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



def log_b(x: npt.ArrayLike, a: float, b: float, /) -> npt.NDArray:
    x = np.asarray(x)
    return a + b * np.log(x)

def double_exponential(x: npt.ArrayLike, A: float, alpha: float, tal_01: float, tal_02: float, /) -> npt.NDArray:
    x = np.asarray(x)
    return A * alpha * tal_01 * ( 1 - np.exp(-x / tal_01) ) + A * (1 - alpha) * tal_02 * ( 1 - np.exp(-x / tal_02) )

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

def autocorrelation_function(series: npt.ArrayLike, /) -> npt.NDArray:
    series = np.asarray(series)
    series = series - np.mean(series)

    series_fft = np.fft.fft(series)
    series_fft_pow = np.abs(series_fft) ** 2 / len(series_fft)
    series_ifft = np.fft.ifft(series_fft_pow)
    
    autocorrelation = series_ifft.real
    return autocorrelation.ravel()

def estimate_viscosity_value(curves: Iterable[npt.ArrayLike], /) -> float:
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
        method='trf',
        x_scale='jac'
    )

    viscosity = A * alpha * tal_01 + A * (1 - alpha) * tal_02
    return float(viscosity)


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

def run_simulation(data: Simulation, lammps_command: str, /, new_terminal: bool=False, structure_outputs: bool=False) -> None:
    if not data.simulation_done:
        setup_dir(data.lammps_folder, clear=True, create=True)
        execute_script(
            data.lammps_input_filepath,
            lammps_command,
            new_terminal=new_terminal
        )
    if structure_outputs:
        structure_simulation_outputs(data)

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
        }).with_columns(
            ( (pl.col('PXX_ATM') - pl.col('PYY_ATM')) / 2 ).alias('PXX_YY_ATM'),
            ( (pl.col('PYY_ATM') - pl.col('PZZ_ATM')) / 2 ).alias('PYY_ZZ_ATM'),
            ( (pl.col('PZZ_ATM') - pl.col('PXX_ATM')) / 2 ).alias('PZZ_XX_ATM'),
        ).with_columns(
            pl.lit(index).alias('TRAJECTORY_ID'),
            pl.lit(filepath.stem).alias('TRAJECTORY_NAME'),
        ).with_columns(
            (pl.col('TIME_FS') * FEMTOSECOND_TO_SECOND).alias('TIME_SECONDS'),
            (pl.col('TIMESTEP_FS') * FEMTOSECOND_TO_SECOND).alias('TIMESTEP_SECONDS'),
            (pl.col('VOLUME_ANGSTROM3') * ANGSTROM_TO_METER**3).alias('VOLUME_M3'),
            (pl.col('PXY_ATM') * ATM_TO_PA).alias('PXY_PASCAL'),
            (pl.col('PYZ_ATM') * ATM_TO_PA).alias('PYZ_PASCAL'),
            (pl.col('PXZ_ATM') * ATM_TO_PA).alias('PXZ_PASCAL'),
            (pl.col('PXX_ATM') * ATM_TO_PA).alias('PXX_PASCAL'),
            (pl.col('PYY_ATM') * ATM_TO_PA).alias('PYY_PASCAL'),
            (pl.col('PZZ_ATM') * ATM_TO_PA).alias('PZZ_PASCAL'),
            (pl.col('PXX_YY_ATM') * ATM_TO_PA).alias('PXX_YY_PASCAL'),
            (pl.col('PYY_ZZ_ATM') * ATM_TO_PA).alias('PYY_ZZ_PASCAL'),
            (pl.col('PZZ_XX_ATM') * ATM_TO_PA).alias('PZZ_XX_PASCAL'),
        )
        dataframes.append(dataframe)

    if not dataframes:
        raise ValueError(f'No LAMMPS output files found in {data.lammps_folder}.')

    final_dataframe = pl.concat(dataframes, how='vertical')
    final_dataframe.sink_parquet(data.nvt_data_output_filepath)

def calculate_simulation_details(data: Simulation, /, max_steps: Optional[int]=None) -> SimulationDetails:
    dataframe = pl.scan_parquet(data.nvt_data_output_filepath)
    dataframe = dataframe.cast({
        'STEP':                 pl.Int64,
        'VOLUME_M3':            pl.Float64,
        'TEMPERATURE_KELVIN':   pl.Float64,
        'PXX_PASCAL':           pl.Float64,
        'PYY_PASCAL':           pl.Float64,
        'PZZ_PASCAL':           pl.Float64,
        'PXX_YY_PASCAL':        pl.Float64,
        'PYY_ZZ_PASCAL':        pl.Float64,
        'PZZ_XX_PASCAL':        pl.Float64,
        'TIMESTEP_SECONDS':     pl.Float64,
    })
    if max_steps is not None:
        dataframe = dataframe.filter(
            pl.col('STEP') <= pl.lit(max_steps)
        )
    dataframe = dataframe.sort('TRAJECTORY_ID', 'STEP')
    dataframe = dataframe.collect(engine='auto')


    viscosity_average_curves: list[npt.NDArray] = []

    for (_, trajectory) in dataframe.group_by('TRAJECTORY_ID'):

        pxx         = trajectory.get_column('PXX_PASCAL').to_numpy()
        pyy         = trajectory.get_column('PYY_PASCAL').to_numpy()
        pzz         = trajectory.get_column('PZZ_PASCAL').to_numpy()
        pxx_yy      = trajectory.get_column('PXX_YY_PASCAL').to_numpy()
        pyy_zz      = trajectory.get_column('PYY_ZZ_PASCAL').to_numpy()
        pzz_xx      = trajectory.get_column('PZZ_XX_PASCAL').to_numpy()
        volume      = trajectory.get_column('VOLUME_M3').to_numpy()
        timestep    = trajectory.get_column('TIMESTEP_SECONDS').to_numpy()
        temperature = trajectory.get_column('TEMPERATURE_KELVIN').to_numpy()

        volume = np.mean(volume)    
        timestep = np.mean(timestep)
        temperature = np.mean(temperature)
        
        green_kubo_constant = volume / (BOLTZMANN_CONSTANT * temperature)

        autocorrelation = np.array([
            autocorrelation_function(pxx),
            autocorrelation_function(pyy),
            autocorrelation_function(pzz),
            autocorrelation_function(pxx_yy),
            autocorrelation_function(pyy_zz),
            autocorrelation_function(pzz_xx)
        ])
        average_autocorrelation = np.mean(autocorrelation, axis=0)

        viscosity_curve = (
            PA_S_TO_CENTIPOISE * 
            green_kubo_constant * spint.cumulative_trapezoid(average_autocorrelation, initial=0.0) * timestep
        )
        viscosity_average_curves.append(viscosity_curve)

    viscosity_average_curve = np.mean(viscosity_average_curves, axis=0)
    viscosity_uncertainty_curve = np.std(viscosity_average_curves, axis=0, ddof=1)

    details = SimulationDetails(
        viscosity_average_curves=viscosity_average_curves,
        viscosity_average_curve=viscosity_average_curve,
        viscosity_uncertainty_curve=viscosity_uncertainty_curve
    )
    return details

def calculate_simulation_assets(data: Simulation, /, max_steps: Optional[int]=None, bootstraps: Optional[int]=None) -> SimulationAssets:
    if bootstraps is None:
        bootstraps = 1000

    details = calculate_simulation_details(data, max_steps=max_steps)
    viscosity_bootstrap_values: list[float] = []
    
    for _ in range(bootstraps + 1):
        viscosity_bootstrap_curves = sample_with_replacement(details.viscosity_average_curves)
        viscosity_bootstrap_values.append(
            estimate_viscosity_value(viscosity_bootstrap_curves)
        )

    viscosity_bootstrap_values = np.asarray(viscosity_bootstrap_values, dtype=np.float64)
    viscosity_average, viscosity_uncertainty = measure(viscosity_bootstrap_values)

    assets = SimulationAssets(
        viscosity_average=viscosity_average,
        viscosity_uncertainty=viscosity_uncertainty,
        viscosity_bootstrap_values=viscosity_bootstrap_values
    )
    return assets
