from typing import (
    Any, 
    Self,
    TypeVar,
    Callable,
    Iterator, 
    TypeGuard,
    Optional,
    final
)
from enum import StrEnum, auto
from os import PathLike
from pathlib import Path
from string import Template

import polars as pl

from viscofit.utils import (
    mro,
    cache,
    search_files, 
    map_filenames,
    split_selecting,
)
from viscofit.tables import (
    select_renaming,
    apply_table_schema,
    select_casting, 
    cast_columns,
    is_filled,
    is_not_filled,
    to_lowercase,
    ensure_not_partially_filled,
    transform_headers
)

FOLDER_MOLECULES:                   str = r'data/molecules'

FILEPATH_EXPERIMENTS:               str = r'data/experiments.xlsx'
FILEPATH_SYSTEMS:                   str = r'data/systems.xlsx'
FILEPATH_COEFFS:                    str = r'data/coeffs.xlsx'
FILEPATH_ELECTROLYTES_META:         str = r'data/electrolytes_meta.xlsx'
FILEPATH_COEFFS_TEMPLATE:           str = r'data/.template.coeffs.mds'
FILEPATH_PLAYMOL_BOX_TEMPLATE:      str = r'data/.template.start_box.mds'
FILEPATH_LAMMPS_VISCOSITY_TEMPLATE: str = r'data/.template.viscosity.mds'
FILEPATH_PURE_WATER_RESULT:         str = r'data/outputs/pure_water_simulations.xlsx'

TYPES_MAP_POLARS: dict[type, pl.DataType] = {
    str:    pl.String,
    int:    pl.Int64,
    float:  pl.Float64,
    bool:   pl.Boolean
}

TF = TypeVar('TF', bound=Callable)

@final
class ColumnDef(str):
    def __new__(cls, name: str, dtype: type, /) -> Self:
        instance = super().__new__(cls, name)
        instance._datatype = dtype
        return instance

    @property
    def dtype(self) -> type:
        return self._datatype
    
    def __reduce__(self) -> tuple[Callable, tuple]:
        args = (self,)
        return (str, args)

class TableSchema:

    @classmethod
    def columns(cls) -> Iterator[ColumnDef]:
        for base in mro(cls):
            values = vars(base).values()
            yield from filter(is_column_definition, values)

    @classmethod
    def to_polars_schema(cls) -> dict[str, pl.DataType]:
        output = {
            str(column): TYPES_MAP_POLARS[column.dtype] for column in cls.columns()
        }
        return output


class CoeffRangeSchema(TableSchema):
    ATOM_TYPE_1 = ColumnDef('ATOM_TYPE_1', str)
    ATOM_TYPE_2 = ColumnDef('ATOM_TYPE_2', str)
    SIGMA_MIN   = ColumnDef('SIGMA_MIN', float)
    SIGMA_MAX   = ColumnDef('SIGMA_MAX', float)
    EPSILON_MIN = ColumnDef('EPSILON_MIN', float)
    EPSILON_MAX = ColumnDef('EPSILON_MAX', float)

class CoeffSchema(TableSchema):
    ATOM_TYPE_1 = ColumnDef('ATOM_TYPE_1', str)
    ATOM_TYPE_2 = ColumnDef('ATOM_TYPE_2', str)
    SIGMA       = ColumnDef('SIGMA', float)
    EPSILON     = ColumnDef('EPSILON', float)
    
class ElectrolyteSchema(TableSchema):
    ELECTROLYTE             = ColumnDef('ELECTROLYTE', str)
    ELECTROLYTE_MOLAR_MASS  = ColumnDef('ELECTROLYTE_MOLAR_MASS', float)

    CATION                  = ColumnDef('CATION', str)
    CATION_ESTEQ            = ColumnDef('CATION_ESTEQ', int)
    CATION_CHARGE           = ColumnDef('CATION_CHARGE', int)
    CATION_MOLAR_MASS       = ColumnDef('CATION_MOLAR_MASS', float)

    ANION                   = ColumnDef('ANION', str)
    ANION_ESTEQ             = ColumnDef('ANION_ESTEQ', int)
    ANION_CHARGE            = ColumnDef('ANION_CHARGE', int)
    ANION_MOLAR_MASS        = ColumnDef('ANION_MOLAR_MASS', float)

class SystemSchema(ElectrolyteSchema, TableSchema):
    REFERENCE   = ColumnDef('REFERENCE', str)
    NOTES       = ColumnDef('NOTES', str)
    MOLALITY    = ColumnDef('MOLALITY_MOL_KG', float)
    PRESSURE    = ColumnDef('PRESSURE_ATM', float)
    TEMPERATURE = ColumnDef('TEMPERATURE_KELVIN', float)
    VISCOSITY   = ColumnDef('VISCOSITY_CP', float)


class DataContext(StrEnum):
    PURE_WATER_SIMULATIONS  = 'pure-water-simulations'
    FIT_PARAMS              = 'fit_params'
    PRELIMINAR_TEST         = 'preliminar_tests'
    EVALUATE_MADRID_2019    = 'evaluate_madrid_2019'



def is_column_definition(data: Any, /) -> TypeGuard[ColumnDef]:
    return isinstance(data, ColumnDef)

def match_schema(schema: TableSchema, /, pause: bool=False) -> Callable[[TF], TF]:
    dict_schema = schema.to_polars_schema()
    def decorator(func: Callable[..., pl.DataFrame]):
        def wrapper(*args, **kwargs):
            dataframe = func(*args, **kwargs)
            if not pause:
                dataframe = select_casting(dataframe, dict_schema)
                dataframe = dataframe.match_to_schema(dict_schema, missing_columns='raise')
            return dataframe
        return wrapper
    return decorator

def write_text(filepath: PathLike, text: str, /) -> None:
    return Path(filepath).write_text(text, encoding='utf-8')

def read_excel_file(filepath: PathLike, /, sheet_name: Optional[str]=None) -> pl.DataFrame:
    return pl.read_excel(filepath, sheet_name=sheet_name, engine='calamine', infer_schema_length=0)



@cache
def read_text(filepath: PathLike, /) -> str:
    return Path(filepath).read_text(encoding='utf-8')

@cache
@match_schema(ElectrolyteSchema)
def load_electrolytes_meta() -> pl.DataFrame:
    ions = read_excel_file(FILEPATH_ELECTROLYTES_META, sheet_name='IONS') 
    electrolytes = read_excel_file(FILEPATH_ELECTROLYTES_META, sheet_name='ELECTROLYTES')

    ions = transform_headers(ions, str.strip)
    electrolytes = transform_headers(electrolytes, str.strip)
    
    electrolytes = apply_table_schema(electrolytes, {
        'ELECTROLYTE':      (ElectrolyteSchema.ELECTROLYTE, pl.String),
        'CATION':           (ElectrolyteSchema.CATION, pl.String),
        'ANION':            (ElectrolyteSchema.ANION, pl.String),
        'CATION_ESTEQ':     (ElectrolyteSchema.CATION_ESTEQ, pl.Int64),
        'ANION_ESTEQ':      (ElectrolyteSchema.ANION_ESTEQ, pl.Int64)
    })
    ions = apply_table_schema(ions, {
        'ION':              ('ION', pl.String),
        'MOLAR_MASS_G_MOL': ('MOLAR_MASS_G_MOL', pl.Float64),
        'CHARGE':           ('CHARGE', pl.Int64)
    })

    electrolytes = electrolytes.with_columns(
        pl.col('CATION', 'ANION', 'ELECTROLYTE').str.strip_chars()
    )
    ions = ions.with_columns(
        pl.col('ION').str.strip_chars()
    )
    electrolytes = electrolytes.join(
        ions,
        left_on='CATION',
        right_on='ION',
        validate='m:1'
    ).rename({
        'MOLAR_MASS_G_MOL': ElectrolyteSchema.CATION_MOLAR_MASS,
        'CHARGE':           ElectrolyteSchema.CATION_CHARGE
    })
    electrolytes = electrolytes.join(
        ions,
        left_on='ANION',
        right_on='ION',
        validate='m:1'
    ).rename({
        'MOLAR_MASS_G_MOL': ElectrolyteSchema.ANION_MOLAR_MASS,
        'CHARGE':           ElectrolyteSchema.ANION_CHARGE
    })
    electrolytes = electrolytes.with_columns(
        (
            pl.col(ElectrolyteSchema.ANION_MOLAR_MASS) * pl.col(ElectrolyteSchema.ANION_ESTEQ) +
            pl.col(ElectrolyteSchema.CATION_MOLAR_MASS) * pl.col(ElectrolyteSchema.CATION_ESTEQ)
        ).alias(
            ElectrolyteSchema.ELECTROLYTE_MOLAR_MASS
        )
    )
    return electrolytes
   
@cache
@match_schema(CoeffRangeSchema)
def load_coeffs_range(source: DataContext, /) -> pl.DataFrame: 
    dataframe = read_excel_file(FILEPATH_EXPERIMENTS, sheet_name='PAIR-COEFF-RANGES')

    dataframe = transform_headers(dataframe, str.strip)
    dataframe = apply_table_schema(dataframe, {
        'EXPERIMENT':   ('EXPERIMENT',                  pl.String),
        'ATOM_TYPE_01': (CoeffRangeSchema.ATOM_TYPE_1,  pl.String),
        'ATOM_TYPE_02': (CoeffRangeSchema.ATOM_TYPE_2,  pl.String),
        'EPSILON_MIN':  (CoeffRangeSchema.EPSILON_MIN,  pl.Float64),
        'EPSILON_MAX':  (CoeffRangeSchema.EPSILON_MAX,  pl.Float64),
        'SIGMA_MIN':    (CoeffRangeSchema.SIGMA_MIN,    pl.Float64),
        'SIGMA_MAX':    (CoeffRangeSchema.SIGMA_MAX,    pl.Float64)
    })
    dataframe = dataframe.with_columns(
        pl.col('EXPERIMENT', CoeffRangeSchema.ATOM_TYPE_1, CoeffRangeSchema.ATOM_TYPE_2).str.strip_chars()
    )
    dataframe = dataframe.filter(
        pl.col('EXPERIMENT') == pl.lit(source)
    )
    return dataframe
    
@cache
@match_schema(CoeffSchema)
def load_coeffs(source: DataContext, /) -> pl.DataFrame:  
    dataframe = read_excel_file(FILEPATH_EXPERIMENTS, sheet_name='PAIR-COEFF-VALUES')

    dataframe = transform_headers(dataframe, str.strip)
    dataframe = apply_table_schema(dataframe, {
        'EXPERIMENT':       ('EXPERIMENT',              pl.String),
        'ATOM_TYPE_01':     (CoeffSchema.ATOM_TYPE_1,   pl.String),
        'ATOM_TYPE_02':     (CoeffSchema.ATOM_TYPE_2,   pl.String),
        'EPSILON_VALUE':    (CoeffSchema.EPSILON,       pl.Float64),
        'SIGMA_VALUE':      (CoeffSchema.SIGMA,         pl.Float64)
    })
    dataframe = dataframe.with_columns(
        pl.col('EXPERIMENT', CoeffSchema.ATOM_TYPE_1, CoeffSchema.ATOM_TYPE_2).str.strip_chars()
    )
    dataframe = dataframe.filter(
        pl.col('EXPERIMENT') == pl.lit(source)
    )
    return dataframe

@cache
@match_schema(SystemSchema)
def load_systems(source: DataContext, /) -> pl.DataFrame:
    dataframe = read_excel_file(FILEPATH_EXPERIMENTS, sheet_name='SYSTEMS')
    
    dataframe = transform_headers(dataframe, str.strip)
    dataframe = apply_table_schema(dataframe, {
        'EXPERIMENT':           ('EXPERIMENT',              pl.String),
        'ELECTROLYTE':          (SystemSchema.ELECTROLYTE,  pl.String),
        'MOLALITY_MOL_KG':      (SystemSchema.MOLALITY,     pl.Float64),
        'TEMPERATURE_KELVIN':   (SystemSchema.TEMPERATURE,  pl.Float64),
        'PRESSURE_ATM':         (SystemSchema.PRESSURE,     pl.Float64),
        'VISCOSITY_CP':         (SystemSchema.VISCOSITY,    pl.Float64),
        'REFERENCE':            (SystemSchema.REFERENCE,    pl.String)
    })
    dataframe = dataframe.with_columns(
        pl.col('EXPERIMENT', SystemSchema.ELECTROLYTE, SystemSchema.REFERENCE).str.strip_chars()
    )
    dataframe = dataframe.join(
        load_electrolytes_meta(),
        on=SystemSchema.ELECTROLYTE,
        validate='m:1'
    )
    dataframe = dataframe.filter(
        pl.col('EXPERIMENT') == pl.lit(source)
    )
    return dataframe




@cache
def map_molecular_files(no_extension: bool=False) -> dict[str, Path]:
    available_files = search_files(FOLDER_MOLECULES, r'*.mol', recursive=True)
    return map_filenames(*available_files, no_extension=no_extension)

def get_molecule_filepath(molecule_name: str, /) -> Path:
    molecule_files = map_molecular_files(no_extension=True)
    molecule_filepath = molecule_files.get(molecule_name)

    if molecule_filepath is None:
        molecule_files = sorted(molecule_files)
        raise FileExistsError(
            f'Cannot found a molecule file (.mol) for {molecule_name!r}. Available options are: {molecule_files}'
        )
    return molecule_filepath

def extract_atom_types(filepath: PathLike, /, raise_empty: bool=False) -> tuple[str, ...]:
    lines = (
        line.strip() for line in read_text(filepath).splitlines()
    )
    
    flag = 'atom_type'
    atom_types: list[str] = []

    for line_index, line in enumerate(lines, start=1):
        if line.startswith(flag):
            try:
                atom_type = split_selecting(line, 1)
            except IndexError:
                raise RuntimeError(
                    f'Cannot extract atom-type value from line {line_index}.\n'
                    f'Please, fix this file: {filepath!r}.\n'
                    f'The line: {line!r}'
                )
            atom_types.append(atom_type)

    if raise_empty and not atom_types:
        raise ValueError(
            f'No atom-type found in {filepath!r}. '
            f'Expected line(s) starting with the literal flag {flag!r}.'
        )
    return tuple(atom_types)


def parse_coefficients_template(atom_types: str, pair_coeffs: str, /) -> str:
    values = {
        'INPUT_ATOM_TYPES_DESCRIPTION': atom_types,
        'INPUT_PAIR_COEFFS': pair_coeffs
    }
    template_text = read_text(FILEPATH_COEFFS_TEMPLATE)
    return Template(template_text).safe_substitute(values)

def parse_playmol_start_box_template(includes: str, box_dimensions: str, packs: str, lammps_output_file: str, xyz_output_file: str, /) -> str:
    values = {
        'INPUT_MOLECULE_FILENAMES': includes,
        'INPUT_BOX_SIZE': box_dimensions,
        'INPUT_PACKS': packs,
        'INPUT_DUMP_LAMMPS_FILENAME': lammps_output_file,
        'INPUT_DUMP_XYZ_FILENAME': xyz_output_file
    }
    template_text = read_text(FILEPATH_PLAYMOL_BOX_TEMPLATE)
    return Template(template_text).safe_substitute(values)

def parse_lammps_viscosity_template(temperature: str, pressure: str, start_box_file: str, coeffs_file: str, /) -> str:
    values = {
        'INPUT_TEMPERATURE': temperature,
        'INPUT_PRESSURE': pressure,
        'INPUT_START_BOX_FILE': start_box_file,
        'INPUT_COEFFS_FILE': coeffs_file
    }
    template_text = read_text(FILEPATH_LAMMPS_VISCOSITY_TEMPLATE)
    return Template(template_text).safe_substitute(values)