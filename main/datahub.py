from typing import Any, Self, Iterator, TypeGuard, final
from pathlib import Path

import polars as pl

from core.utils import mro, cache
from core.tables import ensure_not_empty, select_renaming, select_casting

FILEPATH_SYSTEMS:           str = r'data/systems.xlsx'
FILEPATH_MADRID_SIGMA:      str = r'data/madrid2019_sigma.xlsx'
FILEPATH_MADRID_EPSILON:    str = r'data/madrid2019_epsilon.xlsx'

TYPES_MAP_POLARS: dict[type, pl.DataType] = {
    str:    pl.String,
    int:    pl.Int64,
    float:  pl.Float64,
    bool:   pl.Boolean
}


@final
class ColumnDef(str):
    def __new__(cls, name: str, dtype: type, /) -> Self:
        instance = super().__new__(cls, name)
        instance._dtype = dtype
        return instance

    @property
    def dtype(self) -> type:
        return self._dtype
    
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


@final
class CoeffSchema(TableSchema):
    ATOM_TYPE_1 = ColumnDef('ATOM_TYPE_1', str)
    ATOM_TYPE_2 = ColumnDef('ATOM_TYPE_2', str)
    SIGMA       = ColumnDef('SIGMA', float)
    EPSILON     = ColumnDef('EPSILON', float)

@final
class SystemSchema(TableSchema):
    SALT        = ColumnDef('SALT', str)
    MOLALITY    = ColumnDef('MOLALITY_MOL_KG', float)
    PRESSURE    = ColumnDef('PRESSURE_BAR', float)
    TEMPERATURE = ColumnDef('TEMPERATURE_KELVIN', float)



def is_column_definition(data: Any, /) -> TypeGuard[ColumnDef]:
    return isinstance(data, ColumnDef)




@cache
def load_madrid_pair_coeffs() -> pl.DataFrame:
    ...

@cache
def load_systems(allow_empty: bool=True) -> pl.DataFrame:
    schema = SystemSchema.to_polars_schema()
    
    filepath = Path(FILEPATH_SYSTEMS)

    if not filepath.is_file():
        raise FileNotFoundError(f'File {filepath!r} cannot be found. Please, read the documentation.')
    
    dataframe = pl.read_excel(FILEPATH_SYSTEMS, sheet_name='SYSTEMS')

    if not allow_empty:
        ensure_not_empty(dataframe)
    
    dataframe = select_renaming(dataframe, {
        'SALT':                 SystemSchema.SALT,
        'PRESSURE_BAR':         SystemSchema.PRESSURE,
        'MOLALITY_MOL_KG':      SystemSchema.MOLALITY,
        'TEMPERATURE_KELVIN':   SystemSchema.TEMPERATURE
    })
    dataframe = select_casting(dataframe, schema)
    dataframe = dataframe.match_to_schema(schema)
    
    return dataframe
