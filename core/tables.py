
from typing import (
    Any,
    Mapping,
    TypeAlias
)

import polars as pl

from core.utils import keys

PolarsLike: TypeAlias = Any
ColumnName: TypeAlias = str

def transform_dataframe(data: PolarsLike, /) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        return data
    try:
        return pl.DataFrame(data)
    except Exception as error:
        raise ValueError('...')
    

def select_renaming(data: PolarsLike, renamer: Mapping[ColumnName, str], /) -> pl.DataFrame:
    return transform_dataframe(data).select(keys(renamer)).rename(renamer)

def select_casting(data: PolarsLike, dtypes: Mapping[ColumnName, pl.DataType], /) -> pl.DataFrame:
    return transform_dataframe(data).select(keys(dtypes)).cast(dtypes)

def ensure_not_empty(data: PolarsLike, /) -> None:
    if transform_dataframe(data).is_empty():
        raise RuntimeError('DataFrame cannot be empty.') 
