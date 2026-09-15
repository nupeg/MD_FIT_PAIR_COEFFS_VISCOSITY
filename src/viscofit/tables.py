from typing import (
    Any,
    Iterable,
    Hashable,
    Mapping,
    Optional,
    TypeAlias,
    Callable
)

import polars as pl

from src.viscofit.utils import type_name, keys, values, coalesce, pipe

PolarsLike:         TypeAlias = Any
ColumnName:         TypeAlias = str
TextTransformer:    TypeAlias = Callable[[str], str]

def transform_dataframe(data: PolarsLike, /) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        return data
    try:
        return pl.DataFrame(data)
    except Exception as error:
        raise ValueError(
            f'Failed to convert object of type {type_name(data)!r} to Polars DataFrame. Details: {error}'
        ) from error

def to_expr(data: ColumnName | pl.Expr, /) -> pl.Expr:
    if isinstance(data, pl.Expr):
        return data
    return pl.col(data)

def list_columns_selection(data: Iterable[ColumnName] | ColumnName, /) -> list[ColumnName]:
    if isinstance(data, str):
        return [data]
    return list(data)

def key_value_table(data: Mapping[Hashable, Any], /, key_column: Optional[ColumnName]=None, value_column: Optional[ColumnName]=None) -> pl.DataFrame:
    key_column = coalesce(key_column, 'key')
    value_column = coalesce(value_column, 'value')
    
    dataframe = pl.DataFrame({
        key_column: keys(data),
        value_column: values(data)
    })
    return dataframe


def ensure_not_partially_filled(data: PolarsLike, subsets: Optional[Iterable[ColumnName] | ColumnName]=None) -> pl.DataFrame:
    dataframe = transform_dataframe(data)

    if subsets is None:
        subsets = dataframe.columns
    subsets = list_columns_selection(subsets)

    partially_filled = dataframe.filter(
        is_partially_filled(*subsets)
    )
    if not partially_filled.is_empty():
        raise ValueError(
            f'Partially filled rows detected.\n\n'
            f'Columns checked cannot be partially filled: {subsets}\n'
            f'Invalid rows: {partially_filled.height}\n\n'
            f'Sample:\n{partially_filled.head(10)}'
        )
    return dataframe
    
def select_renaming(data: PolarsLike, renamer: Mapping[ColumnName, str], /) -> pl.DataFrame:
    return transform_dataframe(data).select(keys(renamer)).rename(renamer)

def select_casting(data: PolarsLike, dtypes: Mapping[ColumnName, pl.DataType], /) -> pl.DataFrame:
    return transform_dataframe(data).select(keys(dtypes)).cast(dtypes)

def ensure_not_empty(data: PolarsLike, /) -> None:
    if transform_dataframe(data).is_empty():
        raise RuntimeError('DataFrame cannot be empty.')

def transform_headers(data: PolarsLike, *transformations: TextTransformer) -> pl.DataFrame:
    dataframe = transform_dataframe(data)
    dataframe.columns = [
        pipe(name, *transformations) for name in map(str, dataframe.columns)
    ]
    return dataframe

def unique_values_from(data: PolarsLike, columns: ColumnName | Iterable[ColumnName], /) -> pl.Series:
    columns = list_columns_selection(columns)
    uniques = (
        transform_dataframe(data)
        .select(columns)
        .unpivot(value_name='VALUE')
        .get_column('VALUE')
        .unique()
    )
    return uniques

def replace_columns(data: PolarsLike, subsets: ColumnName | Iterable[ColumnName], replaces: Mapping[Any, Any], /, strict: bool=True) -> pl.DataFrame:
    replaces = dict(replaces)

    dataframe = transform_dataframe(data)
    subsets = list_columns_selection(subsets)

    replacer = pl.Expr.replace

    if strict:
        replacer = pl.Expr.replace_strict

    exprs = (
        replacer(col, replaces) for col in map(pl.col, subsets)
    )
    try:
        return dataframe.with_columns(exprs)
    except pl.exceptions.InvalidOperationError as error:
        replaces = sorted(replaces)
        raise ValueError(
            f'Strict replacement failed. Some values in columns {subsets} '
            f'were not found in the replacer mapping: {replaces}'
        ) from error

def remove_empty_rows(data: PolarsLike, /, empty_values: Optional[Iterable[Any]]=None) -> pl.DataFrame:
    if empty_values is None:
        empty_values = []
    empty_values = tuple(empty_values)

    is_empty = pl.all_horizontal(
        pl.all().is_null() | pl.all().is_in(empty_values)
    )
    return transform_dataframe(data).filter(~is_empty)



def to_lowercase(column: ColumnName | pl.Expr, /) -> pl.Expr:
    return to_expr(column).str.to_lowercase()

def cast_columns(*columns: ColumnName, dtype: pl.DataType) -> list[pl.Expr]:
    return pl.col(columns).cast(dtype)

def is_filled(column: ColumnName | pl.Expr, /) -> pl.Expr:
    return to_expr(column).is_not_null()

def is_not_filled(column: ColumnName | pl.Expr, /) -> pl.Expr:
    return to_expr(column).is_null()

def is_partially_filled(*columns: ColumnName | pl.Expr) -> pl.Expr:
    cols = [to_expr(col) for col in columns]

    filled = map(is_filled, cols)
    not_filled = map(is_not_filled, cols)

    return pl.any_horizontal(*filled) & pl.any_horizontal(*not_filled)

 