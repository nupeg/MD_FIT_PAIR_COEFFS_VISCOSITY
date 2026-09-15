from typing import Any, Self, Mapping, TypeVar

from os import PathLike
from tomllib import load as load_toml
from dataclasses import dataclass

from src.viscofit.utils import cast

DEFAULT_SETTINGS_FILEPATH: str = r'.config.toml'

T = TypeVar('T')

def replace_placeholders(data: Mapping[str, Any], /) -> dict[str, Any]:
    datacopy = dict(data)
    for key, value in datacopy.items():
        if isinstance(value, str) and value.strip() in {'', 'none'}:
            value = None
        datacopy[key] = value
    return datacopy            

def get_config(data: Mapping[str, Any], name: str, dtype: type[T], *, nullable: bool = False) -> T | None:
    return cast(data[name], dtype, ignore_null=nullable)

@dataclass(slots=True, frozen=True)
class Settings:
    checkpoint_folder_path: str
    
    @classmethod
    def from_file(cls, filepath: PathLike, /) -> Self:
        ...