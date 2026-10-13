from typing import Self, TypeVar

from os import PathLike
from tomllib import load as load_toml
from dataclasses import dataclass, fields

from viscofit.utils import cast, mirror_signature

DEFAULT_SETTINGS_FILEPATH: str = r'.config.toml'

T = TypeVar('T')


@dataclass(slots=True, frozen=True)
class Settings:
    root_folder: str
    playmol_command: str
    lammps_commands: str | list[str]
    njobs: int | None
    
    @classmethod
    def from_file(cls, filepath: PathLike, /) -> Self:
        raise NotImplementedError

@mirror_signature(Settings)
def partial_configurations(**kwargs) -> Settings:
    full_kwargs = {
        field.name: None for field in fields(Settings)
    }
    full_kwargs.update(kwargs)

    new = Settings(**full_kwargs)
    return new




