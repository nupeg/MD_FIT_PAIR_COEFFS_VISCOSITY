from typing import Any, Self, Mapping, TypeVar

from os import PathLike
from tomllib import load as load_toml
from dataclasses import dataclass, fields

from viscofit.utils import cast, mirror_signature

DEFAULT_SETTINGS_FILEPATH: str = r'.config.toml'

T = TypeVar('T')


@dataclass(slots=True, frozen=True)
class Settings:
    checkpoint_folder_path: str
    njobs: int | None

    n_particles_water: int
    root_folder_path: str

    lammps_cmd: str | list[str]
    playmol_cmd: str

    box: list[float]

    filename_simulation: str
    filename_coeffs: str
    filename_start_box_playmol: str
    filename_start_box_lammps: str
    filename_start_box_xyz: str

    npt_steps: int
    nvt_steps: int
    num_trajectories: int
    
    @classmethod
    def from_file(cls, filepath: PathLike, /) -> Self:
        raise NotImplementedError

def replace_placeholders(data: Mapping[str, Any], /) -> dict[str, Any]:
    datacopy = dict(data)
    for key, value in datacopy.items():
        if isinstance(value, str) and value.strip() in {'', 'none'}:
            value = None
        datacopy[key] = value
    return datacopy            

def get_config(data: Mapping[str, Any], name: str, dtype: type[T], *, nullable: bool = False) -> T | None:
    return cast(data[name], dtype, ignore_null=nullable)

@mirror_signature(Settings)
def partial_configurations(**kwargs) -> Settings:
    full_kwargs = {
        field.name: None for field in fields(Settings)
    }
    full_kwargs.update(kwargs)

    new = Settings(**full_kwargs)
    return new




