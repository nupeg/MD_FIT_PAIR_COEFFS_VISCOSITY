from typing import Any, Self, Mapping, TypeVar

from os import PathLike
from tomllib import load as load_toml
from dataclasses import dataclass

from core.utils import cast

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
    fit_params_results_path: str
    fit_params_checkpoint_path: str | None
    fit_params_acq_func: str | None
    fit_params_n_calls: int | None
    fit_params_n_initial: int | None
    fit_params_random_state: int | None
    
    @classmethod
    def from_file(cls, filepath: PathLike, /) -> Self:
        with open(filepath, mode='rb') as bit_stream:
            values = load_toml(bit_stream)
        
        fit_params_configs = values['FIT_PARAMS']
        fit_params_configs = replace_placeholders(fit_params_configs)

        instance = Settings(
            fit_params_results_path=get_config(fit_params_configs, 'RESULTS_FOLDER', str, nullable=False),
            fit_params_checkpoint_path=get_config(fit_params_configs, 'CHECKPOINT_FOLDER', str, nullable=True),
            fit_params_acq_func=get_config(fit_params_configs, '...', str, nullable=True),
            fit_params_n_calls=get_config(fit_params_configs, 'N_CALLS', int, nullable=True),
            fit_params_n_initial=get_config(fit_params_configs, 'N_INITIAL_POINTS', int, nullable=True),
            fit_params_random_state=get_config(fit_params_configs, 'RANDOM_STATE', int, nullable=True)
        )
        return instance