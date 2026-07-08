from typing import (
    Any,
    Self,
    Union,
    Iterable,
    Mapping, 
    Optional, 
    TypeAlias,
    Protocol,
)
from os import PathLike
from pathlib import Path    
from enum import StrEnum
from shutil import copy2

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Dimension
from skopt.callbacks import CheckpointSaver


from core.utils import (
    coalesce, 
    type_name, 
    generate_integer, 
    select_keys,
    search_files,
    checkpoint_name,
    split_checkpoint_name
)
from main.settings import DEFAULT_SETTINGS_FILEPATH, Settings
from main.artifact import Artifact

class OptimizeFunction(Protocol):
    def __call__(self, **params: dict[str, float]) -> float: ...

class AcquisitionFunctions(StrEnum):
    LOWER_CONFIDENCE_BOUND = 'LCB'
    EXPECTED_IMPROVEMENT = 'EI'
    PROBABILITY_OF_IMPROVEMENT = 'PI'
    GP_HEDGE = 'gp_hedge'



Filepath:       TypeAlias = PathLike
DimensionSpec:  TypeAlias = Union[Iterable[Dimension], Mapping[str, Dimension], Dimension]


def ensure_named_dimensions(*dims: Dimension) -> None:
    for dimension in dims:
        if isinstance(dimension.name, str):
            continue
        raise RuntimeError('All dimensions must have a name.')

def normalize_dimensions(dims: DimensionSpec, /, ensure_named: bool=False) -> list[Dimension]:
    if isinstance(dims, Mapping):
        output = []
        for name, value in dims.items():
            if not isinstance(value, Dimension):
                raise ValueError(f'Expected a Dimension instance for {name!r}, got {type_name(value)!r}.')
            value.name = str(name)
            output.append(value)
        if ensure_named:
            ensure_named_dimensions(*output)
        return output

    if isinstance(dims, Dimension):
        dims = [dims]
    dims = list(dims)

    for index, value in enumerate(dims):
        if not isinstance(value, Dimension):
            raise ValueError(f'Expected a Dimension instance at index {index!s}, got {type_name(value)!r}.')
        
    if ensure_named:
        ensure_named_dimensions(*dims)

    return dims

def force_named_args(func: OptimizeFunction, dimensions: DimensionSpec, /) -> OptimizeFunction:
    return use_named_args(dimensions)(func)






def optimization_workflow(
        objective_function: OptimizeFunction,
        dimensions: DimensionSpec,
        acq_func: Optional[str | AcquisitionFunctions]=None,
        n_calls: Optional[int]=None,
        n_jobs: Optional[int]=None,
        n_initial_points: Optional[int]=None,
        random_state: Optional[int]=None,
        ensure_named_dimensions: bool=False,
        checkpoint: Optional[CheckpointSaver]=None
    ) -> Artifact:
    
    dimensions = normalize_dimensions(dimensions, ensure_named=ensure_named_dimensions)
    objective_function = force_named_args(objective_function, dimensions)

    acq_func = coalesce(acq_func, AcquisitionFunctions.LOWER_CONFIDENCE_BOUND)
    n_calls = coalesce(n_calls, 50)
    n_jobs = coalesce(n_jobs, 1)
    n_initial_points = coalesce(n_initial_points, 10)
    random_state = coalesce(random_state, generate_integer())
    
    state: dict[str, Any] = {
        'acq_func': str(acq_func),
        'n_calls': int(n_calls),
        'n_jobs': int(n_jobs),
        'n_initial_points': int(n_initial_points),
        'random_state': int(random_state),
    }

    if checkpoint is not None:
        checkpoint_state = checkpoint.load()
        checkpoint_state = select_keys(checkpoint_state, 'x0', 'y0', 'base_estimator')

        completed_calls = len(checkpoint_state['x0'])
        remaining_calls = state['n_calls'] - completed_calls

        state.update(checkpoint_state)
        state['n_calls'] = remaining_calls
        state['callbacks'] = [checkpoint]

    output = gp_minimize(func=objective_function, dimensions=dimensions, **state)

    artifact = Artifact.from_scipy(output)
    return artifact

def resolve_dimensions() -> dict[str, Dimension]: ...

def resolve_checkpoint(repository: Optional[PathLike]=None, /) -> CheckpointSaver | None:
    if repository is None:
        return None
    
    repository = Path(repository)

    if not repository.is_dir():
        raise NotADirectoryError(
            f'Expected a valid checkpoint repository, got: {repository}'
        )
    
    checkpoint_files = search_files(repository, r'*checkpoint*.pkl')

    if not checkpoint_files:
        filename = checkpoint_name(version=1)
        filepath = repository.joinpath(filename).with_suffix('.pkl')
        return CheckpointSaver(filepath)

    # NOTE: Get last version, update to new version
    last_checkpoint_filepath = max(checkpoint_files)
    parts = split_checkpoint_name(last_checkpoint_filepath.stem)

    next_version = parts.version + 1

    filename = checkpoint_name(
        version=next_version, cool_name=parts.cool_name, timestamp=parts.timestamp, salt=parts.salt)
    
    next_checkpoint_filepath = last_checkpoint_filepath.with_name(filename)
    next_checkpoint_filepath = next_checkpoint_filepath.with_suffix('.pkl')
    
    # NOTE: Make a copy, preserving versioned files
    copy2(last_checkpoint_filepath, next_checkpoint_filepath)
    checkpoint = CheckpointSaver(next_checkpoint_filepath)

    return checkpoint

def calculate_viscosity_fit_error(**params: Any) -> float: ...

def main() -> None:
    settings = Settings.from_file(DEFAULT_SETTINGS_FILEPATH)
    
    results_path = settings.fit_params_results_path
    checkpoint_path = settings.fit_params_checkpoint_path
    
    if not checkpoint_path is None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(exist_ok=True, parents=True)

    results_path = Path(results_path)
    results_path.mkdir(exist_ok=True, parents=True)

    checkpoint = resolve_checkpoint(settings.fit_params_checkpoint_path)
    dimensions = resolve_dimensions(...)
    
    artifact = optimization_workflow(
        objective_function=calculate_viscosity_fit_error,
        dimensions=dimensions,
        checkpoint=checkpoint,
        acq_func=settings.fit_params_acq_func,
        random_state=settings.fit_params_random_state,
        n_calls=settings.fit_params_n_calls,
        n_initial_points=settings.fit_params_n_initial,
        n_jobs=1,
        ensure_named_dimensions=True,
    )   
    artifact.save(results_path)
    
    
if __name__ == '__main__':
    main()
