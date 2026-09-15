from typing import (
    Any,
    Mapping, 
    TypeVar,
    Iterable, 
    Hashable,
    Callable,
    Protocol,
    Optional,
    NamedTuple,
    Iterator,
    runtime_checkable
)
from shutil import rmtree
from random import randint

from os import PathLike, fspath
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from itertools import combinations_with_replacement
from collections import Counter

from shlex import split as shell_split
from functools import partial, lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from shutil import copy
from platform import system
from subprocess import CREATE_NEW_CONSOLE, run

from questionary import Style, Choice, select
from coolname import generate_slug

@runtime_checkable
class SupportsFspath(Protocol):
    def __fspath__(self) -> str: ...

T   = TypeVar('T')
R   = TypeVar('R')
TF  = TypeVar('TF', bound=Callable)
TK  = TypeVar('TK', bound=Hashable)
TV  = TypeVar('TV', bound=object)


class CheckpointNameParts(NamedTuple):
    version: int
    cool_name: str
    timestamp: datetime
    salt: int


def cast(value: Any, dtype: Callable[[Any], T], /, ignore_null: bool=False) -> T:
    if ignore_null and value is None:
        return None
    return dtype(value)

def keys(m: Mapping[TK, Any], /) -> list[TK]:
    return list(m.keys())

def values(m: Mapping[Hashable, TV], /) -> list[TV]:
    return list(m.values())

def filename(filepath: PathLike, /, remove_extension: bool=False) -> str:
    return Path(filepath).stem if remove_extension else Path(filepath).name

def get_parent_and_filename(filepath: PathLike, /) -> tuple[str, str]:
    filepath = Path(filepath)
    return filepath.parent, filepath.name

def mro(tp: type, /) -> tuple[type, ...]:
    return tp.__mro__

def cache(func: TF, /) -> TF:
    return lru_cache(func) 

def add_prefix(data: str, prefix: str, /) -> str:
    if data.startswith(prefix):
        return data
    return f'{prefix}{data}'

def join_as_text(values: Iterable[Any], sep: str, /) -> str:
    return str(sep).join(map(str, values))

def coalesce(value: Any | None, default: Any, /) -> Any:
    return default if value is None else value

def generate_integer() -> int:
    return randint(0, 1_000_000)

def type_name(it: Any, /) -> str:
    return getattr(type(it), '__name__', 'unnamed_type')

def get_name(it: Any, /, default: Optional[str]=None) -> str:
    if default is None:
        default = 'unnamed'
    return str(getattr(it, '__name__', default))

def select_keys(m: Mapping[TK, Any], *keys: TK) -> dict[TK, Any]:
    return {key: m[key] for key in keys}

def search_files(folder: PathLike, pattern: str, /, recursive: bool=False) -> list[Path]:
    searcher = Path(folder).rglob if recursive else Path(folder).glob
    files = searcher(pattern)
    return list(files)

def sorted_tuple(values: Iterable[T], /, key: Optional[Callable]=None, reverse: bool=False) -> tuple[T, ...]:
    return tuple(sorted(values, key=key, reverse=reverse))

def is_pathlike(it: Any, /) -> bool:
    return isinstance(it, (str, Path, SupportsFspath, PathLike))

def map_filenames(*filepaths: PathLike, no_extension: bool=False) -> dict[str, Path]:
    key: Callable[[Path], str] = lambda path: path.stem if no_extension else path.name
    return {key(filepath): filepath for filepath in map(Path, filepaths) }

def get_difference(values: Iterable[Hashable], other: Iterable[Hashable], /, sort: bool=False) -> list[Hashable]:
    diff = set(values).difference(other)
    if sort:
        return sorted(diff)
    return diff

def apply(*values: Any, transformation: Callable[[Any], R]) -> list[R]:
    return list(map(transformation, values))

def pipe(data: Any, *functions: Callable[[Any], Any] ) -> Any:
    for func in functions:
        data = func(data)
    return data

def copy_files(*filepaths: PathLike, destination_folder: PathLike, create_folder: bool=False) -> list[Path]:
    destination_folder = Path(destination_folder)

    if create_folder:
        destination_folder.mkdir(parents=True, exist_ok=True)

    copied_filepaths = []

    for filepath in map(Path, filepaths):
        filepath = copy(filepath, destination_folder)
        copied_filepaths.append(
            Path(filepath)
        )
    return copied_filepaths

def ensure_file_exists(*filepaths: PathLike) -> None:
    for filepath in map(Path, filepaths):
        if not filepath.is_file():
            raise FileExistsError(f'The file {filepath!r} not exists!')
    
def ensure_no_symlinks(path: PathLike, /) -> None:
    for part in Path(path).rglob("*"):
        if part.is_symlink():
            raise ValueError(
                f'Symlink detected: {part}'
            )
        
def run_parallel(items: Iterable[T], process: Callable[[T], R], njobs: Optional[int]=None, timeout: Optional[float]=None, maintain_order: bool=False) -> Iterator[R]: 
    with ThreadPoolExecutor(max_workers=njobs) as workers:
        if maintain_order:
            yield from workers.map(
                process, 
                items, 
                timeout=timeout
            )
            return
        futures = (
            workers.submit(process, item) for item in items
        )
        for future in as_completed(futures, timeout=timeout):
            yield future.result()

def worker(task: Callable[[T], R], workpool: Queue) -> Iterator[R]:
    while True:
        try:
            item = workpool.get_nowait()
        except Empty:
            break

        try:
            yield task(item)
        finally:
            workpool.task_done()

def run_work_pool(items: Iterable[T], processes: Iterable[Callable[[T], R]], /) -> Iterator[R]:
    processes = list(processes)
    total_processes = len(processes)
    
    workpool = Queue()

    for item in items:
        workpool.put(item)

    with ThreadPoolExecutor(max_workers=total_processes) as executor:
        workers = (
            partial(worker, task=task, workpool=workpool) for task in processes
        )
        futures = [
            executor.submit(prepared_worker) for prepared_worker in workers
        ]
        for future in as_completed(futures):
            yield from future.result()

def split_selecting(text: str, index: int, /, sep: Optional[str]=None) -> str:
    return str(text).split(sep=sep)[index]

def setup_dir(folder_path: PathLike, /, create: bool=False, clear: bool=False) -> Path:
    folder_path = Path(folder_path)

    if create:
        folder_path.mkdir(exist_ok=True, parents=True)

    if clear:
        if not folder_path.is_dir():
            raise NotADirectoryError(
                f'Cannot clear non-existing directory: {folder_path!r}'
            )
        ensure_no_symlinks(folder_path)
        
        for sys_path in folder_path.iterdir():

            if sys_path.is_dir():
                rmtree(sys_path)
                continue
            
            if sys_path.is_file():
                sys_path.unlink()
                continue

    return folder_path

def enumerate_unique(values: Iterable[TK], /) -> dict[TK, int]:
    ordering = {}
    for value in values:
        if value in ordering:
            continue
        ordering[value] = len(ordering) + 1
    return ordering

def ensure_not_duplicates(values: Iterable[Any], /) -> None:
    not_uniques = {
        value: count for value, count in Counter(values).items() if count > 1
    }
    if not_uniques:
        raise ValueError(
            f'The following values must be uniques, instead got repetitions:\n'
            f'{not_uniques}'
        )

def pairwise_combinations(values: Iterable[T], /) -> list[tuple[T, T]]:
    return [pair for pair in combinations_with_replacement(values, 2) ]

def dict_from_lists(keys: Iterable[TK], values: Iterable[TV], /) -> dict[TK, TV]:
    return dict(zip(keys, values, strict=True))

def ask_user_choice(options: Iterable[str] | Mapping[str, str], /) -> str:
    style = Style.from_dict({
        'qmark':        'fg:#5f87ff bold',
        'question':     'bold',
        'pointer':      'fg:#ff9d00 bold',
        'highlighted':  'fg:#ff9d00 bold',
        'selected':     'fg:#5f87ff',
        'answer':       'fg:#5f87ff bold',
        'separator':    'fg:#6c6c6c',
        'instruction':  'fg:#6c6c6c',
        'disabled':     'fg:#6c6c6c italic',
    })

    if isinstance(options, Mapping):
        options = (
            Choice(title=name, value=name, description=description) for name, description in options.items()
        )
    choices = list(options)
    
    selection = select(
        message='Please, select an option:',
        choices=choices, 
        style=style
    )
    return selection.ask()

def memorable_name(size: int) -> str:
    return generate_slug(size)

def parse_cmd(cmd: str | PathLike | Iterable[str], /) -> list[str]:
    if isinstance(cmd, str):
        return shell_split(cmd)
    if is_pathlike(cmd):
        return [fspath(cmd)]
    return [str(value) for value in cmd]

def execute_script(filepath: PathLike, cmd: str | Iterable[str], /, new_terminal: bool=False) -> None:
    folder_path, filename = get_parent_and_filename(filepath)

    cmd_parts = parse_cmd(cmd)

    if not cmd_parts:
        raise ValueError(f'Cannot parse shell command {cmd!r}.')
    
    cmd_parts.append(filename)

    creationflags = 0

    if new_terminal and system() in {'Windows'}:
        creationflags = CREATE_NEW_CONSOLE

    run(cmd_parts, cwd=folder_path, creationflags=creationflags, check=True, text=True, shell=False)

def docstring(func: Any, /) -> str:
    return str(getattr(func, '__doc__', 'No documentation available.'))