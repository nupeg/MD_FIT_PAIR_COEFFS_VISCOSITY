from typing import (
    Any,
    Mapping, 
    TypeVar,
    TypeAlias, 
    Hashable,
    Callable,
    Protocol,
    Optional,
    NamedTuple,
    runtime_checkable
)
from os import PathLike
from pathlib import Path
from random import randint
from datetime import datetime
from functools import lru_cache

from coolname import generate_slug

@runtime_checkable
class SupportsFspath(Protocol):
    def __fspath__(self) -> str: ...

T   = TypeVar('T')
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

def mro(tp: type, /) -> tuple[type, ...]:
    return tp.__mro__

def cache(func: TF, /) -> TF:
    return lru_cache(func) 

def coalesce(value: Any | None, default: Any, /) -> Any:
    return default if value is None else value

def generate_integer() -> int:
    return randint(0, 1_000_000)

def type_name(it: Any, /) -> str:
    return getattr(type(it), '__name__', 'unnamed_type')

def select_keys(m: Mapping[TK, Any], *keys: TK) -> dict[TK, Any]:
    return {key: m[key] for key in keys}

def search_files(folder: PathLike, pattern: str, /) -> list[Path]:
    files = Path(folder).glob(pattern)
    return list(files)

def is_pathlike(it: Any, /) -> bool:
    return isinstance(it, (str, Path, SupportsFspath, PathLike))

def checkpoint_name(version: int, cool_name: Optional[str]=None, timestamp: Optional[datetime]=None, salt: Optional[int]=None) -> str:
    if cool_name is None:
        cool_name = generate_slug(2)

    if salt is None:
        salt = randint(1000, 9999)

    if timestamp is None:
        timestamp = datetime.now()

    cool_name = str(cool_name)
    version = int(version)
    salt = int(salt)
    timestamp = timestamp.strftime(r'%Y%m%dT%H%M%S')

    return f'checkpoint_v{version:06d}_{cool_name}_{timestamp}_{salt}'

def split_checkpoint_name(text: str, /) -> CheckpointNameParts:
    _, version, cool_name, timestamp, salt = text.split('_')
    version = version.removeprefix('v')

    version = int(version)
    cool_name = str(cool_name)
    timestamp = datetime.strptime(timestamp, r'%Y%m%dT%H%M%S')
    salt = int(salt)

    output = CheckpointNameParts(
        version=version, 
        cool_name=cool_name, 
        timestamp=timestamp, 
        salt=salt
    )
    return output
