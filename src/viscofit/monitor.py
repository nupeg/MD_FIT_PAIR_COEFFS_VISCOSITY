from typing import Any, Protocol, Optional, Self, TypeAlias, Mapping, Hashable

import sys
import time
import msvcrt

import itertools as it
import tqdm as tq
import json as js
import ctypes as ct
import argparse as ap
import subprocess as sp

Json:           TypeAlias = Any
TracebackType:  TypeAlias = type


class Arguments(Protocol):
    total: int | None
    title: str | None
    description: str | None

    keep_open: bool

class MonitorChannel:

    def __init__(self, process: sp.Popen, /) -> None:
        self.process = process

        if self.process.stdin is None:
            raise ValueError('Process stdin instance cannot be None')
        
    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType) -> bool: 
        if exc is not None:
            message = {
                'event': 'error',
                'type': name(exc_type),
                'message': str(exc)
            }
            self.write(message)
        
        finish(self.process)
        return False

    def write(self, message: Optional[Mapping[Hashable, Any] | Any]=None, /) -> None:
        if not isinstance(message, Mapping):
            message = {'message': message}

        message = dict(message)
        message_json = js.dumps(message)    
        
        self.process.stdin.write(message_json + '\n')
    
    def flush(self) -> None:
        self.process.stdin.flush()
    
    def communicate(self, message: Optional[Mapping[Hashable, Any] | Any]=None, /) -> None:
        self.write(message)
        self.flush()

def open_monitor(
        total: Optional[int]=None,  
        title: Optional[str]=None,
        description: Optional[str]=None,
        keep_open: bool=False
    ) -> MonitorChannel:

    cmd = [sys.executable, __file__]

    if total is not None:
        cmd.extend([f'--total', total])

    if title is not None:
        cmd.extend(['--title', title])

    if description is not None:
        cmd.extend(['--description', description])

    if keep_open:
        cmd.extend(['--keep-open'])
    
    cmd = [
        str(value) for value in cmd
    ]
    process = sp.Popen(cmd, stdin=sp.PIPE, creationflags=sp.CREATE_NEW_CONSOLE, text=True)

    return MonitorChannel(process)

def name(it: Any, /, default: Optional[str]=None) -> str:
    return str(getattr(it, '__name__', default))

def finish(process: sp.Popen, /, timeout: Optional[float]=None) -> None:
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    try:
        process.wait(timeout=timeout)
    except sp.TimeoutExpired:
        process.kill()

def wait_key(message: Optional[str]=None, /) -> None:
    if message:
        print(f'\n{message}')
    msvcrt.getch()

def set_console_title(title: str, /) -> None:
    ct.windll.kernel32.SetConsoleTitleW(title)

def set_args_rules(parser: ap.ArgumentParser, /) -> ap.ArgumentParser:
    parser.add_argument('--total', type=int, required=False)
    parser.add_argument('--title', type=str, required=False) 
    parser.add_argument('--description', type=str, required=False) 
    parser.add_argument('--keep-open', action='store_true')
    return parser

def handle_input_stream(
        total: Optional[int]=None,
        description: Optional[str]=None,
        title: Optional[str]=None,
        keep_open: bool=False
    ) -> None:

    if title is None:
        title = 'Python Monitor'

    set_console_title(title)

    bar_format = '{l_bar}{bar:30} {n_fmt}/{total_fmt} | {elapsed}<{remaining} | {rate_fmt} {postfix}'
    
    if total is None:
        bar_format = '{desc} | {elapsed} | {rate_fmt} | step={n_fmt}{postfix}'
    
    progress_bar = tq.tqdm(total=total, desc=description, bar_format=bar_format, colour='cyan', dynamic_ncols=True)
    
    for json in map(str, sys.stdin):
        kwargs = js.loads(json)

        progress_bar.update(1)
        progress_bar.set_postfix(kwargs)

    progress_bar.close()

    if keep_open:
        wait_key('Pressione qualquer tecla para fechar...')

def main() -> None:
    parser = ap.ArgumentParser()
    parser = set_args_rules(parser)

    args: Arguments = parser.parse_args()
    handle_input_stream(
        total=args.total,
        description=args.description,
        title=args.title,
        keep_open=args.keep_open
    )
    
if __name__ == '__main__':
    main()


