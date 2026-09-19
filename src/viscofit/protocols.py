from typing import Protocol

from viscofit.settings import Settings

class SettingsHandler(Protocol):
    def __call__(self, settings: Settings, /) -> None: ...

class MixtureRule(Protocol):
    def __call__(self, atom_type_01_value: float, atom_type_02_value: float, /) -> float: ...
