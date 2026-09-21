from typing import Protocol, Hashable

from viscofit.settings import Settings

class SettingsHandler(Protocol):
    def __call__(self, settings: Settings, /) -> None: ...

class MixtureRule(Protocol):
    def __call__(self, atom_type_01_value: float, atom_type_02_value: float, /) -> float: ...


class IdentityHashing:
    identity: Hashable

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self) ) and self.identity == other.identity

    def __hash__(self) -> int:
        return hash(self.identity)