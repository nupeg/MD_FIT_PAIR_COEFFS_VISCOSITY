from typing import Protocol

from viscofit.settings import Settings

class SettingsHandler(Protocol):
    def __call__(self, settings: Settings, /) -> None: ...

