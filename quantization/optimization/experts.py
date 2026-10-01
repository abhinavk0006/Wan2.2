"""Framework-independent lazy loading and exclusive expert swapping."""

from __future__ import annotations

import gc
from collections.abc import Callable

import torch


class ExpertManager:
    """Manage named high/low experts without coupling to WanI2V.

    ``get`` lazily loads an expert and leaves other loaded experts alone.
    ``activate`` enforces a one-resident-expert policy by releasing the
    previously active expert before activating the next one.
    """

    def __init__(
        self,
        loader: Callable[[str], object],
        *,
        device: str | torch.device = "cpu",
        mover: Callable[[object, str | torch.device], object] | None = None,
        releaser: Callable[[object], None] | None = None,
        exclusive: bool = True,
    ) -> None:
        self.loader = loader
        self.device = torch.device(device)
        self.mover = mover or self._move
        self.releaser = releaser or self._release
        self.exclusive = exclusive
        self._experts: dict[str, object] = {}
        self.active_name: str | None = None

    @staticmethod
    def _move(expert: object, device: str | torch.device) -> object:
        to = getattr(expert, "to", None)
        return to(device) if callable(to) else expert

    @staticmethod
    def _release(expert: object) -> None:
        to = getattr(expert, "to", None)
        if callable(to):
            to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def loaded_names(self) -> tuple[str, ...]:
        return tuple(self._experts)

    def get(self, name: str) -> object:
        if name not in self._experts:
            self._experts[name] = self.loader(name)
        return self._experts[name]

    def activate(self, name: str) -> object:
        if self.exclusive:
            for other_name in tuple(self._experts):
                if other_name != name:
                    self.unload(other_name)
        expert = self.get(name)
        expert = self.mover(expert, self.device)
        self._experts[name] = expert
        self.active_name = name
        return expert

    def unload(self, name: str) -> bool:
        expert = self._experts.pop(name, None)
        if expert is None:
            return False
        self.releaser(expert)
        if self.active_name == name:
            self.active_name = None
        return True

    def unload_all(self) -> None:
        for name in tuple(self._experts):
            self.unload(name)

