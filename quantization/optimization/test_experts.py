import torch

from quantization.optimization.experts import ExpertManager


class FakeExpert:
    def __init__(self, name):
        self.name = name
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self


def test_high_and_low_experts_load_and_unload_independently():
    loads = []
    manager = ExpertManager(lambda name: loads.append(name) or FakeExpert(name))
    high = manager.get("high_noise")
    low = manager.get("low_noise")
    assert high.name == "high_noise"
    assert low.name == "low_noise"
    assert manager.loaded_names == ("high_noise", "low_noise")
    assert manager.unload("high_noise")
    assert manager.loaded_names == ("low_noise",)
    assert manager.get("low_noise") is low
    assert loads == ["high_noise", "low_noise"]
    assert manager.unload("low_noise")
    assert manager.loaded_names == ()


def test_activation_lazily_swaps_experts_and_releases_previous():
    loads = []
    released = []
    manager = ExpertManager(
        lambda name: loads.append(name) or FakeExpert(name),
        device="cpu",
        releaser=lambda expert: released.append(expert.name),
    )
    high = manager.activate("high_noise")
    assert high.moves == ["cpu"]
    manager.activate("low_noise")
    assert released == ["high_noise"]
    assert manager.loaded_names == ("low_noise",)
    assert manager.active_name == "low_noise"
    assert loads == ["high_noise", "low_noise"]
    manager.unload_all()
