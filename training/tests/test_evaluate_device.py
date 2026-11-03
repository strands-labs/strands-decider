"""`python -m training.evaluate eval|calibrate --device mlx` (or neuron) is refused."""

import pytest
import typer

from training.evaluate import _device


@pytest.mark.parametrize("device", ["mlx", "neuron"])
def test_device_refuses_serving_backends(device):
    with pytest.raises(typer.BadParameter):
        _device(device)


def test_device_passes_an_explicit_choice_through():
    assert _device("cpu") == "cpu"
