"""When the Neuron backend's tests run: on a Trainium (or Inferentia) instance only.

`NEURON` is true where torch-neuronx is installed and the Neuron driver's devices exist. It
finds the package without importing it, so collecting the tests stays cheap everywhere. In CI,
on GPU hosts and on Macs it is false, and every test marked `needs_neuron` skips. On a Trn2
instance, inside the Neuron container's venv, `pytest -q -m neuron` runs only these tests.
"""

from __future__ import annotations

import glob
import importlib.util
import os

import pytest

NEURON = importlib.util.find_spec("torch_neuronx") is not None and bool(glob.glob("/dev/neuron*"))
REAL = os.environ.get("STRANDS_DECIDER_TEST_CHECKPOINT")

needs_neuron = [
    pytest.mark.neuron,
    pytest.mark.skipif(
        not NEURON, reason="needs a Trainium instance (torch_neuronx and /dev/neuron*)"
    ),
]
needs_real_neuron = [
    *needs_neuron,
    pytest.mark.skipif(not REAL, reason="set STRANDS_DECIDER_TEST_CHECKPOINT"),
]
