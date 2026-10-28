"""Guards the CPU-only install."""

import importlib

import torch


class TestEnvironment:
    """Checks the installed torch and pyg-lib builds."""

    def test_torch_is_a_cpu_build(self):
        """Torch reports no CUDA version."""
        assert torch.version.cuda is None

    def test_pyg_lib_is_installed(self):
        """The pyg-lib wheel imports."""
        assert importlib.import_module("pyg_lib") is not None
