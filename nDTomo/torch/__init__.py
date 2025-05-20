"""
Pytorch module for nDTomo.

This package contains modules for using Pytorch code, including:
- `models_torch`: methods related to creating various neural network architectures and models
- `tomo_torch`: methods related to computed tomography
- `utils_torch`: methods used for computed tomography, peak fitting and others
- `peak_models_torch`: various peak shape models
"""

from ._mock_torch import *  # Safe import; real or mock depending on env
import importlib
import sys

# List of submodules
__all__ = []
modules = ["models_torch", "tomo_torch", "utils_torch", "peak_models_torch"]

# Import all functions dynamically
for module in modules:
    mod = importlib.import_module(f".{module}", package=__name__)
    globals().update({name: obj for name, obj in vars(mod).items() if not name.startswith("_")})
    __all__.extend([name for name in vars(mod) if not name.startswith("_")])
