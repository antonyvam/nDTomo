"""
Tomography module for nDTomo.

This package contains modules for simulating tomography data, including:
- `sinograms`: methods related to preprocessing and handling sinograms
- `conv_tomo`: methods related to reconstructing CT images using numpy and scipy
- `astra_tomo`: methods related to reconstructing CT images using the astra-toolbox

"""

import importlib
import sys

# List of submodules
__all__ = []
modules = ["sinograms", "conv_tomo", "astra_tomo"]

# Import all functions dynamically
for module in modules:
    mod = importlib.import_module(f".{module}", package=__name__)
    globals().update({name: obj for name, obj in vars(mod).items() if not name.startswith("_")})
    __all__.extend([name for name in vars(mod) if not name.startswith("_")])
