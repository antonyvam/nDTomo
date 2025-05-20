"""
Analysis module for nDTomo.

This package contains modules for peak analysis, including:
- `peaks`: methods related to peak models and peak fitting
"""

import importlib
import sys

# List of submodules
__all__ = []
modules = ["peaks"]

# Import all functions dynamically
for module in modules:
    mod = importlib.import_module(f".{module}", package=__name__)
    globals().update({name: obj for name, obj in vars(mod).items() if not name.startswith("_")})
    __all__.extend([name for name in vars(mod) if not name.startswith("_")])
