"""
nDTomoGUI module

This package contains modules for the nDTomoGUI:
- `nDTomoGUI`: methods related to the nDTomoGUI
"""

import importlib
import sys

# List of submodules
__all__ = []
modules = ["nDTomoGUI"]

# Import all functions dynamically
for module in modules:
    mod = importlib.import_module(f".{module}", package=__name__)
    globals().update({name: obj for name, obj in vars(mod).items() if not name.startswith("_")})
    __all__.extend([name for name in vars(mod) if not name.startswith("_")])
