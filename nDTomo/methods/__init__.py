"""
Methods module for nDTomo.

This package contains modules for peak analysis, including:
- `hyperexpl`: methods related to visualising chemical imaging data
- `metrics`: methods related to calculating common metrics in data science
- `noise`: methods related to noise models
- `plots`: methods related to handling plots
- `xrays`: methods related to X-ray science
- `zip`: methods for zipping files and folders
- `misc`: a broad collection of general-purpose functions and tools used across the nDTomo package
"""

import importlib
import sys

# List of submodules
__all__ = []
modules = ["hyperexpl", "metrics", "noise", "plots", "xrays", "misc", "zip"]

# Import all functions dynamically
for module in modules:
    mod = importlib.import_module(f".{module}", package=__name__)
    globals().update({name: obj for name, obj in vars(mod).items() if not name.startswith("_")})
    __all__.extend([name for name in vars(mod) if not name.startswith("_")])
