"""
Simulations module for nDTomo.

This package contains modules for simulating tomography data, including:
- `shapes2D`: methods related to simulating 2D shapes
- `shapes3D`: methods related to simulating 3D shapes
- `phantoms`: methods for creating multi-dimensional phantoms
- `xrdct`: class for simulating a 2D xrd-ct dataset with 2D diffraction patterns from an xrd-ct sinogram dataset containing 1D diffraction patterns
- `pencilbeam`: methods for simulating various data acquisition strategies used in pencil beam ct
"""

import importlib
import sys

# List of submodules
__all__ = []
modules = ["shapes2D", "shapes3D", "phantoms", "xrdct", "pencilbeam"]

# Import all functions dynamically
for module in modules:
    mod = importlib.import_module(f".{module}", package=__name__)
    globals().update({name: obj for name, obj in vars(mod).items() if not name.startswith("_")})
    __all__.extend([name for name in vars(mod) if not name.startswith("_")])