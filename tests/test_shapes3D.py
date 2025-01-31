

import numpy as np
import pytest

from nDTomo.sim.shapes3D import (
    create_sphere)

# Test sphere creation
def test_create_sphere():
    shape = (10, 10, 10)
    vol = np.zeros(shape, dtype='float32')
    vol = create_sphere(vol, center=(5,5,5), outer_radius=6, thickness=0, fill_value=1)
