# -*- coding: utf-8 -*-
"""
Perform a 1D to 2D diffraction pattern conversion

@author: Antony Vamvakeros
"""

from nDTomo.utils.misc import h5read_data, h5write_data, closefigs, showplot, showspectra, showim, addpnoise1D, addpnoise2D
from nDTomo.utils.misc import KeVtoAng
from nDTomo.sim.shapes.phantoms import load_example_patterns

import time
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors
from pyFAI.calibrant import get_calibrant

#%%

dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()

#%%
dist = 0.75 # sample-to-detector distance in meters
detpixsz = 0.000172 # detector pixel size

E = 50 # Energy in KeV
wavelength = KeVtoAng(E)/1E10

beam_centre_x = 1000 
beam_centre_y = 1000 

poni1 = beam_centre_x * detpixsz
poni2 = beam_centre_y * detpixsz

detector = pyFAI.detectors.Detector(pixel1=detpixsz, pixel2=detpixsz, max_shape=(2000,2000))
ai = AzimuthalIntegrator(dist=dist, poni1=poni1, poni2=poni2, rot1=0, rot2=0, rot3=0, detector=detector, wavelength=wavelength)

# dp = addpnoise1D(dpAl, 100)

start = time.time()
img_theo = ai.calcfrom1d(q, dpAl, mask=None, dim1_unit="q_A^-1",
                         correctSolidAngle=False,
                         polarization_factor=0.95)
print(time.time() -  start)

im = addpnoise2D(img_theo, 0.1)

showim(im, 2, clim=(0,5), cmap='gray')

