# -*- coding: utf-8 -*-
"""
nDAacq to simulate an XRD-CT dataset with 2D diffraction patterns from 1D diffraction patterns

@author: Antony Vamvakeros
"""

#%%

from nDTomo.utils.misc import ndtomopath, showim, addpnoise2D
from nDTomo.ct.virtacq import nDVAcq

#%% Perform a test

p = ndtomopath()
poni = "%sexamples\\xrd2D\\CeO2.poni" %p
savedir = 'Y:\\Antony\\nDTomo\\test'

Acq = nDVAcq(file_format ='cbf', scantype = 'Zigzig', fastaxis = 'Translation', slowaxis = 'Rotation', units = "q_A^-1")

Acq.readponi(poni)
Acq.setdetector(shape=(1000,1000))
Acq.azimint()
Acq.create_nDTomo_phantom(npix = 100, nproj = 110)
Acq.savedir = savedir
Acq.dname = 'xrdct_test'

#%%
p = ndtomopath()
poni = "%sexamples\\xrd2D\\CeO2.poni" %p
savedir = 'Y:\\Antony\\nDTomo\\test2'


Acq = nDVAcq(file_format ='cbf', scantype = 'Zigzig', fastaxis = 'Translation', slowaxis = 'Rotation', units = "q_A^-1")

Acq.setwvl(wvl=1.5e-11)
Acq.setxrdprms(dist=0.35, poni1=(500/2)*0.000172, poni2=(500/2)*0.000172, rot1=0, rot2=0, rot3=0)
Acq.setdetector(shape=(500,500))
Acq.azimint()
Acq.create_nDTomo_phantom(npix = 100, nproj = 110)
Acq.savedir = savedir
Acq.dname = 'xrdct_test'

#%%

im = Acq.conv1Dto2D( Acq.data[50,50,:])
im = addpnoise2D(im, ct = 0.05)

showim(im,1,cmap='gray')

#%%

Acq.saveprms()

#%%

Acq.readscanprms()

#%% Perform the XRD-CT scan

Acq.start_scan(addnoise = 'Yes', ct = 0.05)



















