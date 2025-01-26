# -*- coding: utf-8 -*-
"""
nDAacq to simulate an XRD-CT dataset with 2D diffraction patterns from 1D diffraction patterns

@author: Antony Vamvakeros
"""

#%%

from nDTomo.utils.misc import ndtomopath
from nDTomo.utils.noise import addpnoise2D
from nDTomo.ct.virtacq import nDVAcq
from nDTomo.utils.plots import showim

#%% Perform a test

p = ndtomopath()
poni = "%sexamples\\xrd2D\\CeO2.poni" %p
savedir = 'Y:\\Antony\\nDTomo\\test'

Acq = nDVAcq(file_format ='cbf', scantype = 'Zigzig', fastaxis = 'Translation', slowaxis = 'Rotation', units = "q_A^-1")

Acq.readponi(poni)
Acq.setdetector(shape=(1000,1000))
Acq.azimint()
Acq.create_nDTomo_phantom(npix = 101, nproj = 110)
Acq.savedir = savedir
Acq.dname = 'xrdct_test'

#%%

p = ndtomopath()
savedir = 'Y:\\Antony\\nDTomo\\test'

Acq = nDVAcq(file_format ='cbf', scantype = 'Zigzig', fastaxis = 'Translation', slowaxis = 'Rotation', units = "q_A^-1")

Acq.setwvl(wvl=1.24e-11)
Acq.setxrdprms(dist=0.35, poni1=(500/2)*0.000172, poni2=(500/2)*0.000172, rot1=0, rot2=0, rot3=0)
Acq.setdetector(shape=(500,500))
Acq.azimint()
Acq.create_nDTomo_phantom(npix = 101, nproj = 110)
Acq.savedir = savedir
Acq.dname = 'xrdct_test'

#%%

im = Acq.conv1Dto2D( Acq.data[50,50,:])
im = addpnoise2D(im, ct = 0.15)

showim(im,1,cmap='gray')

#%%

Acq.saveprms()

#%%

Acq.readscanprms()

#%% Perform the XRD-CT scan

Acq.start_scan(addnoise = 'Yes', ct = 0.15)







#%% Integrate the data

import pyFAI, fabio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from nDTomo.utils.hyperexpl import HyperSliceExplorer
from nDTomo.ct.astra_tomo import astra_rec_vol
from nDTomo.ct.conv_tomo import sinocentering
from nDTomo.utils.misc import cirmask

#%%

fn = 'Y:\\Antony\\nDTomo\\test\\xrdct_test_1_50.cbf'

im = fabio.open(fn).data

showim(im)

poni = 'Y:\\Antony\\nDTomo\\calib.poni'
ai = pyFAI.load(poni)


npt_rad = 250


kwargs = {"npt":npt_rad,
          "unit":"2th_deg",
          "correctSolidAngle": False,
          "polarization_factor": 0.95}


kwargs["method"] = pyFAI.method_registry.IntegrationMethod.select_method(dim=1, split="no",
                                                   algo="csr",
                                                   impl="python")[0]

print(kwargs)

tth, I = ai.integrate1d(data = im, **kwargs)

plt.figure(2);plt.clf()
plt.plot(tth,I)
plt.show()

#%%

npix = 101
nproj = 110

sinos = np.zeros((npix, nproj, npt_rad), dtype = 'float32')

for ii in tqdm(range(nproj)):
    for jj in range(npix):

        fn = 'Y:\\Antony\\nDTomo\\test\\xrdct_test_%d_%d.cbf' %(ii+1, jj+1)

        im = fabio.open(fn).data
        
        tth, I = ai.integrate1d(data = im, **kwargs)

        sinos[jj,ii,:] = I


#%%

s = sinocentering(sinos)

#%%

hs = HyperSliceExplorer(s)
hs.explore()

#%%

rec = astra_rec_vol(s)
rec[rec<0] = 0

#%%

rec = cirmask(rec, 2)

#%%

hs = HyperSliceExplorer(rec)
hs.explore()




