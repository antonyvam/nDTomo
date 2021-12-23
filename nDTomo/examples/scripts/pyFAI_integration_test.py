# -*- coding: utf-8 -*-
"""
pyFAI tests

Author: Antony Vamvakeros
"""

import numpy as np
import fabio, pyFAI, time, sys
import matplotlib.pyplot as plt
print(pyFAI.version)

'''
This is also crucial
'''

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

import pyopencl as cl

platforms = cl.get_platforms()
devices = [plat.get_devices(cl.device_type.ALL) for plat in platforms]
devices = [dev for devices in devices for dev in devices]
print(devices)

#%%

poni = "C:\\Users\\Admin\\Documents\\GitHub\\nDTomo\\nDTomo\\examples\\xrd2D\\CeO2.poni"
ai = pyFAI.load(poni)


fn = 'C:\\Users\\Admin\\Documents\\GitHub\\nDTomo\\nDTomo\\examples\\xrd2D\\CeO2.cbf'

im = fabio.open(fn)
im = np.array(im.data)

plt.figure(2);plt.clf();
plt.imshow(im, cmap ='jet')
plt.colorbar()
plt.clim(0, 3*np.std(im))
plt.show()

#%% Mask

msk = fabio.open("C:\\Users\\Admin\\Documents\\GitHub\\nDTomo\\nDTomo\\examples\\xrd2D\\mask.edf")
msk = np.array(msk.data)

plt.figure(1);plt.clf();
plt.imshow(msk, cmap ='jet')
plt.colorbar()
plt.show()

#%%

npt_rad = 790

Imethod = pyFAI.method_registry.IntegrationMethod.select_method(dim=1, split="bbox", algo="csr", impl="opencl", target_type="gpu")
# tth, I = ai.integrate1d(data=im, npt=npt_rad, mask=msk, unit="2th_deg", method='csr_ocl_gpu', correctSolidAngle=False, polarization_factor=0.95)


kwargs = {"npt":npt_rad,
          "correctSolidAngle": False,
          "polarization_factor": 0.95}

kwargs["method"] = pyFAI.method_registry.IntegrationMethod.select_method(dim=1, split="bbox split",
                                                   algo="csr",
                                                   impl="opencl",
                                                   target_type="gpu")[0]


print(kwargs)

tth, I = ai.integrate1d(data = im, unit="2th_deg", mask=msk, **kwargs)

plt.figure(2);plt.clf();
plt.plot(tth,I)
plt.show();

#%% Equivalence of different rebinning engines ... looking for the fastest:

'''
Taken from pyFAI documentation
'''    

kwargs = {"npt":npt_rad,
          "unit":"2th_deg",
          "mask":msk,
          "correctSolidAngle": False,
          "polarization_factor": 0.95}
fastest = sys.maxsize
best = None
print(f"| {'Method':70s} | {'error':8s} | {'time(ms)':7s}|")
print("+"+"-"*72+"+"+"-"*10+"+"+"-"*9+"+")

kk = 0
for method in pyFAI.method_registry.IntegrationMethod.select_method(dim=1):
    
    if kk not in [10, 11]:
        
        kwargs["method"] = method
        res_flat = ai.integrate1d(im, **kwargs)
        #print(f"timeit for {method} max error: {abs(res_flat.intensity-1).max()}")
        m = str(method).replace(")","")[26:96]
        err = abs(res_flat.intensity-1).max()
    
        tm = %timeit -o -r1 -q ai.integrate1d(im, **kwargs)
        tm_best = tm.best
        print(f"| {m:70s} | {err:6.2e} | {tm_best*1000:7.3f} |")
        if tm_best<fastest:
            fastest = tm_best
            best = method
        
    kk = kk + 1
print("+"+"-"*72+"+"+"-"*10+"+"+"-"*9+"+")
print(f"\nThe fastest method is {best} in {1000*fastest:.3f} ms/1Mpix frame")










