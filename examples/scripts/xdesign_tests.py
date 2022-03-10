# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:11:29 2021

@author: Antony
"""

import matplotlib.pyplot as plt
import numpy as np
from xdesign import Mesh, HyperbolicConcentric,SiemensStar, Circle, Triangle, DynamicRange, DogaCircles, Phantom, Polygon, Point, SimpleMaterial, plot_phantom, discrete_phantom, SlantedSquares

# Make a circle with a triangle cut out
m = Mesh()
m.append(Circle(Point([0.0, 0.0]), radius=0.5))
m.append(-Triangle(Point([-0.3, -0.2]),
                   Point([0.0, -0.3]),
                   Point([0.3, -0.2])))

head = Phantom(geometry=m)

# Make two eyes separately
eyeL = Phantom(geometry=Circle(Point([-0.2, 0.0]), radius=0.1))
eyeR = Phantom(geometry=Circle(Point([0.2, 0.0]), radius=0.1))
# poly = Phantom(geometry=Polygon([Point([-0.2, -0.2]), Point([0.0, 0.2]), Point([0.2, 0.1])]))


material = SimpleMaterial(mass_attenuation=1.0)

head.material = SimpleMaterial(mass_attenuation=0.5)
eyeL.material = SimpleMaterial(mass_attenuation=1.0)
eyeR.material = SimpleMaterial(mass_attenuation=1.0)
# poly.material = material

head.append(eyeL)
head.append(eyeR)
# head.append(poly)

# print(repr(p))

p = SlantedSquares(count=16, angle=15/360*2*np.pi, gap=0.05)
# h = HyperbolicConcentric()
d = DogaCircles(n_sizes=8, size_ratio=0.75, n_shuffles=2)
s = SiemensStar(32)

nt = 256
im = discrete_phantom(head, nt, prop='mass_attenuation')
# im = discrete_phantom(p, nt)
# im = discrete_phantom(h, nt)
# im = discrete_phantom(d, nt)
# im = discrete_phantom(s, nt)

# im = discrete_phantom(p, 200, prop='mass_attenuation') + discrete_phantom(head, 200, prop='mass_attenuation') + discrete_phantom(h, 200)
# im = discrete_phantom(d, 200) + discrete_phantom(s, 200)


# im = [discrete_phantom(p, nt, prop='mass_attenuation') + 
#       discrete_phantom(head, nt, prop='mass_attenuation')*2 + 
#       discrete_phantom(h, nt)*3 + 
#       discrete_phantom(d, nt)*4 + 
#       discrete_phantom(s, nt)*5][0]

plt.figure(1);plt.clf();
plt.imshow(im, cmap = 'jet')
plt.colorbar()
plt.show()

#%%


fig = plt.figure(figsize=(7, 3), dpi=100)

# plot geometry
axis = fig.add_subplot(121, aspect='equal')
plt.grid()
plot_phantom(head, axis=axis, labels=False)
plt.xlim([-.5, .5])
plt.ylim([-.5, .5])

# plot property
plt.subplot(1, 2, 2)
im = plt.imshow(discrete_phantom(head, 100, prop='mass_attenuation'),
                interpolation='none', cmap=plt.cm.inferno, origin='lower')

# plot colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.16, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

# # save the figure
# plt.savefig('Shepp_sidebyside.png', dpi=600,
#         orientation='landscape', papertype=None, format=None,
#         transparent=True, bbox_inches='tight', pad_inches=0.0,
#         frameon=False)
plt.show()