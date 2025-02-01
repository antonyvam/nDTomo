#!/usr/bin/python
# coding: utf8
# /*##########################################################################
#
# Copyright (c) 2019 Dr A. Vamvakeros
#
# This package is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# 
# This package is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>
# 
# On Debian systems, the complete text of the GNU General
# Public License version 3 can be found in "/usr/share/common-licenses/GPL-3".
#
# ###########################################################################*/
  

from setuptools import setup, find_packages

setup(
	name="nDTomo",
	version="2024.02",
	description="nDTomo software suite",
	url="http://github.com/antonyvam/nDTomo",
	author="A. Vamvakeros",
	author_email="antonyvam@gmail.com",
	install_requires=[
		"h5py", "matplotlib", "numpy",  "scipy", "siphash24",
		"pyqtgraph", "scikit-image", "scikit-learn", "xdesign", "tqdm", "ipykernel",
		"pystackreg", "pyopencl", "tifffile", "jupyterlab", 
		"mayavi", "algotom", "fabio","pyFAI>=2025.01", "napari",
	],
    extras_require={
        "torch": ["torch", "torchvision", "torchaudio"]
    },
	packages=find_packages(),
    package_data={
        '': ['*.txt', '*.rst'],
    },
    entry_points={
        'gui_scripts': ['nDTomoGUI = nDTomo.gui.nDTomoGUI:main']
    },		
	license="LICENSE.txt",
	classifiers=[
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering",
		"Topic :: Scientific/Engineering :: Chemistry",
		"Topic :: Scientific/Engineering :: Visualization",
		],
) 

