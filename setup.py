#!/usr/bin/python
# coding: utf8
# /*##########################################################################
#
# Copyright (c) 2018 Finden
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
  
from setuptools import setup

<<<<<<< HEAD
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

=======
with open("README.md", "r") as fh:
    long_description = fh.read()
>>>>>>> 3fb35ab491ff4613baebfb4408fe068a5cca257a

setup(
	name="nDTomo",
	version="0.1.0",
	description="nDTomo software suite",
	url="http://github.com/antonyvam/nDTomo",
	author="A. Vamvakeros",
	author_email="antony@finden.co.uk",
	install_requires=[
		"fabio >= 0.6.0",
		"h5py >= 2.7.1",
		"matplotlib >= 1.4.2",
		"numpy >= 1.8.2",
		"pyFAI >=0.13.0",
		"scipy >=0.14.0",
		"silx >=0.6.0",
		"scikit-image >=0.10.0",
	],
	packages=['nDTomo'],
	extras_require={
		"PDF":  ["diffpy"],
		"Iterative algorithms": ["tomopy", "astra"],
	},
    package_data={
        '': ['*.txt', '*.rst'],
    },
	license="LICENSE.txt",
	classifiers=[
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering",
		"Topic :: Scientific/Engineering :: Chemistry",
		"Topic :: Scientific/Engineering :: Visualization",
		]
	
<<<<<<< HEAD
) 
=======
) 		
>>>>>>> 3fb35ab491ff4613baebfb4408fe068a5cca257a
