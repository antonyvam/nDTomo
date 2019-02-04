#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2003-2018 European Synchrotron Radiation Facility, Grenoble,
#             France
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Utilities, mainly for image treatment
"""

__authors__ = ["Jérôme Kieffer", "Valentin Valls"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "22/10/2018"
__status__ = "production"

import logging
import numpy
import fabio
import weakref
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.optimize.optimize import fmin
from scipy.optimize.optimize import fminbound

class ImageReductionFilter(object):
    """
    Generic filter applied in a set of images.
    """

    def init(self, max_images=None):
        """
        Initialize the filter before using it.
        :param int max_images: Max images supported by the filter
        """
        pass

    def add_image(self, image):
        """
        Add an image to the filter.
        :param numpy.ndarray image: image to add
        """
        raise NotImplementedError()

    def get_parameters(self):
        """Return a dictionary containing filter parameters
        :rtype: dict
        """
        return {"cutoff": None, "quantiles": None}

    def get_result(self):
        """
        Get the result of the filter.
        :return: result filter
        """
        raise NotImplementedError()


class ImageAccumulatorFilter(ImageReductionFilter):
    """
    Filter applied in a set of images in which it is possible
    to reduce data step by step into a single merged image.
    """

    def init(self, max_images=None):
        self._count = 0
        self._accumulated_image = None

    def add_image(self, image):
        """
        Add an image to the filter.
        :param numpy.ndarray image: image to add
        """
        self._accumulated_image = self._accumulate(self._accumulated_image, image)
        self._count += 1

    def _accumulate(self, accumulated_image, added_image):
        """
        Add an image to the filter.
        :param numpy.ndarray accumulated_image: image use to accumulate
            information
        :param numpy.ndarray added_image: image to add
        """
        raise NotImplementedError()

    def get_result(self):
        """
        Get the result of the filter.
        :return: result filter
        :rtype: numpy.ndarray
        """
        result = self._accumulated_image
        # release the allocated memory
        self._accumulated_image = None