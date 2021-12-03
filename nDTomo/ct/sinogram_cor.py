#!/usr/bin/env python

from skimage.transform import iradon
from skimage.transform import iradon_sart
from scipy import ndimage
import scipy.optimize
from scipy.optimize import curve_fit
from skimage.feature import register_translation
from skimage import feature

#ndimage.measurements.center_of_mass(a)
#feature.register_translation

def simplecent(sino):
	""" Applies an integer offset to centre the sinogram """
	xc = numpy.zeros((sino.shape[1]));
	for x in range(0,sino.shape[1]):
		xc[x] = numpy.array(ndimage.measurements.center_of_mass(sino[:,x]))
	gl_xc = numpy.mean(xc)
	o = numpy.round(gl_xc - sino.shape[0]/2)
	if gl_xc<sn2.shape[0]/2:
		sino = sino[0:sino.shape[0]+2*o,:]
	else:
		sino = sino[2*o:sino.shape[0],:]
	return sino, o

def fit_sin(tt, yy):
	'''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
	tt = numpy.array(tt)
	yy = numpy.array(yy)
	ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
	Fyy = abs(numpy.fft.fft(yy))
	guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
	guess_amp = numpy.std(yy) * 2.**0.5
	guess_offset = numpy.mean(yy)
	guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])

	def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c
	popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
	A, w, p, c = popt
	f = w/(2.*numpy.pi)
	fitfunc = lambda t: A * numpy.sin(w*t + p) + c
	return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}

def sinefunc(t, A, w, p, c):  
	return A * numpy.sin(w*t + p) + c

def find_center_pc(proj1, proj2, tol=0.5):
    """
    Find rotation axis location by finding the offset between the first
    projection and a mirrored projection 180 degrees apart using
    phase correlation in Fourier space.
    The ``register_translation`` function uses cross-correlation in Fourier
    space, optionally employing an upsampled matrix-multiplication DFT to
    achieve arbitrary subpixel precision. :cite:`Guizar:08`.

    Parameters
    ----------
    proj1 : ndarray
        2D projection data.

    proj2 : ndarray
        2D projection data.

    tol : scalar, optional
        Subpixel accuracy

    Returns
    -------
    float
        Rotation axis location.
    """

    # create reflection of second projection
    proj2 = np.fliplr(proj2)

    # Determine shift between images using scikit-image pcm
    shift = register_translation(proj1, proj2, upsample_factor=1.0/tol)

    # Compute center of rotation as the center of first image and the
    # registered translation with the second image
    center = (proj1.shape[1] + shift[0][1] - 1.0)/2.0

    return center