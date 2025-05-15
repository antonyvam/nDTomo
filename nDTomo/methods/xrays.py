# -*- coding: utf-8 -*-
"""
Methods for X-rays

@author: Antony Vamvakeros
"""

import numpy as np

def simulate_synchrotron_intensity(num_points, drop_ratio=0.75, num_topups=1):
    """
    Simulate synchrotron beam intensity with linear decay and periodic top-ups.

    Parameters
    ----------
    num_points : int
        Total number of points in the output array.
    drop_ratio : float
        Intensity level just before each top-up (e.g., 0.75).
    num_topups : int
        Number of top-up events (i.e., decay cycles) in the vector.

    Returns
    -------
    intensity : np.ndarray
        Simulated beam intensity values over time.
    """
    if num_topups < 1:
        raise ValueError("num_topups must be at least 1.")
    
    points_per_cycle = num_points // num_topups
    decay = np.linspace(1.0, drop_ratio, points_per_cycle, endpoint=False)

    # Repeat decay cycles
    intensity = np.tile(decay, num_topups)

    # Pad remaining points if needed
    remainder = num_points - len(intensity)
    if remainder > 0:
        last_cycle = np.linspace(1.0, drop_ratio, points_per_cycle, endpoint=False)
        intensity = np.concatenate([intensity, last_cycle[:remainder]])

    return intensity

def KeVtoAng(E):
    
    """
	Convert photon energy in KeV to Angstrom
	"""  
    
    h = 6.620700406E-34;c = 3E8
    return(1E10*6.242E18*h*c/(E*1E3))


def AngtoKeV(wavel):
    
    """
	Convert photon energy in KeV to Angstrom
	"""  
    
    h = 6.620700406E-34;c = 3E8
    return(1E10*6.242E18*h*c/(wavel*1E3))

def tth2q(tth, E):

    """
	Convert 2theta to q
    E is energy in KeV
	"""  	
    
    wavel = KeVtoAng(E)
    q = np.pi*2/(wavel/(2*np.sin(np.deg2rad(0.5*tth))))

    return(q)

def q2tth(q, E):

    """
	Convert q to 2theta
	"""  	
    
    h = 6.620700406E-34;c = 3E8
    wavel = 1E10*6.242E18*h*c/(E*1E3)
    
    
    tth = np.rad2deg(2*np.arcsin(wavel/(4*np.pi/q)))

    return(tth)