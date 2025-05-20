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
    Convert photon energy in keV to wavelength in Angstroms.
    
    Parameters
    ----------
    E : float or ndarray
        Photon energy in keV.
        
    Returns
    -------
    float or ndarray
        Wavelength in Angstroms.
    """
    h = 4.135667696e-15  # Planck constant in eV·s
    c = 2.99792458e8     # Speed of light in m/s
    E_eV = E * 1e3       # Convert keV to eV
    wavelength_m = h * c / E_eV  # in meters
    return wavelength_m * 1e10   # Convert to Angstroms

def AngtoKeV(wavel):
    """
    Convert wavelength in Angstroms to photon energy in keV.

    Parameters
    ----------
    wavel : float or ndarray
        Wavelength in Angstroms.
        
    Returns
    -------
    float or ndarray
        Photon energy in keV.
    """
    h = 4.135667696e-15  # Planck constant in eV·s
    c = 2.99792458e8     # Speed of light in m/s
    wavelength_m = wavel * 1e-10
    E_eV = h * c / wavelength_m
    return E_eV / 1e3  # Convert to keV

def tth2q(tth, E):
    """
    Convert 2θ (in degrees) to momentum transfer q (in Å⁻¹).
    
    Parameters
    ----------
    tth : float or ndarray
        Two-theta angle in degrees.
    E : float
        Photon energy in keV.
        
    Returns
    -------
    float or ndarray
        q in Å⁻¹.
    """
    wavel = KeVtoAng(E)  # in Angstroms
    return (4 * np.pi * np.sin(np.deg2rad(tth / 2))) / wavel

def q2tth(q, E):
    """
    Convert momentum transfer q (in Å⁻¹) to 2θ (in degrees).
    
    Parameters
    ----------
    q : float or ndarray
        Momentum transfer in Å⁻¹.
    E : float
        Photon energy in keV.
        
    Returns
    -------
    float or ndarray
        Two-theta angle in degrees.
    """
    wavel = KeVtoAng(E)  # in Angstroms
    theta_rad = np.arcsin((q * wavel) / (4 * np.pi))
    return np.rad2deg(2 * theta_rad)