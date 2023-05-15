# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 09:19:09 2023

@author: Antony Vamvakeros
"""

import torch


def polynomial(x, coefficients):
    """
    Generate a polynomial with the given coefficients and evaluate it at the x-axis values
    """
    degree = len(coefficients) - 1
    y = torch.zeros_like(x)
    for i in range(degree + 1):
        y += coefficients[i] * x ** i
    return y

def chebyshev_polynomial(x, n_terms):
    """
    Generates a Chebyshev polynomial of the first kind of a given degree.

    Parameters
    ----------
    x : torch.Tensor
        The x-axis values.
    n_terms : int
        The number of terms in the Chebyshev polynomial.

    Returns
    -------
    torch.Tensor
        The Chebyshev polynomial evaluated at the given x-axis values.
    """
    # Create an array of Chebyshev polynomial coefficients
    coeffs = torch.zeros(n_terms)
    coeffs[-1] = 1.0
    coeffs[-2] = 0.0
    if n_terms > 2:
        coeffs[:-2] = -coeffs[2:]
    coeffs = coeffs.to(x.device)

    # Initialize the Chebyshev polynomial
    T_n = torch.tensor(1.0, device=x.device)
    T_n_1 = x

    # Iteratively compute the Chebyshev polynomial
    for i in range(2, n_terms):
        T_n_2 = 2 * x * T_n_1 - T_n
        T_n = T_n_1
        T_n_1 = T_n_2

    return torch.dot(coeffs, T_n_1)

def gaussian_peak(x, amplitude, mean, std):
    return amplitude * torch.exp(-(x - mean)**2 / (2 * std**2))

def lorentzian_peak(x, amplitude, x0, gamma):
    return amplitude / (1 + ((x - x0) / gamma) ** 2)

# Define a function for generating a Pseudo-Voigt peak
def pseudo_voigt_peak(x, amplitude, x0, sigma, fraction):
    """
    fraction is the fraction of the gaussian component, 
    1-fraction is the fraction of lorentzian component
    """
    gaussian = gaussian_peak(x, amplitude, x0, sigma)
    lorentzian = lorentzian_peak(x, amplitude, x0, sigma)
    return fraction*gaussian + (1-fraction)*lorentzian


def pearson7_peak(x, amplitude, x0, sigma, m, a, b, c):
    return amplitude * (1 + m*((x-x0)/sigma)**2) / (1 + a*((x-x0)/sigma)**2 + b*((x-x0)/sigma)**4 + c*((x-x0)/sigma)**6)

def voigt_peak(x, amplitude, x0, sigma, gamma):
    return amplitude * torch.exp(-((x-x0)**2)/(2*sigma**2)) * torch.exp(-gamma*torch.abs(x-x0))


def doniach_sunjic_peak(x, amplitude, x0, sigma, b):
    return amplitude * (1 + (x-x0)**2/(sigma**2 + b*(x-x0)**2))**-1


def split_pearson_peak(x, amplitude, x0, sigma1, sigma2, m1, m2, a1, a2, b1, b2, c1, c2):
    return amplitude * (1 + m1*((x-x0)/sigma1)**2) / (1 + a1*((x-x0)/sigma1)**2 + b1*((x-x0)/sigma1)**4 + c1*((x-x0)/sigma1)**6) + amplitude * (1 + m2*((x-x0)/sigma2)**2) / (1 + a2*((x-x0)/sigma2)**2 + b2*((x-x0)/sigma2)**4 + c2*((x-x0)/sigma2)**6)

def exponential_power_peak(x, amplitude, x0, sigma, alpha, beta):
    return amplitude * (1 + (x-x0)/sigma)**(-alpha) * torch.exp(-beta*(x-x0))















