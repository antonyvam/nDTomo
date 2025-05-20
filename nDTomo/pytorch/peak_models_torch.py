# -*- coding: utf-8 -*-
"""
Analytical peak functions and polynomial models using PyTorch.

This module defines several 1D peak profiles (e.g., Gaussian, Lorentzian, Pseudo-Voigt) and 
polynomial bases commonly used in signal processing, spectroscopy, and diffraction analysis. 
These functions are differentiable and compatible with PyTorch-based optimization and learning workflows.

Author: Antony Vamvakeros
"""

import torch


def polynomial(x, coefficients):
    """
    Evaluate a 1D polynomial at the given x values using provided coefficients.

    The polynomial is of the form:
        y = c₀ + c₁·x + c₂·x² + ... + cₙ·xⁿ

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of x values at which to evaluate the polynomial.
    coefficients : list or torch.Tensor
        Sequence of polynomial coefficients [c₀, c₁, ..., cₙ].

    Returns
    -------
    torch.Tensor
        Evaluated polynomial y(x).
    """
    degree = len(coefficients) - 1
    y = torch.zeros_like(x)
    for i in range(degree + 1):
        y += coefficients[i] * x ** i
    return y

def chebyshev_polynomial(x, n_terms):
    """
    Generate a Chebyshev polynomial of the first kind Tₙ(x), evaluated using recurrence relations.

    The Chebyshev polynomials of the first kind are defined by:
        T₀(x) = 1
        T₁(x) = x
        Tₙ₊₁(x) = 2x·Tₙ(x) - Tₙ₋₁(x)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of x values.
    n_terms : int
        Number of terms or the target degree of the polynomial.

    Returns
    -------
    torch.Tensor
        The Chebyshev polynomial of the first kind Tₙ(x) evaluated at `x`.

    Notes
    -----
    This implementation returns the dot product of the coefficient vector and the final
    recurrence value. It does not return the full basis expansion.
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
    """
    Generate a Gaussian peak.

    Parameters
    ----------
    x : torch.Tensor
        Input axis values.
    amplitude : float or torch.Tensor
        Peak amplitude.
    mean : float or torch.Tensor
        Peak center.
    std : float or torch.Tensor
        Standard deviation (controls width).

    Returns
    -------
    torch.Tensor
        Gaussian peak evaluated at `x`.
    """    
    return amplitude * torch.exp(-(x - mean)**2 / (2 * std**2))

def lorentzian_peak(x, amplitude, x0, gamma):
    """
    Generate a Lorentzian peak.

    Parameters
    ----------
    x : torch.Tensor
        Input axis values.
    amplitude : float or torch.Tensor
        Peak amplitude.
    x0 : float or torch.Tensor
        Center of the peak.
    gamma : float or torch.Tensor
        Half-width at half-maximum (HWHM).

    Returns
    -------
    torch.Tensor
        Lorentzian peak evaluated at `x`.
    """    
    return amplitude / (1 + ((x - x0) / gamma) ** 2)

def pseudo_voigt_peak(x, amplitude, x0, sigma, fraction):
    """
    Generate a symmetric Pseudo-Voigt peak composed of a linear combination of Gaussian 
    and Lorentzian components with equal width.

    Parameters
    ----------
    x : torch.Tensor
        Input axis values.
    amplitude : float or torch.Tensor
        Peak amplitude.
    x0 : float or torch.Tensor
        Peak center position.
    sigma : float or torch.Tensor
        Common width used for both Gaussian and Lorentzian components.
    fraction : float
        Mixing ratio of the Gaussian component. Must be in [0, 1]. 
        The Lorentzian component weight is (1 - fraction).

    Returns
    -------
    torch.Tensor
        Pseudo-Voigt peak evaluated at `x`.
    """
    gaussian = gaussian_peak(x, amplitude, x0, sigma)
    lorentzian = lorentzian_peak(x, amplitude, x0, sigma)
    return fraction*gaussian + (1-fraction)*lorentzian


def pearson7_peak(x, amplitude, x0, sigma, m, a, b, c):
    """
    Generate a Pearson VII-type peak with flexible tailing.

    Parameters
    ----------
    x : torch.Tensor
        Input axis values.
    amplitude : float
        Peak amplitude.
    x0 : float
        Peak center.
    sigma : float
        Width scale factor.
    m, a, b, c : float
        Tail-shaping parameters.

    Returns
    -------
    torch.Tensor
        Pearson VII profile evaluated at `x`.
    """    
    return amplitude * (1 + m*((x-x0)/sigma)**2) / (1 + a*((x-x0)/sigma)**2 + b*((x-x0)/sigma)**4 + c*((x-x0)/sigma)**6)

def voigt_peak(x, amplitude, x0, sigma, gamma):
    """
    Approximate a Voigt peak (convolution of Gaussian and Lorentzian).

    Parameters
    ----------
    x : torch.Tensor
        Input axis values.
    amplitude : float
        Peak amplitude.
    x0 : float
        Peak center.
    sigma : float
        Gaussian width.
    gamma : float
        Lorentzian width (controls tails).

    Returns
    -------
    torch.Tensor
        Voigt-like peak evaluated at `x`.
    """    
    return amplitude * torch.exp(-((x-x0)**2)/(2*sigma**2)) * torch.exp(-gamma*torch.abs(x-x0))


def doniach_sunjic_peak(x, amplitude, x0, sigma, b):
    """
    Generate a Doniach-Šunjić peak with asymmetric broadening.

    Parameters
    ----------
    x : torch.Tensor
        Input axis values.
    amplitude : float
        Peak amplitude.
    x0 : float
        Peak center.
    sigma : float
        Base width parameter.
    b : float
        Asymmetry parameter.

    Returns
    -------
    torch.Tensor
        Doniach-Šunjić peak evaluated at `x`.
    """    
    return amplitude * (1 + (x-x0)**2/(sigma**2 + b*(x-x0)**2))**-1

def split_pearson_peak(x, amplitude, x0, sigma1, sigma2, m1, m2, a1, a2, b1, b2, c1, c2):
    """
    Generate a two-component asymmetric Pearson peak.

    Parameters
    ----------
    x : torch.Tensor
        Input axis values.
    amplitude : float
        Peak amplitude.
    x0 : float
        Peak center.
    sigma1, sigma2 : float
        Widths for left and right peak components.
    m1, m2, a1, a2, b1, b2, c1, c2 : float
        Tail parameters for left and right components.

    Returns
    -------
    torch.Tensor
        Asymmetric Pearson profile evaluated at `x`.
    """    
    return amplitude * (1 + m1*((x-x0)/sigma1)**2) / (1 + a1*((x-x0)/sigma1)**2 + b1*((x-x0)/sigma1)**4 + c1*((x-x0)/sigma1)**6) + amplitude * (1 + m2*((x-x0)/sigma2)**2) / (1 + a2*((x-x0)/sigma2)**2 + b2*((x-x0)/sigma2)**4 + c2*((x-x0)/sigma2)**6)

def exponential_power_peak(x, amplitude, x0, sigma, alpha, beta):
    """
    Generate a skewed peak based on an exponential power law.

    Parameters
    ----------
    x : torch.Tensor
        Input axis values.
    amplitude : float
        Peak amplitude.
    x0 : float
        Peak center.
    sigma : float
        Width parameter.
    alpha : float
        Power law decay exponent.
    beta : float
        Skewness factor.

    Returns
    -------
    torch.Tensor
        Peak evaluated at `x` using the exponential power law form.
    """
    return amplitude * (1 + (x-x0)/sigma)**(-alpha) * torch.exp(-beta*(x-x0))

def pseudo_voigt_peak_t(x, A, xo, st, fraction, sl, i):
    """
    Time-efficient version of a Pseudo-Voigt peak with a linear background.

    Parameters
    ----------
    x : torch.Tensor
        Input axis values.
    A : float
        Amplitude of the peak.
    xo : float
        Center position.
    st : float
        Width of the Gaussian/Lorentzian components.
    fraction : float
        Gaussian fraction; (1 - fraction) is Lorentzian fraction.
    sl : float
        Slope of the linear background.
    i : float
        Intercept of the linear background.

    Returns
    -------
    torch.Tensor
        Composite Pseudo-Voigt + linear background evaluated at `x`.
    """ 
    return fraction*(A * torch.exp(-(x - xo)**2 / (2 * st**2))) + (1-fraction)*((A / torch.pi) * (st / ((x - xo)**2 + st**2))) + sl*x + i













