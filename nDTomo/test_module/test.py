# -*- coding: utf-8 -*-
"""
Test module for nDTomo documentation.

This module is used to verify that Sphinx correctly generates API documentation.
It contains example functions and a class with properly formatted docstrings.

Functions
---------
- `add(a, b)`: Returns the sum of two numbers.
- `multiply(a, b)`: Returns the product of two numbers.

Classes
-------
- `Calculator`: A simple calculator class.
"""

def add(a, b):
    """
    Add two numbers.

    Parameters
    ----------
    a : int or float
        The first number.
    b : int or float
        The second number.

    Returns
    -------
    int or float
        The sum of `a` and `b`.
    """
    return a + b

def multiply(a, b):
    """
    Multiply two numbers.

    Parameters
    ----------
    a : int or float
        The first number.
    b : int or float
        The second number.

    Returns
    -------
    int or float
        The product of `a` and `b`.
    """
    return a * b

class Calculator:
    """
    A simple calculator class.

    Methods
    -------
    add(a, b)
        Returns the sum of two numbers.
    multiply(a, b)
        Returns the product of two numbers.
    """

    def add(self, a, b):
        """
        Add two numbers.

        Parameters
        ----------
        a : int or float
            The first number.
        b : int or float
            The second number.

        Returns
        -------
        int or float
            The sum of `a` and `b`.
        """
        return a + b

    def multiply(self, a, b):
        """
        Multiply two numbers.

        Parameters
        ----------
        a : int or float
            The first number.
        b : int or float
            The second number.

        Returns
        -------
        int or float
            The product of `a` and `b`.
        """
        return a * b