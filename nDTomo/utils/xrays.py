# -*- coding: utf-8 -*-
"""
Methods for X-rays

@author: Antony Vamvakeros
"""

from numpy import pi, sin, arcsin, deg2rad, rad2deg

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
    q = pi*2/(wavel/(2*sin(deg2rad(0.5*tth))))

    return(q)

def q2tth(q, E):

    """
	Convert q to 2theta
	"""  	
    
    h = 6.620700406E-34;c = 3E8
    wavel = 1E10*6.242E18*h*c/(E*1E3)
    
    
    tth = rad2deg(2*arcsin(wavel/(4*pi/q)))

    return(tth)