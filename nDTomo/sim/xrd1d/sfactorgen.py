# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:44:39 2020

@author: Antony
"""


import numpy as np
from libraries import atomlib
import periodictable as elements
from libraries import SpaceGroups as sglib

def FormFactor(atomtype, stl):
    """
     Calculation of the atomic form factor at a specified sin(theta)/lambda
     using the analytic fit to the  form factors from 
     Int. Tab. Cryst Sect. C 6.1.1.4

     INPUT:   atomtype: Atom type (string) e.g. 'C' 
              stl: form factor calculated sin(theta)/lambda = stl
     OUTPUT:  atomic form factor (no dispersion)
              

     Henning Osholm Sorensen, Risoe National Laboratory, April 9, 2008.

    """

    data = atomlib.formfactor[atomtype]

    # Calc form factor
    formfac = 0

    for i in range(4):
        formfac = formfac + data[i]*np.exp(-data[i+4]*stl*stl) 
    formfac = formfac + data[8]

    return formfac


def StructureFactor(q, hkl, ucell, mysg, atoms, disper = None):
    """
    Calculation of the structure factor of reflection hkl
    
    [Freal Fimg] = StructureFactor(hkl,unit_cell,sg,atoms)
    
    INPUT : hkl =       [h, k, l] 
            unit_cell = [a, b, c, alpha, beta, gamma] 
            sgname:     space group name (e.g. 'P 21/c')
            atoms:      structural parameters (as an object)
    OUTPUT: The real and imaginary parts of the the structure factor
    
    Henning Osholm Sorensen, June 23, 2006.
    Translated to python code April 8, 2008
    """
    
    stl = sintl(ucell, hkl)
    noatoms = len(atoms)

    Freal = 0.0
    Fimg = 0.0

    for i in range(noatoms):
        #Check whether isotrop or anisotropic displacements 
        if atoms[i].adp_type == 'Uiso':
            U = atoms[i].adp
            expij = np.exp(-8*np.pi**2*U*stl**2)
        # elif atoms[i].adp_type == 'Uani':
        #     # transform Uij to betaij
        #     betaij = Uij2betaij(atoms[i].adp, ucell)
        else:
            expij = 1
            atoms[i].adp = 'Uiso'
            #logging.error("wrong no of elements in atomlist")

        # Atomic form factors
        # f = FormFactor(atoms[i].atomtype, stl)
        f = elements.elements[atoms[i].atomic_numbers].xray.f0(q)
        
        if disper == None or disper[atoms[i].atomtype] == None :
            fp = 0.0
            fpp = 0.0
        else:
            fp = disper[atoms[i].atomtype][0]
            fpp = disper[atoms[i].atomtype][1]

        for j in range(mysg.nsymop):
            # atomic displacement factor
            # if atoms[i].adp_type == 'Uani':
            #     betaijrot = np.dot(mysg.rot[j], np.dot(betaij, mysg.rot[j]))
            #     expij = np.exp(-np.dot(hkl, np.dot(betaijrot, hkl)))
                
            # exponent for phase factor
            r = np.dot(mysg.rot[j], atoms[i].pos) + mysg.trans[j]
            exponent = 2*np.pi*np.dot(hkl, r)

            #forming the real and imaginary parts of F
            s = np.sin(exponent)
            c = np.cos(exponent)
            site_pop = atoms[i].occ*atoms[i].symmulti/mysg.nsymop
            Freal = Freal + expij*(c*(f+fp)-s*fpp)*site_pop
            Fimg = Fimg + expij*(s*(f+fp)+c*fpp)*site_pop

    return [Freal, Fimg]


def sintl(unit_cell, hkl):
    """
    sintl calculate sin(theta)/lambda of the reflection "hkl" given
    the unit cell "unit_cell" 
    
    sintl(unit_cell,hkl)
    
    INPUT:  unit_cell = [a, b, c, alpha, beta, gamma]
            hkl = [h, k, l]
    OUTPUT: sin(theta)/lambda
    
    Henning Osholm Sorensen, Risoe National Laboratory, June 23, 2006.
    """
    a   = float(unit_cell[0])
    b   = float(unit_cell[1])
    c   = float(unit_cell[2])
    calp = np.cos(unit_cell[3]*np.pi/180.)
    cbet = np.cos(unit_cell[4]*np.pi/180.)
    cgam = np.cos(unit_cell[5]*np.pi/180.)

    (h, k, l) = hkl
    
    part1 = (h*h/a**2) * (1-calp**2) + (k*k/b**2) *\
            (1-cbet**2) + (l*l/c**2) * (1-cgam**2) +\
            2*h*k*(calp*cbet-cgam)/(a*b) + 2*h*l*(calp*cgam-cbet)/(a*c) +\
            2*k*l*(cbet*cgam-calp)/(b*c)

    part2 = 1 - (calp**2 + cbet**2 + cgam**2) + 2*calp*cbet*cgam

    stl = np.sqrt(part1) / (2*np.sqrt(part2))
    
    return stl

def multiplicity(position, sgname=None, sgno=None, cell_choice='standard'):
    """
    Calculates the multiplicity of a fractional position in the unit cell.
    If called by sgno, cell_choice is necessary for eg rhombohedral space groups.

    """

    if sgname != None:
        mysg = sglib.sg(sgname=sgname, cell_choice=cell_choice)
    elif sgno !=None:
        mysg = sglib.sg(sgno=sgno, cell_choice=cell_choice)
    else:
        raise ValueError('No space group information provided')

    lp = np.zeros((mysg.nsymop, 3))

    for i in range(mysg.nsymop):
        lp[i, :] = np.dot(position, mysg.rot[i]) + mysg.trans[i]

    lpu = np.array([lp[0, :]])
    multi = 1

    for i in range(1, mysg.nsymop):
        for j in range(multi):
            t = lp[i]-lpu[j]
            if np.sum(np.mod(t, 1)) < 0.00001:
                break
            else:
                if j == multi-1:
                    lpu = np.concatenate((lpu, [lp[i, :]]))
                    multi += 1
    return multi

def int_intensity(F2, L, P, I0, wavelength, cell_vol, cryst_vol):
    """
    Calculate the reflection intensities scaling factor
    
    INPUT:
    F2        : the structure factor squared
    L         : Lorentz factor
    P         : Polarisation factor
    I0        : Incoming beam flux
    wavelength: in Angstroem
    cell_vol  : Volume of unit cell in AA^3
    cryst_vol : Volume of crystal in mm^3

    OUTPUT:
    int_intensity: integrated intensity

    """
#    print F2,L,P,I0,wavelength,cell_vol,cryst_vol
    
    emass = 9.1093826e-31
    echarge = 1.60217653e-19
    pi4eps0 = 1.11265e-10
    c = 299792458.0
    k1 = (echarge**2/(pi4eps0*emass*c**2)*1000)**2 # Unit is mm
    # the factor 1e21 used below is to go from mm^3 to AA^3
    k2 = wavelength**3 * cryst_vol * 1e21/cell_vol**2 
    return k1*k2*I0*L*P*F2