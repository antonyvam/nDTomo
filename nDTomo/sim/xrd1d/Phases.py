# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:38:53 2020

@author: Antony
"""

import numpy as np
import matplotlib.pyplot as plt
import time, os, sys, re
import ase.io
from lmfit import minimize, Parameters
import periodictable as elements

# import periodictable as elements
'''Import DiffracTools code'''
import CSReader
import ReflectionsGeneration as rf
from libraries import SpaceGroups as sglib
from libraries import atmdata

sind = lambda x: np.sin(x*np.pi/180.)
asind = lambda x: 180.*np.arcsin(x)/np.pi
tand = lambda x: np.tan(x*np.pi/180.)
atand = lambda x: 180.*np.arctan(x)/np.pi
atan2d = lambda y,x: 180.*np.arctan2(y,x)/np.pi
cosd = lambda x: np.cos(x*np.pi/180.)
acosd = lambda x: 180.*np.arccos(x)/np.pi
rdsq2d = lambda x,p: np.round(1.0/np.sqrt(x),p)

# globals
gc1 = 2.0*(np.log(2.0)/np.pi)**0.5
gc2 = 4.0*np.log(2.0)
K = 0.9

class phase(object):
    
    def __init__(self):
        
        try:
            self.prepare_pars()
        except:
            pass

        
    def prepare_pars(self):
        
        self.ScF = 1
        
        self.pars = Parameters()
        self.pars.add('CLS', 500)
        self.pars.add('SF', 1)
    
        self.pars.add('U', 0.001)
        self.pars.add('V', -0.001)
        self.pars.add('W', 0.001)
        
        self.pars.add('a')
        self.pars.add('b')
        self.pars.add('c')
        self.pars.add('alpha')
        self.pars.add('beta')
        self.pars.add('gamma')

        self.pars.add('bkga', 1.0)
        self.pars.add('bkgb', 0.0)
        
    def readCIFinfo(self, fn, dispinfo = True):

        
        self.cifinfo =  CSReader.build_atomlist()
        self.cifinfo.CIFopen(fn)
        self.cifinfo.CIFread(fn)
                
        self.ucellpars = self.cifinfo.atomlist.cell
        
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = self.ucellpars
              
        self.pars['a'].value = self.a
        self.pars['b'].value = self.b
        self.pars['c'].value = self.c
        self.pars['alpha'].value = self.alpha
        self.pars['beta'].value = self.beta
        self.pars['gamma'].value = self.gamma
        
        
        self.sgno = int(self.cifinfo.atomlist.sgno)
        self.sgname = self.cifinfo.atomlist.sgname
        self.natms = len(self.cifinfo.atomlist.atom)
        self.atoms = self.cifinfo.atomlist.atom
        self.cform = self.cifinfo.atomlist.formula
        
        if dispinfo:
            print('Unit cell parameters: ', self.ucellpars)
            print('Space group name: ', self.sgname)
            print('Space group number: ', self.sgno)
            print('Number of atoms:', self.natms)
        
        self.amlist = []; self.aplist = []; self.aolist = []; self.allist = []; self.asyms = []; self.amnum = []
        
        for ii in range(0, self.natms):
        
            self.allist.append(self.cifinfo.atomlist.atom[ii].label)
            
            self.aolist.append(self.cifinfo.atomlist.atom[ii].occ)
            
            self.aplist.append(self.cifinfo.atomlist.atom[ii].pos)
            
            self.amlist.append(self.cifinfo.atomlist.atom[ii].symmulti)
            
            if dispinfo:
                print('Atom type:', self.cifinfo.atomlist.atom[ii].atomtype)
                print('Atom label:',self.cifinfo.atomlist.atom[ii].label)
                print('Atom occupancy:',self.cifinfo.atomlist.atom[ii].occ)
                print('Atom position:',self.cifinfo.atomlist.atom[ii].pos)
                print('Atom symmetry multiplicity:',self.cifinfo.atomlist.atom[ii].symmulti)
            
            for a in range(len(atmdata.chemical_symbols)):
        
                sy = re.split('(\d+)',self.allist[ii])
                
                if atmdata.chemical_symbols[a] == sy[0]:
                    
                    self.asyms.append(sy[0].upper())
                    self.amnum.append(a)
                    
            self.atoms[ii].atomtype = self.asyms[ii]
            self.atoms[ii].atomic_numbers = self.amnum[ii]
                
        if dispinfo:
            print('Atom list: ', self.asyms, self.amnum, self.aolist, self.amlist)
        
        #Ase part
        self.cell = ase.io.read(fn)         
        self.atms = self.cell.get_chemical_symbols()
        self.pos = self.cell.get_scaled_positions()
        self.natoms = self.cell.get_atomic_numbers().size
        self.spaceGroup = self.cell.info['spacegroup']
        
        self.getCrystalSystem()
        self.symmetry()
        
    def getCrystalSystem(self):
        crystalClasses = ['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal', 'Hexagonal', 'Cubic']
        if self.sgno <= 2:
            self.crystalSystem = 0
        elif self.sgno >= 3 and self.sgno <= 15:
            self.crystalSystem = 1
        elif self.sgno >= 16 and self.sgno <= 74:
            self.crystalSystem = 2
        elif self.sgno >= 75 and self.sgno <= 142:
            self.crystalSystem = 3
        elif self.sgno >= 143 and self.sgno <= 167:
            self.crystalSystem = 4
        elif self.sgno >= 168 and self.sgno <= 194:
            self.crystalSystem = 5
        elif self.sgno >= 195 and self.sgno <= 230:
            self.crystalSystem = 6
            
        self.crystalClass = crystalClasses[self.crystalSystem]
        tmp = self.cell.get_cell_lengths_and_angles()
        self.shortCell = np.unique(tmp[tmp!=90])
                
    def symmetry(self, sintlmax= 0.5):
        
        self.spg = getattr(sglib, "Sg%s" %self.sgno)()
        sintlmin = 0 #sin(theta_min)/wavelength
        
        hkl = rf.genhkl_base(self.ucellpars, self.spg.syscond, sintlmin, sintlmax, crystal_system=self.spg.crystal_system, Laue_class = self.spg.Laue, cell_choice = self.spg.cell_choice, output_stl=False)
        self.hkl = hkl[:,:3]
        self.mcalc()
        
    def mcalc(self):            
            
        self.j = np.zeros((self.hkl.shape[0]))
        for ii in range(0,self.hkl.shape[0]):
            self.j[ii] = len(self.spaceGroup.equivalent_reflections(self.hkl[ii,:]))

    def set_wvl(self,wvl):
        
        self.wvl = wvl
        
    def set_unitcell_pars(self, a, b, c, alpha, beta, gamma):
        
        self.pars['a'].value = a
        self.pars['b'].value = b
        self.pars['c'].value = c
        self.pars['alpha'].value = alpha
        self.pars['beta'].value = beta
        self.pars['gamma'].value = gamma
        self.ucellpars = [a, b, c, alpha, beta, gamma]
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = self.ucellpars
                
    def set_bkg(self, a, b):
        
        self.pars['bkga'].value = self.a
        self.pars['bkgb'].value = self.b
        
    def ttheta_gen(self):
        
        self.twotheta = np.zeros((self.hkl.shape[0]))
        for ii in range(0,self.hkl.shape[0]):
            self.twotheta[ii] = 180*self.tthcalc(hkl = self.hkl[ii,:])/np.pi
        self.d = self.wvl/(2*sind(0.5*self.twotheta))
        self.q = 2 * np.pi / self.d
            
    def tthcalc(self, hkl):
            
        return(2*np.arcsin(self.wvl*self.stlcalc(hkl)))

    def stlcalc(self, hkl):
        
        calp = np.cos(self.alpha*np.pi/180.)
        cbet = np.cos(self.beta*np.pi/180.)
        cgam = np.cos(self.gamma*np.pi/180.)
    
        (h, k, l) = hkl
        
        part1 = (h*h/self.a**2) * (1-calp**2) + (k*k/self.b**2) *\
                (1-cbet**2) + (l*l/self.c**2) * (1-cgam**2) +\
                2*h*k*(calp*cbet-cgam)/(self.a*self.b) + 2*h*l*(calp*cgam-cbet)/(self.a*self.c) +\
                2*k*l*(cbet*cgam-calp)/(self.b*self.c)
    
        part2 = 1 - (calp**2 + cbet**2 + cgam**2) + 2*calp*cbet*cgam
    
        self.stl = np.sqrt(part1) / (2*np.sqrt(part2))
    
        return(self.stl)
    
    def PLcalc(self):
        
        self.P = (1.0 + cosd(self.twotheta)**2 )/2.0
        self.L = 1 / (sind(self.twotheta/2.0) * sind(self.twotheta))
   
    def StructureFactor(self, q, hkl, disper = None):

        stl = self.stlcalc(hkl)
        noatoms = len(self.atoms)
    
        Freal = 0.0
        Fimg = 0.0
    
        for i in range(noatoms):
            #Check whether isotrop or anisotropic displacements 
            if self.atoms[i].adp_type == 'Uiso':
                U = self.atoms[i].adp
                expij = np.exp(-8*np.pi**2*U*stl**2)
            # elif self.atoms[i].adp_type == 'Uani':
            #     # transform Uij to betaij
            #     betaij = Uij2betaij(self.atoms[i].adp, self.ucellpars)
            else:
                expij = 1
                self.atoms[i].adp = 'Uiso'
                #logging.error("wrong no of elements in atomlist")
    
            # Atomic form factors
            # f = FormFactor(atoms[i].atomtype, stl)
            f = elements.elements[self.atoms[i].atomic_numbers].xray.f0(q)
            
            if disper == None or disper[self.atoms[i].atomtype] == None :
                fp = 0.0
                fpp = 0.0
            else:
                fp = disper[self.atoms[i].atomtype][0]
                fpp = disper[self.atoms[i].atomtype][1]
    
            for j in range(self.spg.nsymop):
                # atomic displacement factor
                # if atoms[i].adp_type == 'Uani':
                #     betaijrot = np.dot(mysg.rot[j], np.dot(betaij, mysg.rot[j]))
                #     expij = np.exp(-np.dot(hkl, np.dot(betaijrot, hkl)))
                    
                # exponent for phase factor
                r = np.dot(self.spg.rot[j], self.atoms[i].pos) + self.spg.trans[j]
                exponent = 2*np.pi*np.dot(hkl, r)
    
                #forming the real and imaginary parts of F
                s = np.sin(exponent)
                c = np.cos(exponent)
                site_pop = self.atoms[i].occ*self.atoms[i].symmulti/self.spg.nsymop
                Freal = Freal + expij*(c*(f+fp)-s*fpp)*site_pop
                Fimg = Fimg + expij*(s*(f+fp)+c*fpp)*site_pop
    
        return(Freal, Fimg)

    def Icalc(self):
        
        self.ttheta_gen()
        self.PLcalc()
                
        self.I = np.zeros((self.hkl.shape[0]))
        
        for ii in range(self.hkl.shape[0]):
            Freal, Fimg = self.StructureFactor(q = self.q[ii], hkl = self.hkl[ii])
            F2_hkl = Freal**2 + Freal**2 
            self.I[ii] = self.j[ii] * self.P[ii] * self.L[ii] * F2_hkl 

        # self.I = self.I/np.max(self.I)
        return(self.I)
    
    def plot_refl(self):
        plt.figure(1);plt.clf();
        for ii in range(0,len(self.twotheta)):
            plt.plot([self.twotheta[ii],self.twotheta[ii]], [0,self.I[ii]]);
        plt.show();    
        
    def instpars(self, U= 0.02, V= -0.01, W= 0.01):
        
        self.pars['U'].value = U 
        self.pars['V'].value = V
        self.pars['W'].value = W
        
        self.U, self.V, self.W = U, V, W
        
    def set_sf(self, SF=1):
        
        self.pars['SF'].value = SF
        self.ScF = SF
        
    def set_xaxis(self, tth_1d = np.linspace(0.5, 110, 1000)):

        self.tth_1d = tth_1d
        self.d_1d = self.wvl/(2*sind(self.tth_1d / 2.0))
        self.q_1d = 2 * np.pi / self.d_1d
        self.tth_1d_step = self.tth_1d[1] - self.tth_1d[0]
        
    def ph_cls(self, CLS = 500):
        
        self.pars['CLS'].value = CLS
        self.CLS = CLS
        
    def create_pattern(self):
        
        # self.I = self.Icalc()
        self.I = self.Icalc_ase()
        
        self.binst = self.U * tand(self.twotheta / 2.0) ** 2 + self.V * tand(self.twotheta / 2.0) + self.W
        self.bsample = ((180/np.pi)*(K*self.wvl)/((self.CLS*10) * cosd(self.twotheta/2.0)))**2
        self.btotal = np.sqrt(self.binst + self.bsample)
        
        
        # self.btotal = np.sqrt(self.binst)
        
        self.model = np.zeros_like(self.tth_1d)
        
        for ii in range(0,len(self.I)):
            
            self.model = self.model + gc1*self.I[ii]/self.btotal[ii] * np.exp(- (self.tth_1d - self.twotheta[ii])**2 * (gc2/self.btotal[ii]**2))

        self.model = self.model*self.ScF
        return(self.model)
    
    def bkg_calc(self):
        
        self.bkg = self.tth_1d*self.pars['bkga'].value + self.pars['bkgb'].value
    
    def plot_pattern(self):

        plt.figure(2);plt.clf();
        plt.plot(self.q_1d,self.model);
        plt.show();

    def d2tth(self,d):
        self.tth = 2.0*np.degrees(np.arcsin(0.5*self.wvl/d))

    def tth2d(self, tth):
        self.d = 0.5*self.wvl/np.sin( np.radians(tth/2.0) )

    def q2tth(self, q):
        self.tth = 2.0*np.degrees(np.arcsin(np.pi*self.wvl/q))
        
    def set_xy(self, x, y):
        
        self.x = x
        self.y = y
        
    def FitPattern(self):
        
        for i in range(len(self.pars)):
            self.fitParams = self.fitParams + self.pars[i]  
            
        self.out = minimize(self.rmodel, self.pars, args=(self.x, self.y))

    def rmodel(self, params, x, y):
        model = self.create_pattern()
        return(100*np.mean(y-model))

    # Deprecated ASE code
    # def aseGetStoicheomtery(self, s):
    #     occ = self.cell.info.get('occupancy') 
    #     #print(occ)
    #     value = 0
    #     jj = [k for k,v in occ.items() if s in v ]
    #     tags = self.cell.get_tags()
    #     for j in jj:
    #         value = value + occ[j][s]*(tags==j).sum()
    #     #sym = self.cell.get_chemical_symbols()
    #     #value = value*sym.count(s)
    #     #print (self.name, ':', s, ' ',value, ' ', sym.count(s))
    #     return value
        
    def Icalc_ase(self):

        self.ttheta_gen()
        self.PLcalc()
                
        self.I = np.zeros((self.hkl.shape[0]))
                
        for ii in range(self.hkl.shape[0]):
        
            # # get the scattering factors
            f = np.empty(0)
            
            # Using Ase
            for atoms in range(self.natoms):               
                f = np.append(f,elements.elements[self.cell.get_atomic_numbers()[atoms]].xray.f0(self.q[ii]))
            dot2pi = np.zeros((len(self.atms),))
            for jj in range(0,len(self.atms)):
                  dot2pi[jj] = 2*np.pi*np.dot(self.pos[jj,:],self.hkl[ii])

            a = np.sum(np.dot(np.cos(dot2pi), f))
            b = np.sum(np.dot(np.sin(dot2pi), f))
            F2_hkl = a**2 + b**2 
            
            # print(F2_hkl)
            
            self.I[ii] = self.j[ii] * self.P[ii] * self.L[ii] * F2_hkl 
            
            
        # self.I = self.I/np.max(self.I)
        return(self.I)
            

    def Volcalc(self):

        if (self.crystalClass == 6) or (self.crystalClass == 3) or (self.crystalClass == 2 and self.ucellpars[4] == 90):
            
            self.Vol = self.a*self.b*self.c
        
        elif (self.crystalClass == 1) and (self.ucellpars[3] == 90 and self.ucellpars[5] == 90):
            
            self.Vol = self.a*self.b*self.c*sind(self.beta)
            
        else:
            
            self.Vol = self.a*self.b*self.c*np.sqrt(1 + 2*cosd(self.alpha)*cosd(self.beta)*cosd(self.gamma) - cosd(self.alpha)**2 - cosd(self.beta)**2 - cosd(self.gamma)**2 )
        
    def Dcalc(self):
                
        if self.crystalSystem == 6:
            
            self.d_star = np.sqrt( self.a**2 / (self.h**2 + self.k**2 + self.l**2) )
            
        elif self.crystalSystem == 5:
            
            self.d_star = np.sqrt( (4/3) * ( (self.h**2 + self.h*self.k +self.k**2) / self.a**2 ) + (self.l / self.c)**2 )
            
        elif self.crystalSystem == 4:
            
            self.d_star = np.sqrt( ( (self.h**2 + self.k**2 + self.l**2)*sind(self.alpha)**2 + 2*(self.h*self.k + self.k*self.l + self.h*self.l)*(cosd(self.alpha)**2 - cosd(self.alpha)) ) / (self.a**2 * (1 - 3*cosd(self.alpha)**2 + 2*cosd(self.alpha)**3 ) ) )
            
        elif self.crystalSystem == 3:
            
            self.d_star = np.sqrt( (self.h**2 + self.k**2)/self.a**2 + (self.l / self.c)**2)
            
        elif self.crystalSystem == 2:
            
            self.d_star = np.sqrt( (self.h / self.a)**2 + (self.k / self.b)**2 + (self.l / self.c)**2 )
            
        elif self.crystalSystem == 1:
            
            self.d_star = np.sqrt( (self.h / self.a)**2 + ( (self.k * sind(self.beta) )/self.b )**2 + (self.l / self.c)**2 + (2*self.h*self.k*cosd(self.beta)/(self.a*self.c)) ) * (1 / sind(self.beta))
    
        elif self.crystalSystem == 0:
            
            p1 = (sind(self.alpha)*self.h/self.a)**2 + (sind(self.beta)*self.k/self.b)**2 + (sind(self.gamma)*self.l/self.c)**2
            p2 = ((2*self.k*self.l)/(self.b*self.c))*(cosd(self.beta) * cosd(self.gamma) - cosd(self.alpha))
            p3 = ((2*self.h*self.l)/(self.a*self.c))*(cosd(self.alpha) * cosd(self.gamma) - cosd(self.beta))
            p4 = ((2*self.k*self.h)/(self.b*self.a))*(cosd(self.beta) * cosd(self.alpha) - cosd(self.gamma))
            p5 = 1.0 - cosd(self.alpha)**2 - cosd(self.beta)**2 - cosd(self.gamma)**2 + 2*cosd(self.alpha)*cosd(self.beta)*cosd(self.gamma)
            
            self.d_star = np.sqrt((p1 + p2 + p3 + p4)/p5)
        return(self.d_star)
    
    def DScalc(self):
        
        self.ds = np.zeros((self.hkl.shape[0]))
        for ii in range(self.hkl.shape[0]):
            self.h, self.k, self.l = self.hkl[ii,0], self.hkl[ii,1], self.hkl[ii,2]
            self.ds[ii] = self.Dcalc()