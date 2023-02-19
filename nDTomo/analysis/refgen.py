# -*- coding: utf-8 -*-

import numpy as np


def genhkl_all(unit_cell, sintlmin, sintlmax, spg, cell_choice='standard', output_stl=False):
    """
	
    Generate the full set of reflections given a unit cell and space group up to maximum sin(theta)/lambda (sintlmax)
	
	The function is using the function genhkl_base for the actual generation 

    INPUT:  unit cell     : [a , b, c, alpha, beta, gamma]
            sintlmin      : minimum sin(theta)/lambda for generated reflections
            sintlmax      : maximum sin(theta)/lambda for generated reflections
            sgno/sgname   : provide either the space group number or its name 
                            e.g. sgno=225 or equivalently
                                 sgname='Fm-3m'
            output_stl    : Should sin(theta)/lambda be output (True/False)
                            default=False

    OUTPUT: list of reflections  (n by 3) or (n by 4) 
            if sin(theta)/lambda is chosen to be output

    The algorithm follows the method described in: 
    Le Page and Gabe (1979) J. Appl. Cryst., 12, 464-466
	    
    Henning Osholm Sorensen, University of Copenhagen, July 22, 2010.
    """
        
    H = genhkl_base(unit_cell, 
                      spg.syscond, 
                      sintlmin, sintlmax, 
                      crystal_system=spg.crystal_system, 
                      Laue_class = spg.Laue,
                      cell_choice = spg.cell_choice,
                      output_stl=True)

    Hall = np.zeros((0,4))
    # Making sure that the inversion element also for non-centrosymmetric space groups
    Rots = np.concatenate((np.array(spg.rot[:spg.nuniq]),-np.array(spg.rot[:spg.nuniq])))
    (dummy, rows) = np.unique((Rots*np.random.rand(3,3)).sum(axis=2).sum(axis=1),return_index=True)
    Rots = Rots[np.sort(rows)]


    for refl in H[:]:
        hkls = []
        stl = refl[3]
        for R in Rots:
            hkls.append(np.dot(refl[:3],R))
        a = np.array(hkls)
        (dummy, rows) = np.unique((a*np.random.rand(3)).sum(axis=1),
                                   return_index=True)
        Hsub = np.concatenate((a[rows], 
                             np.array([[stl]*len(rows)]).transpose()),
                            axis=1)
        Hall = np.concatenate((Hall,Hsub))

    if output_stl == False:
        return Hall[:,:3]
    else:
        return Hall
    
def genhkl_base(unit_cell, sysconditions, sintlmin, sintlmax, crystal_system='triclinic', Laue_class ='-1', cell_choice='standard', output_stl=None):
    """
	
    Generate the unique set of reflections for the cell up to maximum sin(theta)/lambda (sintlmax)

    The algorithm follows the method described in: 
    Le Page and Gabe (1979) J. Appl. Cryst., 12, 464-466
	
    INPUT:  unit cell     : [a , b, c, alpha, beta, gamma]
            sysconditions : conditions for systematic absent reflections
                            a 26 element list e.g. [0,0,2,0,0,0,0,0,.....,3] 
                            see help(sysabs) function for details.
            sintlmin      : minimum sin(theta)/lambda for generated reflections
            sintlmax      : maximum sin(theta)/lambda for generated reflections
            crystal_system: Crystal system (string), e.g. 'hexagonal'
            Laue class    : Laue class of the lattice (-1, 2/m, mmm, .... etc)
            cell_choice   : If more than cell choice can be made 
                            e.g. R-3 can be either rhombohedral or hexagonal  
            output_stl    : Should sin(theta)/lambda be output (True/False) default=False

    OUTPUT: list of reflections  (n by 3) or 
                                 (n by 4) if sin(theta)/lambda is chosen to be output
    
    Henning Osholm Sorensen, University of Copenhagen, July 22, 2010.
    """
    segm = None

    # Triclinic : Laue group -1
    if Laue_class == '-1':
        print('Laue class : -1', unit_cell)
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 0, 1, 0], [ 0, 0,  1]],
                        [[-1, 0,  1], [-1, 0, 0], [ 0, 1, 0], [ 0, 0,  1]],
                        [[-1, 1,  0], [-1, 0, 0], [ 0, 1, 0], [ 0, 0, -1]],
                        [[ 0, 1, -1], [ 1, 0, 0], [ 0, 1, 0], [ 0, 0, -1]]])
    
    # Monoclinic : Laue group 2/M 
    # unique a
    #segm = n.array([[[ 0, 0,  0], [ 0, 1, 0], [ 1, 0, 0], [ 0, 0,  1]],
    #                [[ 0,-1,  1], [ 0,-1, 0], [ 1, 0, 0], [ 0, 0,  1]]])

    # Monoclinic : Laue group 2/M 
    # unique b        
    if Laue_class == '2/m':
        print('Laue class : 2/m', unit_cell)
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 0, 1, 0], [ 0, 0,  1]],
                        [[-1, 0,  1], [-1, 0, 0], [ 0, 1, 0], [ 0, 0,  1]]])

    # unique c
    #segm = n.array([[[ 0, 0,  0], [ 1, 0, 0], [ 0, 0, 1], [ 0, 1,  0]],
    #                [[-1, 1,  0], [-1, 0, 0], [ 0, 0, 1], [ 0, 1,  0]]])

    # Orthorhombic : Laue group MMM
    if Laue_class == 'mmm':
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 0, 1, 0], [ 0, 0,  1]]])

    # Tetragonal 
    # Laue group : 4/MMM
    if Laue_class == '4/mmm':
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 1, 1, 0], [ 0, 0,  1]]])

    # Laue group : 4/M
    if Laue_class == '4/m':
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 1, 1, 0], [ 0, 0,  1]],
                        [[ 1, 2,  0], [ 1, 1, 0], [ 0, 1, 0], [ 0, 0,  1]]])

    # Hexagonal
    # Laue group : 6/MMM
    if Laue_class == '6/mmm':
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 1, 1, 0], [ 0, 0,  1]]])


    # Laue group : 6/M
    if Laue_class == '6/m':
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 1, 1, 0], [ 0, 0,  1]],
                        [[ 1, 2,  0], [ 0, 1, 0], [ 1, 1, 0], [ 0, 0,  1]]])

    # Laue group : -3M1
    if Laue_class == '-3m1':
        print('Laue class : -3m1 (hex)', unit_cell)
        if unit_cell[4]==unit_cell[5]:
            print('#############################################################')
            print('# Are you using a rhombohedral cell in a hexagonal setting? #')
            print('#############################################################')
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 1, 1, 0], [ 0, 0,  1]],
                        [[ 0, 1,  1], [ 0, 1, 0], [ 1, 1, 0], [ 0, 0,  1]]])

    # Laue group : -31M
    if Laue_class == '-31m':
        print('Laue class : -31m (hex)', unit_cell)
        if unit_cell[4]==unit_cell[5]:
            print('#############################################################')
            print('# Are you using a rhombohedral cell in a hexagonal setting? #')
            print('#############################################################')
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 1, 1, 0], [ 0, 0,  1]],
                        [[ 1, 1, -1], [ 1, 0, 0], [ 1, 1, 0], [ 0, 0, -1]]])

    # Laue group : -3
    if Laue_class == '-3' and cell_choice!='rhombohedral':
        print('Laue class : -3 (hex)', unit_cell)
        if unit_cell[4]==unit_cell[5]:
            print('#############################################################')
            print('# Are you using a rhombohedral cell in a hexagonal setting? #')
            print('#############################################################')
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 1, 1, 0], [ 0, 0,  1]],
                        [[ 1, 2,  0], [ 1, 1, 0], [ 0, 1, 0], [ 0, 0,  1]],
                        [[ 0, 1,  1], [ 0, 1, 0], [-1, 1, 0], [ 0, 0,  1]]])

    # RHOMBOHEDRAL
    # Laue group : -3M
    if Laue_class == '-3m' and cell_choice=='rhombohedral':
        print('Laue class : -3m (Rhom)', unit_cell)
        if unit_cell[4]!=unit_cell[5]:
            print('#############################################################')
            print('# Are you using a hexagonal cell in a rhombohedral setting? #')
            print('#############################################################')
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 1, 0,-1], [ 1, 1,  1]],
                        [[ 1, 1,  0], [ 1, 0,-1], [ 0, 0,-1], [ 1, 1,  1]]])

    # Laue group : -3
    if Laue_class == '-3' and cell_choice=='rhombohedral':
        print('Laue class : -3 (Rhom)', unit_cell)
        if unit_cell[4]!=unit_cell[5]:
            print('#############################################################')
            print('# Are you using a hexagonal cell in a rhombohedral setting? #')
            print('#############################################################')
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 1, 0,-1], [ 1, 1, 1]],
                        [[ 1, 1,  0], [ 1, 0,-1], [ 0, 0,-1], [ 1, 1, 1]],
                        [[ 0,-1, -2], [ 1, 0, 0], [ 1, 0,-1], [-1,-1, -1]],
                        [[ 1, 0, -2], [ 1, 0,-1], [ 0, 0,-1], [-1,-1,-1]]])

    #Cubic
    # Laue group : M3M
    if Laue_class == 'm-3m':
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 1, 1, 0], [ 1, 1,  1]]])

    # Laue group : M3
    if Laue_class == 'm-3':
        segm = np.array([[[ 0, 0,  0], [ 1, 0, 0], [ 1, 1, 0], [ 1, 1,  1]],
                        [[ 1, 2,  0], [ 0, 1, 0], [ 1, 1, 0], [ 1, 1,  1]]])

    if segm is None:
        print('No Laue class found')
        return False

    nref = 0
    H = np.zeros((0, 3))
    stl = np.array([])
    sintlH = 0.0
    
    #####################################################################################
    # The factor of 1.1 in the sintlmax criteria for setting htest=1, ktest=1 and ltest=1
    # was added based on the following observation for triclinic cells (Laue -3) with
    # rhombohedral setting:
    # [0,-5,-6]=[0,-1,-2]+4*[1,0,0]+0*[1,0,-1]+4*[-1,-1,-1],
    # but the algorithm is such that [-4,-5,-6]=[0,-1,-2]+4*[-1,-1,-1] is tested first,
    # and this has a slightly larger sintl than [0,-5,-6]
    # If these values are on opposide sides of sintlmax [0,-5,-6] is never generated.
    # This is a quick and dirty fix, something more elegant would be good!
    # Jette Oddershede, February 2013
    #####################################################################################
    sintl_scale = 1
    if Laue_class == '-3' and cell_choice=='rhombohedral':
        sintl_scale = 1.1

    for i in range(len(segm)):
        segn = i
        # initialize the identifiers
        htest = 0
        ktest = 0
        ltest = 0
        HLAST = segm[segn, 0, :]
        HSAVE = segm[segn, 0, :]
        HSAVE1 = segm[segn, 0, :]  #HSAVE1 =HSAVE
        sintlH = sintl(unit_cell, HSAVE)
        while ltest == 0:
            while ktest == 0:
                while htest == 0:
                    nref = nref + 1
                    if nref != 1:
                        ressss = sysabs(HLAST, sysconditions, crystal_system, cell_choice)
                        if sysabs(HLAST, sysconditions, crystal_system, cell_choice) == 0:
                            if  sintlH > sintlmin and sintlH <= sintlmax:
                                H = np.concatenate((H, [HLAST]))
                                stl = np.concatenate((stl, [sintlH]))
                        else: 
                            nref = nref - 1
                    HNEW = HLAST + segm[segn, 1, :]
                    sintlH = sintl(unit_cell, HNEW)
                    #if (sintlH >= sintlmin) and (sintlH <= sintlmax):
                    if sintlH <= sintlmax*sintl_scale:
                        HLAST = HNEW
                    else: 
                        htest = 1
#                        print HNEW,'htest',sintlH 
      
                HSAVE = HSAVE + segm[segn, 2, :]
                HLAST = HSAVE
                HNEW  = HLAST
                sintlH   = sintl(unit_cell, HNEW)
                if sintlH > sintlmax*sintl_scale:
                    ktest = 1
                htest = 0

            HSAVE1 = HSAVE1 + segm[segn, 3, :]
            HSAVE = HSAVE1
            HLAST = HSAVE1
            HNEW = HLAST
            sintlH = sintl(unit_cell, HNEW)
            if sintlH > sintlmax*sintl_scale:
                ltest = 1
            ktest = 0

    stl = np.transpose([stl])
    H = np.concatenate((H, stl), 1) # combine hkl and sintl
    H =  H[np.argsort(H, 0)[:, 3], :] # sort hkl's according to stl
    if output_stl == None:
        H = H[: , :3]
    return H

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

def sysabs(hkl, syscond, crystal_system='triclinic', cell_choice='standard'):
    """
    Defined as sysabs_unique with the exception that permutations in  
    trigonal and hexagonal lattices are taken into account.

	INPUT: hkl     : [h k l] 
           syscond : [1x26] with condition for systematic absences in this
                     space group, X in syscond should given as shown below
		   crystal_system : crystal system (string) - e.g. triclinic or hexagonal
		   
    OUTPUT: sysbs  : if 1 the reflection is systematic absent 
                     if 0 its not
    
    syscond:
    class        systematic abs               sysconditions[i]
    HKL          H+K=XN                            0
                 H+L=XN                            1
                 K+L=XN                            2
                 H+K,H+L,K+L = XN                  3
                 H+K+L=XN                          4
                 -H+K+L=XN                         5 
    HHL          H=XN                              6
                 L=XN                              7
                 H+L=XN                            8
                 2H+L=XN                           9
    0KL          K=XN                             10
                 L=XN                             11
                 K+L=XN                           12
    H0L          H=XN                             13
                 L=XN                             14
                 H+L=XN                           15
    HK0          H=XN                             16
                 K=XN                             17
                 H+K=XN                           18
    HH0          H=XN                             19
    H00          H=XN                             20
    0K0          K=XN                             21
    00L          L=XN                             22
    H-HL         H=XN                             23
                 L=XN                             24
                 H+L=XN                           25


    """

    sys_type = sysabs_unique(hkl, syscond)
    if cell_choice == 'rhombohedral':
        if sys_type == 0:
            h = hkl[1]
            k = hkl[2]
            l = hkl[0]
            sys_type = sysabs_unique([h, k, l], syscond)
            if sys_type == 0:
                h = hkl[2]
                k = hkl[0]
                l = hkl[1]
                sys_type = sysabs_unique([h, k, l], syscond)    
    elif crystal_system == 'trigonal' or crystal_system == 'hexagonal':
        if sys_type == 0:
            h = -(hkl[0]+hkl[1])
            k = hkl[0]
            l = hkl[2]
            sys_type = sysabs_unique([h, k, l], syscond)
            if sys_type == 0:
                h = hkl[1]
                k = -(hkl[0]+hkl[1])
                l = hkl[2]
                sys_type = sysabs_unique([h, k, l], syscond)

    return sys_type


def sysabs_unique(hkl, syscond):
    """
    sysabs_unique checks whether a reflection is systematic absent
    
    sysabs_unique = sysabs_unique(hkl,syscond)
     
    INPUT:  hkl     : [h k l] 
            syscond : [1x26] with condition for systematic absences in this
                      space group, X in syscond should given as shown below
    OUTPUT: sysbs   :  if 1 the reflection is systematic absent 
                       if 0 its not
    
    syscond:
    class        systematic abs               sysconditions[i]
    HKL          H+K=XN                            0
                 H+L=XN                            1
                 K+L=XN                            2
                 H+K,H+L,K+L = XN                  3
                 H+K+L=XN                          4
                 -H+K+L=XN                         5 
    HHL          H=XN                              6
                 L=XN                              7
                 H+L=XN                            8
                 2H+L=XN                           9
    0KL          K=XN                             10
                 L=XN                             11
                 K+L=XN                           12
    H0L          H=XN                             13
                 L=XN                             14
                 H+L=XN                           15
    HK0          H=XN                             16
                 K=XN                             17
                 H+K=XN                           18
    HH0          H=XN                             19
    H00          H=XN                             20
    0K0          K=XN                             21
    00L          L=XN                             22
    H-HL         H=XN                             23
                 L=XN                             24
                 H+L=XN                           25

    Henning Osholm Sorensen, June 23, 2006.
    """

    (h, k, l) = hkl
    sysabs_type = 0
    
    # HKL class
    if syscond[0] != 0:
        condition = syscond[0]
        if (abs(h+k))%condition !=0:
            sysabs_type = 1

    if syscond[1] != 0 :
        condition = syscond[1]
        if (abs(h+l))%condition !=0:
            sysabs_type = 2

    if syscond[2] != 0:
        condition = syscond[2]
        if (abs(k+l))%condition !=0:
            sysabs_type = 3

    if syscond[3] != 0:
        sysabs_type = 4
        condition = syscond[3]
        if (abs(h+k))%condition == 0:
            if (abs(h+l))%condition == 0:
                if  (abs(k+l))%condition == 0:
                    sysabs_type = 0

    if syscond[4] != 0:
        condition = syscond[4]
        if (abs(h+k+l))%condition != 0:
            sysabs_type = 5

    if syscond[5] != 0:
        condition = syscond[5]
        if (abs(-h+k+l))%condition != 0:
            sysabs_type = 6

    # HHL class
    if (h-k) == 0:
        if syscond[6] != 0 :
            condition = syscond[6]
            if (abs(h))%condition != 0:
                sysabs_type = 7
        if syscond[7] != 0:
            condition = syscond[7]
            if (abs(l))%condition != 0:
                sysabs_type = 8
        if syscond[8] != 0:
            condition = syscond[8]
            if (abs(h+l))%condition != 0:
                sysabs_type = 9
        if syscond[9] != 0:
            condition = syscond[9]
            if (abs(h+h+l))%condition != 0:
                sysabs_type = 10

    # 0KL class
    if h == 0:
        if syscond[10] != 0:
            condition = syscond[10]
            if (abs(k))%condition != 0:
                sysabs_type = 11
        if syscond[11] != 0:
            condition = syscond[11]
            if (abs(l))%condition != 0:
                sysabs_type = 12
        if syscond[12] != 0:
            condition = syscond[12]
            if (abs(k+l))%condition != 0:
                sysabs_type = 13

    # H0L class
    if k == 0:
        if syscond[13] != 0:
            condition = syscond[13]
            if (abs(h))%condition != 0:
                sysabs_type = 14
        if syscond[14] != 0:
            condition = syscond[14]
            if (abs(l))%condition != 0:
                sysabs_type = 15
        if syscond[15] != 0:
            condition = syscond[15]
            if (abs(h+l))%condition != 0:
                sysabs_type = 16


    # HK0 class
    if l == 0:
        if syscond[16] != 0:
            condition = syscond[16]
            if (abs(h))%condition != 0:
                sysabs_type = 17
        if syscond[17] != 0:
            condition = syscond[17]
            if (abs(k))%condition != 0:
                sysabs_type = 18
        if syscond[18] != 0:
            condition = syscond[18]
            if (abs(h+k))%condition != 0:
                sysabs_type = 19

    # HH0 class
    if l == 0:
        if h-k == 0:
            if syscond[19] != 0: 
                condition = syscond[19]
                if (abs(h))%condition != 0:
                    sysabs_type = 20

    # H00 class
    if abs(k)+abs(l) == 0:
        if syscond[20] != 0:
            condition = syscond[20]
            if (abs(h))%condition != 0:
                sysabs_type = 21

    # 0K0 class
    if abs(h)+abs(l) == 0:
        if syscond[21] != 0:
            condition = syscond[21]
            if (abs(k))%condition != 0:
                sysabs_type = 22

    # 00L class
    if abs(h)+abs(k) == 0:
        if syscond[22] != 0:
            condition = syscond[22]
            if (abs(l))%condition != 0:
                sysabs_type = 23

    # H-HL class
    if (h+k) == 0:
        if syscond[23] != 0 :
            condition = syscond[23]
            if (abs(h))%condition != 0:
                sysabs_type = 24
        if syscond[24] != 0:
            condition = syscond[24]
            if (abs(l))%condition != 0:
                sysabs_type = 25
        if syscond[25] != 0:
            condition = syscond[25]
            if (abs(h+l))%condition != 0:
                sysabs_type = 26


##### NEW CONDITION FOR R-3c
#     # H-H(0)L class
#     if -h==k:
#         print '>>>>>>>>>>>>>>>>> DO I EVER GET HERE <<<<<<<<<<<<<<<<<<<<'
#         if syscond[23] != 0:
#             condition = syscond[23]
        
#             if (h+l)%condition != 0 or l%2 !=0:
#                 sysabs_type = 24
    
    return sysabs_type