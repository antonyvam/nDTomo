# -*- coding: utf-8 -*-
"""
Modified code from xfab.structure for reading crystal structure files (cif,pdb)
by Antony Vamvakeros
"""

from numpy import zeros, dot, pi
import logging


class atom_entry:
    def __init__(self, label=None, atomtype=None, pos=None,
                 adp_type=None, adp=None, occ=None, symmulti=None):
        self.label = label
        self.atomtype = atomtype
        self.pos = pos
        self.adp_type = adp_type
        self.adp = adp
        self.occ = occ
        self.symmulti = symmulti

class atomlist:
    def __init__(self, sgname=None, sgno=None, cell=None):
        self.sgname = sgname
        self.sgno = sgno
        self.cell = cell
        self.dispersion = {}
        self.atom = []
    def add_atom(self, label=None, atomtype=None, pos=None, 
                 adp_type=None, adp=None, occ=None, symmulti=None):
        self.atom.append(atom_entry(label=label, atomtype=atomtype,
                                    pos=pos, adp_type=adp_type,
                                    adp=adp, occ=occ, symmulti=symmulti))

class build_atomlist:
    def __init__(self):
        self.atomlist = atomlist()
        
    def CIFopen(self, ciffile=None, cifblkname=None):
        from CifFile import ReadCif # part of the PycifRW module
        try:
            # the following is a trick to avoid that urllib.URLopen.open
            # used by ReadCif misinterprets the url when a drive:\ is 
            # present (Win)
            ciffile = ciffile.replace(':','|')
            cf = ReadCif(ciffile)
        except:
            logging.error('File %s could not be accessed' %ciffile)        

        if cifblkname == None:   
            #Try to guess blockname                                                     
            blocks = list(cf.keys())
            if len(blocks) > 1:
                if len(blocks) == 2 and 'global' in blocks:
                    cifblkname = blocks[abs(blocks.index('global') - 1)]
                else:
                    logging.error('More than one possible data set:')
                    logging.error('The following data block names are in the file:')
                    for block in blocks:
                        logging.error(block)
                    raise Exception
            else:
                # Only one available
                cifblkname = blocks[0]
        #Extract block
        try:
            self.cifblk = cf[cifblkname]
        except:
            logging.error('Block - %s - not found in %s' % (blockname, ciffile))
            raise IOError
        return self.cifblk

    def PDBread(self, pdbfile = None):
        """
        function to read pdb file (www.pdb.org) and make 
        atomlist structure
        """
        from re import sub
        try:
            text = open(pdbfile, 'r').readlines()
        except:
            logging.error('File %s could not be accessed' % pdbfile)


        for i in range(len(text)):
            if text[i].find('CRYST1') == 0:
                a = float(text[i][6:15])
                b = float(text[i][15:24])
                c = float(text[i][24:33])
                alp = float(text[i][33:40])
                bet = float(text[i][40:47])
                gam = float(text[i][47:54])
                sg = text[i][55:66]

        self.atomlist.cell = [a, b, c, alp, bet, gam]

        # Make space group name
        sgtmp = sg.split()
        sg = ''
        for i in range(len(sgtmp)):
            if sgtmp[i] != '1':
                sg = sg + sgtmp[i].lower()
        self.atomlist.sgname = sg

        # Build SCALE matrix for transformation of 
        # orthonormal atomic coordinates to fractional
        scalemat = zeros((3, 4))
        for i in range(len(text)):
            if text[i].find('SCALE') == 0:
                # FOUND SCALE LINE
                scale = text[i].split()
                scaleline = int(scale[0][-1])-1
                for j in range(1, len(scale)):
                    scalemat[scaleline, j-1] = float(scale[j])
                
        no = 0
        for i in range(len(text)):
            if text[i].find('ATOM') == 0 or text[i].find('HETATM') ==0:
                no = no + 1 
                label = sub("\s+", "", text[i][12:16])
                atomtype = sub("\s+", "", text[i][76:78]).upper()
                x = float(text[i][30:38])
                y = float(text[i][38:46])
                z = float(text[i][46:54])
                # transform orthonormal coordinates to fractional
                pos = dot(scalemat, [x, y, z, 1])
                adp = float(text[i][60:66])/(8*pi**2) # B to U
                adp_type = 'Uiso'
                occ = float(text[i][54:60])
                multi = multiplicity(pos, self.atomlist.sgname)
                self.atomlist.add_atom(label=label,
                                       atomtype=atomtype,
                                       pos = pos,
                                       adp_type= adp_type,
                                       adp = adp,
                                       occ=occ,
                                       symmulti=multi)

                self.atomlist.dispersion[atomtype] = None


    def CIFread(self, ciffile = None, cifblkname = None, cifblk = None):
        from re import sub
        if ciffile != None:
            try:
                cifblk = self.CIFopen(ciffile=ciffile, cifblkname=cifblkname)
            except:
                raise 
        elif cifblk == None:
            cifblk = self.cifblk

        self.atomlist.cell = [self.remove_esd(cifblk['_cell_length_a']),
                              self.remove_esd(cifblk['_cell_length_b']),
                              self.remove_esd(cifblk['_cell_length_c']),
                              self.remove_esd(cifblk['_cell_angle_alpha']),
                              self.remove_esd(cifblk['_cell_angle_beta']),
                              self.remove_esd(cifblk['_cell_angle_gamma'])]

        #self.atomlist.sgname = upper(sub("\s+","",
        #                       cifblk['_symmetry_space_group_name_H-M']))
        self.atomlist.sgname = sub("\s+", "",
                                   cifblk['_symmetry_space_group_name_H-M'])

        self.atomlist.sgno = cifblk['_symmetry_Int_Tables_number']
        self.atomlist.formula = cifblk['_chemical_formula_structural']
        
        # Dispersion factors
        for i in range(len(cifblk['_atom_type_symbol'])):
            try:
                self.atomlist.dispersion[cifblk['_atom_type_symbol'][i].upper()] =\
                    [self.remove_esd(cifblk['_atom_type_scat_dispersion_real'][i]),
                     self.remove_esd(cifblk['_atom_type_scat_dispersion_imag'][i])]
            except:
                self.atomlist.dispersion[cifblk['_atom_type_symbol'][i].upper()] = None
                #logging.warning('No dispersion factors for %s in cif file - set to zero'\
                #                    %cifblk['_atom_type_symbol'][i])

        for i in range(len(cifblk['_atom_site_type_symbol'])):
            label = cifblk['_atom_site_label'][i]
            #atomno = atomtype[upper(cifblk['_atom_site_type_symbol'][i])]
            atomtype = cifblk['_atom_site_type_symbol'][i].upper()
            x = self.remove_esd(cifblk['_atom_site_fract_x'][i])
            y = self.remove_esd(cifblk['_atom_site_fract_y'][i])
            z = self.remove_esd(cifblk['_atom_site_fract_z'][i])
            try:
                adp_type = cifblk['_atom_site_adp_type'][i]
            except:
                adp_type = None
            try:
                occ = self.remove_esd(cifblk['_atom_site_occupancy'][i])
            except:
                occ = 1.0

            if '_atom_site_symmetry_multiplicity' in cifblk:
                multi = self.remove_esd(cifblk['_atom_site_symmetry_multiplicity'][i])
            # In old SHELXL versions this code was written
            # as '_atom_site_symetry_multiplicity'
            elif '_atom_site_symetry_multiplicity' in cifblk:
                multi = self.remove_esd(cifblk['_atom_site_symetry_multiplicity'][i])
            else:
                print('unknown site multiplicity')


            if adp_type == None:
                adp = 0.0
            elif adp_type == 'Uiso':
                adp = self.remove_esd(cifblk['_atom_site_U_iso_or_equiv'][i])
            elif adp_type == 'Uani':
                anisonumber = cifblk['_atom_site_aniso_label'].index(label)
                adp = [ self.remove_esd(cifblk['_atom_site_aniso_U_11'][anisonumber]),
                        self.remove_esd(cifblk['_atom_site_aniso_U_22'][anisonumber]),
                        self.remove_esd(cifblk['_atom_site_aniso_U_33'][anisonumber]),
                        self.remove_esd(cifblk['_atom_site_aniso_U_23'][anisonumber]),
                        self.remove_esd(cifblk['_atom_site_aniso_U_13'][anisonumber]),
                        self.remove_esd(cifblk['_atom_site_aniso_U_12'][anisonumber])]
            self.atomlist.add_atom(label=label, atomtype=atomtype,
                                   pos = [x, y, z], adp_type= adp_type, 
                                   adp = adp, occ=occ , symmulti=multi)

    def remove_esd(self, a):
        """                                                                         
        This function will remove the esd part of the entry,
        e.g. '1.234(56)' to '1.234'.
        """

        
        if a.find('(') == -1:
            value = float(a)
        else:
            value = float(a[:a.find('(')])
        return value


