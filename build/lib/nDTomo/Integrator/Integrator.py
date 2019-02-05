# -*- coding: utf-8 -*-
"""
The main code for the Integrator GUI

@author: A. Vamvakeros
"""

from __future__ import unicode_literals
import sys
import os, h5py, json, getpass
#import numpy as np
from numpy import mean, zeros, sum, max, arange, ceil, interp, std, argmin, transpose, floor, tile, swapaxes, round, sqrt, where, empty, searchsorted

from matplotlib import use as u
u('Qt5Agg') 
#import matplotlib.pyplot as plt 
from PyQt5 import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

#% import nDTomo classes

from DCalibration import Calibration, CreatMask

from PDFmask import CreatPDFMask

from CreateAzimJson import CreatAzimint

from ScatterTomoInt import XRDCT_Squeeze

from ZigzagTomoInt_Live import XRDCT_LiveSqueeze
from ZigzagTomoInt_LiveRead import XRDCT_LiveRead
from ZigzagTomoInt_LiveMulti import XRDCT_ID15ASqueeze

from ContRotTomoInt_Live import Fast_XRDCT_LiveSqueeze
from ContRotTomoInt_LiveRead import Fast_XRDCT_LiveRead
from ContRotTomoInt_LiveMulti import Fast_XRDCT_ID15ASqueeze

        

class ApplicationWindow(QtWidgets.QMainWindow):
    
    """
    
    The Integrator GUI
    
    """
    
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.data = zeros(())
        self.sinos = zeros(())        
        self.calibrant = []
        self.poniname = []
        self.maskname = []
        self.mask = []
        self.jsonname = []
        self.npt_rad = []
        self.sess = "User"
        self.units = "2th_deg"
        self.savepath = []
        self.xrdctpath = []
        self.datatype = 'cbf'
        self.procunit = "CPU" 
        self.filt = "No"
        self.prc = 10
        self.thres = 3
        self.scantype = "Zigzag"
        self.parfile = []
        self.nt = []
        self.na = []
        self.dataset = []
        self.prefix = []
        self.c = 3E8
        self.h = 6.620700406E-34
        self.E = 0 #in keV
        self.liveoption = 0
        self.omega = 0
        self.trans = 0
        self.scansize = 0
        self.samplesize = 0
        self.stepsize = 0
        self.dio = []; self.etime = []
        self.xaxis = zeros(())
        
        self.username = getpass.getuser()
        print('User is %s' %self.username)
        
        
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("XRD-CT data integrator")

        self.left = 50
        self.top = 50
        self.width = 1024
        self.height = 1024
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Open Integration Parameters File', self.loadparfile, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.file_menu.addAction('&Save Integration Parameters File', self.saveparfile, QtCore.Qt.CTRL + QtCore.Qt.Key_S)
        self.file_menu.addAction('&Quit', self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.help_menu.addAction('&About', self.about)        
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.main_widget = QtWidgets.QWidget(self)        

        self.l = QtWidgets.QGridLayout(self.main_widget) #QGridLayout QVBoxLayout



        # set up the mapper
        self.mapperWidget = QtWidgets.QWidget(self)
        self.mapper = MyCanvas()
        self.mapperExplorerDock = QtWidgets.QDockWidget("Image", self)
        self.mapperExplorerDock.setWidget(self.mapperWidget)
        self.mapperExplorerDock.setFloating(False)
        self.mapperToolbar = NavigationToolbar(self.mapper, self)
        
        vbox1 = QtWidgets.QVBoxLayout()
        vbox1.addWidget(self.mapperToolbar)
        vbox1.addWidget(self.mapper)
        self.mapperWidget.setLayout(vbox1)
       
        #set up the plotter
        self.plotterWidget = QtWidgets.QWidget(self)
        self.plotter = MyCanvas()
        self.plotterExplorerDock = QtWidgets.QDockWidget("Histogram", self)
        self.plotterExplorerDock.setWidget(self.plotterWidget)
        self.plotterExplorerDock.setFloating(False)
        self.plotterToolbar = NavigationToolbar(self.plotter, self)
        
        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.addWidget(self.plotterToolbar)        
        vbox2.addWidget(self.plotter) # starting row, starting column, row span, column span
        self.plotterWidget.setLayout(vbox2)
        
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.mapperExplorerDock)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.plotterExplorerDock)
        
#        self.mapperExplorerDock.setVisible(0)
#        self.plotterExplorerDock.setVisible(0)

        # Detector calibration pannel
        
        self.label1 = QtWidgets.QLabel(self)
        self.label1.setText('Detector calibration parameters')
        self.l.addWidget(self.label1,0,1)
        
        self.label2 = QtWidgets.QLabel(self)
        self.label2.setText('X-ray energy (keV)')
        self.l.addWidget(self.label2,1,1)
        
        self.EnergySel = QtWidgets.QLineEdit(self)
        self.EnergySel.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.EnergySel.textChanged.connect(self.selEnergy)
        self.l.addWidget(self.EnergySel,1,2)      

        self.pbutton1 = QtWidgets.QPushButton("Calibrant diffraction pattern",self)
        self.pbutton1.clicked.connect(self.calibrantpath)
        self.l.addWidget(self.pbutton1,1,3)
        
        self.calibpath = QtWidgets.QLineEdit(self)
        self.calibpath.textChanged.connect(self.calibrantpath2)
        self.l.addWidget(self.calibpath,1,4)

        self.pbutton2 = QtWidgets.QPushButton("Create detector mask",self)
        self.pbutton2.clicked.connect(self.createmask)
        self.l.addWidget(self.pbutton2,1,5)

        self.pbutton2pdf = QtWidgets.QPushButton("Create detector mask for PDF",self)
        self.pbutton2pdf.clicked.connect(self.createPDFmask)
        self.l.addWidget(self.pbutton2pdf,1,6)
        
        self.pbutton3 = QtWidgets.QPushButton("Perform detector Calibration",self)
        self.pbutton3.clicked.connect(self.calib)
        self.l.addWidget(self.pbutton3,2,1)
        
        self.label4 = QtWidgets.QLabel(self)
        self.label4.setText('Number of radial points')
        self.l.addWidget(self.label4,2,2)

        self.radialpoints = QtWidgets.QLineEdit(self)
        self.radialpoints.setValidator(QtGui.QIntValidator())
        self.radialpoints.textChanged.connect(self.selRadial)
        self.l.addWidget(self.radialpoints,2,3)  
        
        self.pbutton2json = QtWidgets.QPushButton("Create .azimint.json file",self)
        self.pbutton2json.clicked.connect(self.createazimint)
        self.l.addWidget(self.pbutton2json,2,4)
        
#        self.labelse = QtWidgets.QLabel(self)
#        self.labelse.setText('Session')
#        self.l.addWidget(self.labelse,2,4)

#        self.ChooseSession = QtWidgets.QComboBox(self)
#        self.ChooseSession.addItems(["User Proposal","In-house beamtime"])
#        self.ChooseSession.currentIndexChanged.connect(self.Session)
#        self.l.addWidget(self.ChooseSession,2,5) 
        
        # XRD-CT data integration parameters pannel
        
        self.label3 = QtWidgets.QLabel(self)
        self.label3.setText('XRD-CT data integration parameters')
        self.l.addWidget(self.label3,3,1)
        
        self.pbutton4 = QtWidgets.QPushButton("Poni file path",self)
        self.pbutton4.clicked.connect(self.loadponifile)
        self.l.addWidget(self.pbutton4,4,1)

        self.ponipath = QtWidgets.QLineEdit(self)
        self.ponipath.textChanged.connect(self.loadponifile2)
        self.l.addWidget(self.ponipath,4,2)
        
        self.pbutton5 = QtWidgets.QPushButton("Detector mask path",self)
        self.pbutton5.clicked.connect(self.loadmaskfile)
        self.l.addWidget(self.pbutton5,4,3)
        
        self.calibmaskpath = QtWidgets.QLineEdit(self)
        self.calibmaskpath.textChanged.connect(self.loadmaskfile2)
        self.l.addWidget(self.calibmaskpath,4,4)
        
        self.pbutton6 = QtWidgets.QPushButton("json file path",self)
        self.pbutton6.clicked.connect(self.loadjsonfile)
        self.l.addWidget(self.pbutton6,4,5)
        
        self.jsonpath = QtWidgets.QLineEdit(self)
        self.jsonpath.textChanged.connect(self.loadjsonfile2)
        self.l.addWidget(self.jsonpath,4,6)
        
        self.pbutton7 = QtWidgets.QPushButton("Sinogram data save path",self)
        self.pbutton7.clicked.connect(self.selSavepath)
        self.l.addWidget(self.pbutton7,5,1)
        
        self.savedatapath = QtWidgets.QLineEdit(self)
        self.savedatapath.textChanged.connect(self.selSavepath2)
        self.l.addWidget(self.savedatapath,5,2)

        self.pbutton8 = QtWidgets.QPushButton("XRD-CT dataset path",self)
        self.pbutton8.clicked.connect(self.selXRDCTpath)
        self.l.addWidget(self.pbutton8,5,3)
        
        self.datapath = QtWidgets.QLineEdit(self)
        self.datapath.textChanged.connect(self.selXRDCTpath2)
        self.l.addWidget(self.datapath,5,4)

#        self.label5 = QtWidgets.QLabel(self)
#        self.label5.setText('X-axis units')
#        self.l.addWidget(self.label5,5,5)
#        
#        self.ChooseXAxis = QtWidgets.QComboBox(self)
#        self.ChooseXAxis.addItems(["2theta", "Q"])
#        self.ChooseXAxis.currentIndexChanged.connect(self.changeXAxis)
#        self.l.addWidget(self.ChooseXAxis,5,6) 

        self.labelIm = QtWidgets.QLabel(self)
        self.labelIm.setText('Image filetype')
        self.l.addWidget(self.labelIm,5,5)
        
        self.ChooseImageType = QtWidgets.QComboBox(self)
        self.ChooseImageType.addItems(["cbf", "edf"]) # Future version should include "hdf5", "lz4"
        self.ChooseImageType.currentIndexChanged.connect(self.ChooseDataType)
        self.l.addWidget(self.ChooseImageType,5,6) 
        
        self.label5 = QtWidgets.QLabel(self)
        self.label5.setText('Processing unit')
        self.l.addWidget(self.label5,6,1)
        
        self.ChooseProcUnit = QtWidgets.QComboBox(self)
        self.ChooseProcUnit.addItems(["ID15A GPU machine", "CPU", "GPU"])
        self.ChooseProcUnit.currentIndexChanged.connect(self.ProcessingUnit)
        self.l.addWidget(self.ChooseProcUnit,6,2)    
        
        self.label6 = QtWidgets.QLabel(self)
        self.label6.setText('Filters')
        self.l.addWidget(self.label6,6,3)
        
        self.ChooseFilters = QtWidgets.QComboBox(self)
        self.ChooseFilters.addItems(["No filters", "Median filter", "Trimmed mean filter",  "Standard deviation filter"])
        self.ChooseFilters.currentIndexChanged.connect(self.DecideFilter)
#        self.ChooseFilters.setEnabled(False)
        self.l.addWidget(self.ChooseFilters,6,4)     

        self.labelF = QtWidgets.QLabel(self)
        self.labelF.setText('Trimmed mean value')    
#        self.labelF.setVisible(False)
        self.l.addWidget(self.labelF,6,5)
 
        self.filtervalue = QtWidgets.QLineEdit(self)
        self.filtervalue.setValidator(QtGui.QIntValidator())
        self.filtervalue.textChanged.connect(self.selFilterValue)
        self.filtervalue.setEnabled(False)
        self.l.addWidget(self.filtervalue,6,6)    
        
        # XRD-CT scan parameters pannel

        self.label7 = QtWidgets.QLabel(self)
        self.label7.setText('XRD-CT scan parameters')
        self.l.addWidget(self.label7,7,1)

        self.label10 = QtWidgets.QLabel(self)
        self.label10.setText('Prefix')
        self.l.addWidget(self.label10,8,1)

        self.xrdctname = QtWidgets.QLineEdit(self)
        self.xrdctname.textChanged.connect(self.setdatasetname)
        self.l.addWidget(self.xrdctname,8,2)     

        self.label11 = QtWidgets.QLabel(self)
        self.label11.setText('Type of XRD-CT scan')
        self.l.addWidget(self.label11,8,3)
        
        self.ChooseScan = QtWidgets.QComboBox(self)
        self.ChooseScan.addItems(["Zigzag", "Continuous rotation", "Interlaced"])
        self.ChooseScan.currentIndexChanged.connect(self.ChooseScanType)
        self.l.addWidget(self.ChooseScan,8,4)  
        
        self.pbutton10 = QtWidgets.QPushButton("Load XRD-CT spec file",self)
        self.pbutton10.clicked.connect(self.readspecfile)
        self.l.addWidget(self.pbutton10,8,5)
        
        self.xrdctspecpath = QtWidgets.QLineEdit(self)
        self.l.addWidget(self.xrdctspecpath,8,6)
        
#        self.pbutton9 = QtWidgets.QPushButton("Load XRD-CT par file",self)
#        self.pbutton9.clicked.connect(self.readparfile)
#        self.l.addWidget(self.pbutton9,8,7)
        
#        self.xrdctparpath = QtWidgets.QLineEdit(self)
#        self.l.addWidget(self.xrdctparpath,8,8)

        self.label8 = QtWidgets.QLabel(self)
        self.label8.setText('Number of translation steps')
        self.l.addWidget(self.label8,9,1)

        self.setnt = QtWidgets.QLineEdit(self)
        self.setnt.setValidator(QtGui.QIntValidator())
        self.setnt.textChanged.connect(self.setnumtra)
        self.l.addWidget(self.setnt,9,2)
        
        self.label9 = QtWidgets.QLabel(self)
        self.label9.setText('Number of projections')
        self.l.addWidget(self.label9,9,3)

        self.setna = QtWidgets.QLineEdit(self)
        self.setna.setValidator(QtGui.QIntValidator())
        self.setna.textChanged.connect(self.setnumang)
        self.l.addWidget(self.setna,9,4)        
        
        self.label12 = QtWidgets.QLabel(self)
        self.label12.setText('OR')
        self.l.addWidget(self.label12,10,1)

        self.label13 = QtWidgets.QLabel(self)
        self.label13.setText('Scan range (mm)')
        self.l.addWidget(self.label13,11,1)

        self.setscanrange = QtWidgets.QLineEdit(self)
        self.setscanrange.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.setscanrange.textChanged.connect(self.setscans)
        self.l.addWidget(self.setscanrange,11,2)
        
        self.label14 = QtWidgets.QLabel(self)
        self.label14.setText('Sample size (mm)')
        self.l.addWidget(self.label14,11,3)

        self.setsamplesize = QtWidgets.QLineEdit(self)
        self.setsamplesize.setValidator(QtGui.QDoubleValidator(0.99,99.99,2))
        self.setsamplesize.textChanged.connect(self.setsamps)
        self.l.addWidget(self.setsamplesize,11,4)        

        self.label15 = QtWidgets.QLabel(self)
        self.label15.setText('Step size (microns)')
        self.l.addWidget(self.label15,11,5)

        self.setstepsize = QtWidgets.QLineEdit(self)
        self.setstepsize.setValidator(QtGui.QIntValidator())
        self.setstepsize.textChanged.connect(self.setsteps)
        self.l.addWidget(self.setstepsize,11,6)  
        
        # XRD-CT data integration pannel
        
        self.label16 = QtWidgets.QLabel(self)
        self.label16.setText('Perform XRD-CT data integration')
        self.l.addWidget(self.label16,12,1)
        
        self.ChooseIntegration = QtWidgets.QComboBox(self)
        self.ChooseIntegration.addItems(["Data integration", "Data integration with live plotting"])
        self.ChooseIntegration.currentIndexChanged.connect(self.doliveplot)
        self.l.addWidget(self.ChooseIntegration,13,1)  
        
        self.pbutton11 = QtWidgets.QPushButton("Start Integration",self)
        self.pbutton11.clicked.connect(self.processxrdct)
        self.l.addWidget(self.pbutton11,14,1)
        
        self.pbutton12 = QtWidgets.QPushButton("Stop Integration",self)
        self.pbutton12.clicked.connect(self.stopprocessxrdct)
        self.pbutton12.setEnabled(True)
        self.l.addWidget(self.pbutton12,14,2)
        
        self.progressbar = QtWidgets.QProgressBar(self)
        self.l.addWidget(self.progressbar,14,3)


        self.pbuttonrec = QtWidgets.QPushButton("Reconstruct",self)
#        self.pbuttonrec.clicked.connect(self.fbprec_vol)
        self.pbuttonrec.setEnabled(False)
#        self.pbuttonrec.setMaximumWidth(150)
        self.l.addWidget(self.pbuttonrec,14,4)

        
        self.progressbarrec = QtWidgets.QProgressBar(self)
#        self.progressbarrec.setMaximumWidth(150)
        self.l.addWidget(self.progressbarrec,14,5)

        self.ChooseData = QtWidgets.QComboBox(self)
        self.ChooseData.addItems(["Display sinogram data", "Display reconstructed data"])
        self.ChooseData.currentIndexChanged.connect(self.changedata)
        self.ChooseData.setEnabled(False)
        self.ChooseData.setMaximumWidth(170)
        self.l.addWidget(self.ChooseData,14,6)   
        

        self.label17 = QtWidgets.QLabel(self)
        self.label17.setText('Perform batch XRD-CT data integration')
        self.l.addWidget(self.label17,15,1)

        self.pbutton13 = QtWidgets.QPushButton("Add XRD-CT dataset",self)
        self.pbutton13.clicked.connect(self.selXRDCTdata)
        self.l.addWidget(self.pbutton13,16,1)

        self.pbutton14 = QtWidgets.QPushButton("Remove XRD-CT dataset",self)
        self.pbutton14.clicked.connect(self.remXRDCTdata)
        self.l.addWidget(self.pbutton14,17,1)
        
        self.datalist = QtWidgets.QListWidget(self)
        self.l.addWidget(self.datalist,16,2,17,1)

        self.pbutton15 = QtWidgets.QPushButton("Perform batch integration",self)
        self.pbutton15.clicked.connect(self.batchXRDCTdata)
        self.l.addWidget(self.pbutton15,16,3)

        self.pbutton16 = QtWidgets.QPushButton("Stop Integration",self)
        self.pbutton16.clicked.connect(self.batchstopprocessxrdct)
        self.pbutton16.setEnabled(True)
        self.l.addWidget(self.pbutton16,16,4)
        
#        self.progressbarbatch = QtWidgets.QProgressBar(self)
#        self.l.addWidget(self.progressbarbatch,16,5)
        
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        
        self.selectedChannels = empty(0,dtype=object)
        
        
        ####### The methods #######
        
    def calibrantpath(self):
#        (self, "Select File", "", "*.png *.jpg")
        calibrant_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open calibrant diffraction image', "", "*.cbf *.edf")
        self.calibrant = calibrant_fileName
        self.calibpath.setText(self.calibrant)
        print self.calibrant
        
    def calibrantpath2(self,s):
        self.calibrant = str(s)
        print self.calibrant
        
    def selEnergy(self,s):
        self.E = float(s)
    
    def calib(self):
        self.createdetcalib = Calibration(self.calibrant,self.E)
        self.createdetcalib.start()            

    def createmask(self):
        self.createdetmask = CreatMask(self.calibrant)
        self.createdetmask.start()
        
    def loadponifile(self):
        poni_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open poni file', "", "*.poni")
        self.poniname = poni_fileName
        self.ponipath.setText(self.poniname)        

    def loadponifile2(self,s):
        self.poniname = str(s)
        print self.poniname
        
    def loadmaskfile(self):
        mask_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open detector mask', "", "*.cbf *.edf")
        self.maskname = mask_fileName
        self.calibmaskpath.setText(self.maskname)    

    def loadmaskfile2(self,s):
        self.maskname = str(s)
        print self.maskname

    def createPDFmask(self):
        self.createdetpdfmask = CreatPDFMask(self.poniname,self.maskname)
        self.createdetpdfmask.start()

    def selRadial(self,s):
        self.npt_rad = s
        
    def createazimint(self):
        self.createdetazimint = CreatAzimint(self.poniname,self.maskname,self.npt_rad)
        self.createdetazimint.start()        
        
#        self.jsonname = self.createdetazimint.jsonname
#        print self.jsonname
#        self.jsonpath.setText(self.jsonname) 
        
        
    def loadjsonfile(self):
#        json_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open json file', "", "*.json")
#        self.jsonname = json_fileName
#        self.jsonpath.setText(self.jsonname)          

        jsonpath = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select directory of .azimint.json file"))
        self.jsonname = '%s/.azimint.json' %(jsonpath)
        print self.jsonname
        self.jsonpath.setText(self.jsonname)  
        
    def loadjsonfile2(self,s):
        self.jsonname = str(s)
        print self.jsonname
        
    def selSavepath(self):
        savepath = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select directory to save sinogram data"))
        self.savepath = savepath
        self.savedatapath.setText(self.savepath)  

    def selSavepath2(self,s):
        self.savepath = str(s)
        print self.savepath
        
    def selXRDCTpath(self):

        xrdctpath = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select XRD-CT data directory"))
        if len(xrdctpath)>0:
            self.xrdctpath = xrdctpath
            self.datapath.setText(self.xrdctpath)          
            
            prefix = xrdctpath.split("xrdct/")
            self.dataset = prefix[1]
            self.xrdctname.setText(self.dataset)
            
#            self.parfile = '%s/%s.par' %(self.xrdctpath, self.dataset)
#            print self.parfile      
#            
#            try:
#                self.readparinfo()
#                self.xrdctparpath.setText(self.parfile)
#            except: 
#                print '.par file does not exist or prefix is different'

            self.specfile = '%s/%s.spec' %(self.xrdctpath, self.dataset)
            print self.specfile  
            
            try:
                self.readspecinfo()
                self.xrdctspecpath.setText(self.specfile)
            except: 
                print '.spec file does not exist or prefix is different'    
                
            self.xrdctpath = "%sxrdct/" %(prefix[0])
            print self.xrdctpath  

        
    def selXRDCTpath2(self,s):
        self.xrdctpath = str(s)
        print self.xrdctpath
        

    def changeXAxis(self,ind):

        if ind == 0:
            self.units = "2th_deg"
        elif ind == 1:
            self.units = "q_A^-1"
#        self.units  = '"%s"' %self.units
        print self.units
    
#    def Session(self,ind):
#        if ind == 0:
#            self.sess = "User"
#        elif ind == 1:
#            self.sess = "Inhouse"    
#        print self.sess  
            
    def ChooseDataType(self,ind):
        if ind == 0:
            self.datatype = "cbf"
        elif ind == 1:
            self.datatype = "edf"

        print self.datatype   
         
    def ProcessingUnit(self,ind):
        
        if ind == 0:
            self.procunit = "MultiGPU"            
        elif ind == 1:
            self.procunit = "CPU"        
        elif ind == 2:
            self.procunit = "GPU"       
            
        if self.procunit == "MultiGPU":
            try:
                prefix = self.calibrant.split("/gz")
                if len(prefix)==1:
                    self.calibrant = "/gz%s" %(self.calibrant)
                    self.calibpath.setText(self.calibrant)

                prefix = self.poniname.split("/gz")
                if len(prefix)==1:                
                    self.poniname = "/gz%s" %(self.poniname)
                    self.ponipath.setText(self.poniname)    

                prefix = self.maskname.split("/gz")
                if len(prefix)==1:                
                    self.maskname = "/gz%s" %(self.maskname)
                    self.calibmaskpath.setText(self.maskname)   
                
                prefix = self.savepath.split("/gz")
                if len(prefix)==1:
                    self.savepath = "/gz%s" %(self.savepath)
                    self.savedatapath.setText(self.savepath)  

                prefix = self.jsonname.split("/gz")
                if len(prefix)==1:                
                    self.jsonname = "/gz%s" %(self.jsonname)
                    self.jsonpath.setText(self.jsonname) 
                
                prefix = self.xrdctpath.split("/gz")
                if len(prefix)==1:                
                    self.xrdctpath = "/gz%s" %(self.xrdctpath)
                    self.datapath.setText(self.xrdctpath)          

                prefix = self.specfile.split("/gz")
                if len(prefix)==1:                
                    self.specfile = "/gz%s" %(self.specfile)
                    self.xrdctspecpath.setText(self.specfile)
                
            except:
                print "There might be a problem with the directories"
                
        else:
            try:
                prefix = self.calibrant.split("/gz")
                if len(prefix)==2:
                    self.calibrant = prefix[1]
                    self.calibpath.setText(self.calibrant)

                prefix = self.poniname.split("/gz")
                if len(prefix)==2:                
                    self.poniname = prefix[1]
                    self.ponipath.setText(self.poniname)    

                prefix = self.maskname.split("/gz")
                if len(prefix)==2:                
                    self.maskname = prefix[1]
                    self.calibmaskpath.setText(self.maskname)   
                
                prefix = self.savepath.split("/gz")
                if len(prefix)==2:
                    self.savepath = prefix[1]
                    self.savedatapath.setText(self.savepath)  

                prefix = self.jsonname.split("/gz")
                if len(prefix)==2:                
                    self.jsonname = prefix[1]
                    self.jsonpath.setText(self.jsonname) 
                
                prefix = self.xrdctpath.split("/gz")
                if len(prefix)==2:                
                    self.xrdctpath = prefix[1]
                    self.datapath.setText(self.xrdctpath)          

                prefix = self.specfile.split("/gz")
                if len(prefix)==2:                
                    self.specfile = prefix[1]
                    self.xrdctspecpath.setText(self.specfile)
                
            except:
                print "There might be a problem with the directories"            
            
            
            
    def DecideFilter(self,ind):
        if ind == 0:
            self.filt = "No"
            self.labelF.setEnabled(False)
            self.filtervalue.setEnabled(False)
        elif ind == 1:
            self.filt = "Median"        
            self.labelF.setEnabled(False)
            self.filtervalue.setEnabled(False)
        elif ind == 2:
            self.filt = "trimmed_mean"     
            
            self.labelF.setText('Trimmed mean value (%)')  
            self.labelF.setEnabled(True)
            self.filtervalue.setEnabled(True)
            
        elif ind == 3:
            self.filt = "sigma"       
            
            self.labelF.setText('Sigma threshold value')  
            self.labelF.setEnabled(True)
            self.filtervalue.setEnabled(True)
            
    def selFilterValue(self,s):
        
        if self.filt == "trimmed_mean":
            self.prc = s
            print self.prc
        elif self.filt == "sigma":
            self.thres = s
            print self.thres
            
    def ChooseScanType(self,ind):
        if ind == 0:
            self.scantype = "Zigzag"
        elif ind == 1:
            self.scantype = "ContRot"
        elif ind == 2:
            self.scantype = "Interlaced"            

    def readspecfile(self): 
        
        spec_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', "", "*.spec")
        if len(spec_fileName)>0:
            self.specfile = spec_fileName
            print self.specfile
            self.xrdctspecpath.setText(self.specfile)
            self.readspecinfo()

            prefix = spec_fileName.split(".spec")
            prefix = prefix[0]
            prefix = prefix.split("/")
            self.dataset = prefix[-1]
            self.xrdctname.setText(self.dataset)
            
    def readspecinfo(self): 
        
        """
        Method for reading .spec files that contain the motor positions during the tomographic scan
        """
        
        if self.scantype=='Zigzag':
            try:
                f=open(self.specfile).read()
                nomega=f.count('#S')/2
#                print nomega
                lines=f.split('\n')
#                print len(lines)
                for line in lines:
#                    print line
                    if '#S' in line:
                        ny=int(line.split()[8])
#                        print ny
                        break
                omega=empty(ny*nomega)
                y=empty(ny*nomega)
                dio=empty(ny*nomega)
                etime=empty(ny*nomega)
                n=0
                for i in range(0,len(lines)):
                    if '#L' in lines[i]:
                        if lines[i].split()[1] == 'TrigTime':
                            for j in range(i+1,i+1+ny):
                                args=lines[j].split()
                                omega[n]=(float(args[1])+float(args[2]))/2.0
                                y[n]=(float(args[3])+float(args[4]))/2.0
                                dio[n]=float(args[5])
                                etime[n]=float(args[7])
                                n=n+1
#                self.nt = len(y)
#                self.na = len(omega)
                print nomega, ny
                self.omega = omega
                self.trans = y
                self.nomega = nomega
                self.ny = ny
                self.dio = dio
                self.etime = etime
                
            except:
                print 'Can not find spec file',self.specfile,'skipping directory'

        elif self.scantype=='ContRot':
            try:
                with open(self.specfile) as f:
                    for line in f:
                        if '#S' in line:
                            npts=int(line.split()[8])
                            nomega=int(180.0/float(line.split()[4]))
                            ny = int((npts-1)/nomega)
                            omega=empty(npts)
                            y=empty(npts)
                            dio=empty(npts)
                            etime=empty(npts)                            
                        if '#L' in line:
                            if line.split()[1] == 'TrigTime':
                                break
                    n=0
                    for line in f:
                        args=line.split()
                        omega[n]=(float(args[1])+float(args[2]))/2.0 % 360.0
                        y[n]=(float(args[3])+float(args[4]))/2.0
                        dio[n]=float(args[5])
                        etime[n]=float(args[7])                        
                        n=n+1

#                self.nt = len(y)
#                self.na = len(omega)
                print nomega, ny
                self.omega = omega
                self.trans = y
                self.nomega = nomega
                self.ny = ny+1
                self.dio = dio
                self.etime = etime                
            except:
                print 'Cannot find spec file',self.specfile,'skipping directory'


        self.na = self.nomega
        self.nt = self.ny
        self.setnt.setText(str(self.nt))
        self.setna.setText(str(self.na))
            
    def setnumtra(self,s):
        self.nt = int(s)
        print self.nt
        self.calcpar2()
        
    def setnumang(self,s):
        self.na = int(s)
        print self.na
        self.calcpar2()
        
    def setdatasetname(self,s):
        self.dataset = str(s)
        print self.dataset

    def setscans(self,s):
        self.scansize = float(s)
        print self.scansize
        self.calcpar()
        
    def setsamps(self,s):
        self.samplesize = float(s)
        print self.samplesize
        self.calcpar()
        
    def setsteps(self,s):
        self.stepsize = float(s)
        print self.stepsize
        
    def calcpar(self):
        try:
            self.nt = self.scansize*1E3/float(self.stepsize) + 1
            self.na = self.samplesize*1E3/float(self.stepsize) + 1
            self.setnt.setText(str(int(self.nt)));self.setna.setText(str(int(self.na)));
        except: 
            pass
            
    def calcpar2(self):
        try:
            self.scansize = (self.nt - 1) * float(self.stepsize)/1E3
            self.samplesize = (self.na - 1) * float(self.stepsize)/1E3
            self.setscanrange.setText(str(self.scansize));self.setsamplesize.setText(str(self.samplesize));
        except:
            pass
                
    def doliveplot(self,s):
        if s == 0:
            self.liveoption = 0
            self.mapperExplorerDock.setVisible(0)
            self.plotterExplorerDock.setVisible(0)
        elif s == 1:
            self.liveoption = 1
            self.mapperExplorerDock.setVisible(1)
            self.plotterExplorerDock.setVisible(1)  
                    
    def processxrdct(self):
        
        """
        
        Perform the XRD-CT data integration
        
        """
        
        self.ChooseIntegration.setEnabled(False)
        self.pbutton11.setEnabled(False)   
        self.pbutton15.setEnabled(False)  
        
        self.prefix = self.dataset
        na = float(self.na); nt = float(self.nt);npt_rad = float(self.npt_rad);
        
        try:
            if self.scantype == 'Zigzag':

                if self.liveoption == 1:
                    self.Squeezing = XRDCT_LiveSqueeze(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.jsonname,self.omega,self.trans,self.dio,self.etime)
                    self.Squeezing.start()  
                    self.Reading = XRDCT_LiveRead(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.jsonname,self.omega,self.trans,self.dio,self.etime)
                    self.Reading.start()  
                    self.Reading.exploredata.connect(self.explore)
                    self.Reading.updatedata.connect(self.update)
                    
#                    self.Squeezing.squeeze.connect(self.writeintdata)
                    
                elif self.liveoption == 0:
                    if self.procunit == "MultiGPU":
                        self.Squeezing = XRDCT_ID15ASqueeze(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.jsonname,self.omega,self.trans,self.dio,self.etime)
                        self.Squeezing.start()
                        self.Squeezing.liveRead()
                    else:
                        self.Squeezing = XRDCT_Squeeze(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.omega,self.trans,self.dio,self.etime)
                        self.Squeezing.start()
#                    self.Squeezing.squeeze.connect(self.writeintdata)
                    
                self.Squeezing.progress.connect(self.progressbar.setValue)
                    

            elif self.scantype == 'ContRot':

                if self.liveoption == 1:
                    self.Squeezing = Fast_XRDCT_LiveSqueeze(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.jsonname,self.omega,self.trans,self.dio,self.etime)
                    self.Squeezing.start()  
                    self.Reading = Fast_XRDCT_LiveRead(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.jsonname,self.omega,self.trans,self.dio,self.etime)
                    self.Reading.start()  
                    self.Reading.exploredata.connect(self.explore)
                    self.Reading.updatedata.connect(self.update)
                    
                elif self.liveoption == 0:
                    if self.procunit == "MultiGPU":
                        self.Squeezing = Fast_XRDCT_ID15ASqueeze(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.jsonname,self.omega,self.trans,self.dio,self.etime)
                        self.Squeezing.start()
                        self.Squeezing.liveRead()
                    else:
                        self.Squeezing = XRDCT_Squeeze(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.omega,self.trans,self.dio,self.etime)
                        self.Squeezing.start()
                    self.Squeezing.squeeze.connect(self.writeintdata)
                    
                self.Squeezing.progress.connect(self.progressbar.setValue)
                
            if self.progressbar.value() == 100:
                self.pbuttonrec.setEnabled(True)
                
        except:
            pass
        
    
    def readxrdct(self):

        """
        
        Read the integrated XRD-CT data
        
        """
        
        self.ChooseIntegration.setEnabled(False)
        self.pbutton11.setEnabled(False)   
        self.pbutton15.setEnabled(False)          
        self.prefix = self.dataset
        na = float(self.na); nt = float(self.nt);npt_rad = float(self.npt_rad);
        try:
            if self.scantype == 'Zigzag':
                    self.Reading = XRDCT_LiveRead(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.jsonname,self.omega,self.trans,self.dio,self.etime)
                    self.Reading.start()  
                    self.Reading.exploredata.connect(self.explore)
                    self.Reading.updatedata.connect(self.update)
            elif self.scantype == 'ContRot':                    
                    self.Reading = Fast_XRDCT_LiveRead(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.jsonname,self.omega,self.trans,self.dio,self.etime)
                    self.Reading.start()  
                    self.Reading.exploredata.connect(self.explore)
                    self.Reading.updatedata.connect(self.update)
            
            self.Squeezing.progress.connect(self.progressbar.setValue)
                
            if self.progressbar.value() == 100:
                self.pbuttonrec.setEnabled(True)
                
        except:
            pass
        
    
    def stopprocessxrdct(self):
            
        if self.liveoption == 1:
            try:
                self.Squeezing.periodEventraw.stop()
                self.Squeezing.terminate()            
            except:
                print 'No data being integrated'
                
            try:
                self.Reading.periodEvent.stop()
                self.Reading.terminate() 
            except:
                print 'No data being read'        
        
        elif self.liveoption == 0 and self.procunit == "MultiGPU":
            self.Squeezing.periodEvent.stop()
            self.Squeezing.periodEventraw.stop()
            self.Squeezing.terminate()            
            
        self.Squeezing.terminate()
        self.ChooseIntegration.setEnabled(True)
        self.pbutton11.setEnabled(True)   
        self.pbutton15.setEnabled(True)        
        self.progressbar.setValue(0)
        

    def explore(self):
        self.data = self.Reading.data
        self.xaxis = self.Reading.q
        self.xaxislabel = 'Q'
        
        self.map_data = self.mapper.axes.imshow(mean(self.data,axis=2),cmap='jet')
        title = 'Mean Sinogram'
        self.mapper.axes.set_title(title, fontstyle='italic')
        self.mapper.fig.canvas.mpl_connect('button_press_event', self.onMapClick)
        self.mapper.fig.canvas.mpl_connect('motion_notify_event', self.onMapMoveEvent)        
        
        
        self.cb = self.mapper.fig.colorbar(self.map_data)
        
        self.mapper.show()
        self.mapper.draw()  
        
        self.mdp = mean(mean(self.data,axis=1),axis=0)
        self.sdp = sum(sum(self.data,axis=1),axis=0)
        self.dproi = self.mdp
        
        self.histogramCurve = self.plotter.axes.plot(self.xaxis, self.mdp, label='Mean diffraction pattern', color="b")
        self.activeCurve = self.plotter.axes.plot(self.xaxis, self.mdp, label='Mean diffraction pattern', color="r")
        
        ########
        self.vCurve = self.plotter.axes.axvline(x=0, color="k")
        #######
        
#        self.selectedDataSetList.addItem('mean')
        self.plotter.axes.legend()
        self.plotter.axes.set_title("Mean diffraction pattern", fontstyle='italic')
#        self.activeCurve[0].set_visible(False)
#        self.plot_fig.canvas.mpl_connect('button_press_event',self.onPlotClick)        
#        self.plot_fig.canvas.mpl_connect('motion_notify_event',self.onPlotMoveEvent)
#        self.plot_cid = self.plotter.figure.canvas.mpl_connect('motion_notify_event',self.onPlotMoveEvent)
        
        self.plotter.fig.canvas.mpl_connect('motion_notify_event', self.onPlotClick)       
        self.plot_cid = self.plotter.fig.canvas.mpl_connect('motion_notify_event', self.onPlotMoveEvent)   
        self.plotter.show()        
        self.plotter.draw()        
                
        
    def onMapClick(self, event):
        if event.button == 1:
            if event.inaxes:
                self.col = int(event.xdata.round())
                self.row = int(event.ydata.round())
                self.dproi = self.data[self.row,self.col,:]
                self.histogramCurve[0].set_data(self.xaxis, self.dproi) 
                self.histogramCurve[0].set_label(str(self.row)+','+str(self.col))
                self.histogramCurve[0].set_visible(True)          
                self.plotter.axes.legend()
                
#                self.plotter.axes.set_xlim(self.xaxis[0],self.xaxis[-1])
#                self.plotter.axes.set_ylim(0,np.max(self.dproi))
                self.plotter.show()        
                self.plotter.draw()

#                self.selectedDataSetList.addItem(np.str([row,col]))
#                self.selectedDataSetList.setCurrentIndex(self.selectedDataSetList.count()-1)
#                self.selectedDataSetList.update()
                
#            self.plot_fig.canvas.mpl_disconnect(self.plot_cid) 
#            self.plot_cid = self.plot_fig.canvas.mpl_connect('motion_notify_event',self.onPlotMoveEvent)

            else:
                self.selectedVoxels = empty(0,dtype=object)
#                self.selectedDataSetList.clear()
#                self.currentCurve = 0
                self.plotter.axes.clear() # not fast
                self.plotter.axes.plot(self.xaxis, self.mdp, label='Mean diffraction pattern')
#                self.selectedDataSetList.addItem('mean')
                self.plotter.axes.legend()                
                self.plotter.draw() 

        if event.button == 3:
            if event.inaxes:
                self.histogramCurve[0].set_visible(False)          
                
                self.plotter.axes.set_xlim(self.xaxis[0],self.xaxis[-1])
                self.plotter.axes.set_ylim(0,max(self.dproi))
                self.plotter.show()        
                self.plotter.draw()

    def onMapMoveEvent(self, event): # SDMJ version
        if event.inaxes:
            
            col = int(event.xdata.round())
            row = int(event.ydata.round())
            
            dproi = self.data[row,col,:]
            self.activeCurve[0].set_data(self.xaxis, dproi) 
            self.mapper.axes.relim()
            self.activeCurve[0].set_label(str(row)+','+str(col))
#            self.mapper.axes.autoscale(enable=True, axis='both', tight=True)#.axes.autoscale_view(False,True,True)
            self.activeCurve[0].set_visible(True)
            self.plotter.axes.set_ylim(0,max(dproi))

        else:
            self.activeCurve[0].set_visible(False)
            self.activeCurve[0].set_label('')
            
#            self.plotter.axes.set_xlim(self.xaxis[0],self.xaxis[-1])
            self.plotter.axes.set_ylim(0,max(self.dproi))
            
        self.plotter.axes.legend()
        self.plotter.draw()
    
    
    def onPlotMoveEvent(self, event):
        if event.inaxes:
            
            x = event.xdata
            
            if self.xaxislabel == 'd':
                self.nx = len(self.xaxis) - searchsorted(self.xaxisd, [x])[0]
            else:
                self.nx = searchsorted(self.xaxis, [x])[0] -1
            
            if self.nx<0:
                self.nx = 0
            elif self.nx>len(self.xaxis):
                self.nx = len(self.xaxis)-1

            self.selectedChannels = self.nx;
            self.mapper.axes.clear() # not fast
            self.imoi = self.data[:,:,self.nx]
            self.map_data = self.mapper.axes.imshow(self.imoi,cmap='jet')
            title = "Channel = %d; %s = %.3f" % (self.nx, self.xaxislabel, self.xaxis[self.nx])
            self.mapper.axes.set_title(title)

            ############       
#            self.mapper.fig.delaxes(self.mapper.fig.axes[1])
#            self.cb.remove()
#            self.cb = self.mapper.fig.colorbar(self.map_data)
            ############
            
            self.mapper.draw()             
            
            self.vCurve.set_xdata(x) 
                
            self.plotter.draw()

    def onPlotClick(self, event):

        if event.button == 1:
            self.plotter.fig.canvas.mpl_disconnect(self.plot_cid)   

        elif event.button == 3:
            self.plot_cid = self.plotter.fig.canvas.mpl_connect('motion_notify_event', self.onPlotMoveEvent)
            self.plotter.show()  
            self.plotter.draw()   
            
    def update(self):
        
        self.data = self.Reading.data
        
        self.mapper.axes.clear() # not fast
        # this bit is messy
        if (not self.selectedChannels):
            self.mapper.axes.imshow(mean(self.data,2),cmap='jet')
            title = 'Mean image'
        else:
            if self.selectedChannels.size == 1:
                self.mapper.axes.imshow(self.data[:,:,self.selectedChannels],cmap='jet')
                title = "Channel = %d; %s = %.3f" % (self.nx, self.xaxislabel, self.xaxis[self.nx])
            if self.selectedChannels.size > 1:
                self.mapper.axes.imshow(mean(self.data[:,:,self.selectedChannels],2),cmap='jet')
                title = self.name+' '+'mean of selected channels'
        self.mapper.axes.set_title(title)
        self.mapper.show()
        self.mapper.draw()          
        

        
    def update_rec(self):
                
        self.mapper.axes.clear() # not fast
        # this bit is messy
        if (not self.selectedChannels):
            self.mapper.axes.imshow(mean(self.data,2),cmap='jet')
            title = 'Mean image'
        else:
            if self.selectedChannels.size == 1:
                self.mapper.axes.imshow(self.data[:,:,self.selectedChannels],cmap='jet')
                title = "Channel = %d; %s = %.3f" % (self.nx, self.xaxislabel, self.xaxis[self.nx])
            if self.selectedChannels.size > 1:
                self.mapper.axes.imshow(mean(self.data[:,:,self.selectedChannels],2),cmap='jet')
                title = self.name+' '+'mean of selected channels'
        self.mapper.axes.set_title(title)
        self.mapper.show()
        self.mapper.draw()       
        
        
    def changedata(self,ind):
        if ind == 1:
            self.data = self.Rec.bp
            print "Should be reconstructed data displayed"
        else:
            self.data = self.Reading.data
            print "Should be sinogram data displayed"
            
        self.update_rec()        
        
#    def selXRDCTdata(self):
#        xrdctpath = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select XRD-CT data directory"))
#        xrdctpath = xrdctpath.split("xrdct/");xrdctpath = xrdctpath[1]
#        self.datalist.addItem(xrdctpath)

    def selXRDCTdata(self):
        
        self.dialog = FileDialog()
        if self.dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.paths = (self.dialog.selectedFiles())
        
        for ii in range(0,len(self.paths)):
            oldpath = str(self.paths[ii])
            try:
                self.newpath = oldpath.split("xrdct/");
                self.newpath = self.newpath[1]
                print self.newpath
                if len(self.newpath)>0:
                    self.datalist.addItem(self.newpath)        
            except:
                pass
                
    def remXRDCTdata(self):
        
        for item in self.datalist.selectedItems():
            self.datalist.takeItem(self.datalist.row(item))
#        self.datalist.takeItem()
        
    def batchXRDCTdata(self):

        self.pbutton11.setEnabled(False)   
        self.pbutton12.setEnabled(False)
        self.pbutton13.setEnabled(False)  
        self.pbutton14.setEnabled(False)  
        self.pbutton15.setEnabled(False)     
        
        datasets = [] 
        for index in xrange(self.datalist.count()):
            datasets.append(self.datalist.item(index).text())
            
        self.datasets = datasets
#        print self.datasets[0]
        
        self.Squeezing = []       
        self.progressbarbatch = []
        self.labels = []
        for ii in range(0,len(self.datasets)):
            
            self.dataset = self.datasets[ii]
            self.prefix = self.dataset
            na = float(self.na); nt = float(self.nt);npt_rad = float(self.npt_rad);
          
#            self.parfile = '%s/%s/%s.par' %(self.xrdctpath, self.dataset, self.dataset)
#            try:
#                print self.parfile
#                self.readparinfo()
#            except: 
#                print '.par file does not exist or prefix is different'

            self.specfile = '%s/%s/%s.spec' %(self.xrdctpath, self.dataset, self.dataset)
            try:
                print self.specfile
                self.readspecinfo()
                na = self.nomega
                nt = self.ny
            except: 
                print '.spec file does not exist or prefix is different' 
            
             
            if self.procunit == "MultiGPU":
                if self.scantype == 'Zigzag':
                    self.Squeezing.append(XRDCT_ID15ASqueeze(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.jsonname,self.omega,self.trans,self.dio,self.etime))
                elif self.scantype == 'ContRot':
                    self.Squeezing.append(Fast_XRDCT_ID15ASqueeze(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.jsonname,self.omega,self.trans,self.dio,self.etime))
            else:
                self.Squeezing.append(XRDCT_Squeeze(self.prefix,self.dataset,self.xrdctpath,self.maskname,self.poniname,na,nt,npt_rad,self.filt,self.procunit,self.units,self.prc,self.thres,self.datatype,self.savepath,self.scantype,self.E,self.omega,self.trans,self.dio,self.etime))

            
            self.progressbarbatch.append(QtWidgets.QProgressBar(self))
            self.l.addWidget(self.progressbarbatch[ii],16+ii,5)
            
            self.labels.append(QtWidgets.QLabel(self))
            self.labels[ii].setText(self.dataset)
            self.l.addWidget(self.labels[ii],16+ii,6)
            
            self.Squeezing[ii].start()     
            if self.procunit == "MultiGPU":
                self.Squeezing[ii].liveRead()                
            self.Squeezing[ii].progress.connect(self.progressbarbatch[ii].setValue)
                
        
    def batchstopprocessxrdct(self):
        
        for ii in range(0,len(self.Squeezing)):
            
            if self.procunit == "MultiGPU":
                self.Squeezing[ii].periodEvent.stop()
                self.Squeezing[ii].periodEventraw.stop()
                self.Squeezing[ii].terminate()
            
            self.Squeezing[ii].terminate()
#            self.progressbarbatch[ii].setValue(0)
            self.l.removeWidget(self.progressbarbatch[ii])
            self.progressbarbatch[ii].deleteLater()
            self.progressbarbatch[ii] = None
            
            self.l.removeWidget(self.labels[ii])
            self.labels[ii].deleteLater()
            self.labels[ii] = None
            
        self.pbutton11.setEnabled(True)   
        self.pbutton12.setEnabled(True)
        self.pbutton13.setEnabled(True)  
        self.pbutton14.setEnabled(True)  
        self.pbutton15.setEnabled(True)  
        
#        self.progressbarbatch.setValue(0)
        
    def saveparfile(self):
        
        self.fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Scan Parameters File', "", "*.json")
	
        if len(self.fn)>0:

            self.d = {"Energy":float(self.E), "Datatype":self.datatype,"CalibrantPath":self.calibrant,"PoniPath":self.poniname,"MaskPath":self.maskname, "Mask":self.mask, "JsonPath":self.jsonname, 
    		     "RadialPoints":int(self.npt_rad), "Units":self.units, "SavePath":self.savepath, "XRDCTPath":self.xrdctpath, "ProcessingUnit":self.procunit, "Filter":self.filt, 
    		     "TrimmedMean":int(self.prc), "Sigma":int(self.thres), "ScanType":self.scantype, "SpecFile":self.specfile, "SlowAxisSteps":int(self.nt), "FastAxisSteps":int(self.na),
    		     "Dataset":self.dataset, "Prefix":self.prefix}#, "Omega":self.omega, "Translations":self.trans}
    
            st = self.fn.split(".json")
            if len(st)<2:
                self.fn = "%s.json" %self.fn
                print self.fn
    
            with open(self.fn, 'w') as outfile:  
                json.dump(self.d, outfile)
    
            perm = 'chmod 777 %s' %self.fn
            os.system(perm)    
                    
    def loadparfile(self):   

        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Scan Parameters File', "", "*.json")

        if len(fn)>0:
            d = json.load(open(fn))
            self.E = d["Energy"];self.datatype=d["Datatype"];self.calibrant=d["CalibrantPath"];self.poniname=d["PoniPath"];self.maskname=d["MaskPath"];self.mask=d["Mask"];
            self.jsonname=d["JsonPath"];self.npt_rad=d["RadialPoints"];self.units=d["Units"];self.savepath=d["SavePath"];self.xrdctpath=d["XRDCTPath"];self.procunit=d["ProcessingUnit"];
            self.filt=d["Filter"];self.prc=d["TrimmedMean"];self.thres=d["Sigma"];self.scantype=d["ScanType"];
            self.nt=d["SlowAxisSteps"];self.na=d["FastAxisSteps"];
            self.dataset=d["Dataset"];self.prefix=d["Prefix"];#;self.omega=d["Omega"];self.trans=d["Translations"]
    		
            self.E = float(self.E);self.na = int(self.na);self.nt = int(self.nt);
    		# now need to display these things
            self.EnergySel.setText(str(self.E));self.calibpath.setText(self.calibrant);self.ponipath.setText(str(self.poniname));self.calibmaskpath.setText(self.maskname);
            self.jsonpath.setText(self.jsonname);self.savedatapath.setText(self.savepath);self.datapath.setText(self.xrdctpath);self.radialpoints.setText(str(self.npt_rad));
            self.xrdctname.setText(self.dataset);self.setnt.setText(str(self.nt));self.setna.setText(str(self.na));
    		
            try:
                self.specfile=d["SpecFile"]
                self.xrdctspecpath.setText(self.specfile)
                self.readspecinfo()
            except:
                print "No .spec file"
    		
            if self.procunit == "MultiGPU":
                self.ChooseProcUnit.setCurrentIndex(0)
            elif self.procunit == "CPU":
                self.ChooseProcUnit.setCurrentIndex(1)
            elif self.procunit == "GPU":
                self.ChooseProcUnit.setCurrentIndex(2)
                
            if self.filt == "No":
                self.ChooseFilters.setCurrentIndex(0)
            elif self.filt == "Median":
                self.ChooseFilters.setCurrentIndex(1)                
            elif self.filt == "trimmed_mean":
                self.ChooseFilters.setCurrentIndex(2)
                self.filtervalue.setEnabled(True)
                self.filtervalue.setText(str(self.prc))
            elif self.filt == "sigma":
                self.ChooseFilters.setCurrentIndex(3)
                self.filtervalue.setEnabled(True) 
                self.filtervalue.setText(str(self.thres))
		        
            
            if self.scantype == "Zigzag":
                self.ChooseScan.setCurrentIndex(0)
            elif self.scantype == "ContRot":
                self.ChooseScan.setCurrentIndex(1)
            elif self.scantype == "Interlaced":
                self.ChooseScan.setCurrentIndex(2)
  
    
    def fileQuit(self):
#        plt.close('all')
        self.close()

    def closeEvent(self, ce):
#        plt.close('all')
        self.fileQuit()

    def about(self):
        message = '<b>Integrator GUI for ID15A v0.1.0 url:</b><p>'
        message += '<p><i>Created by <a href=www.finden.co.uk>Finden</a>. Running under license under GPLv3'
        message += '\t '
        sImage = QtWidgets.QPixmap(".//images//logoLetters.png")
        d = QtWidgets.QMessageBox()
        d.setWindowTitle('About')
        d.setIconPixmap(sImage)
        d.setText(message)
        d.exec_()
        
class FileDialog(QtWidgets.QFileDialog):
        def __init__(self, *args):
            QtWidgets.QFileDialog.__init__(self, *args)
            self.setOption(self.DontUseNativeDialog, True)
            self.setFileMode(self.DirectoryOnly)
            for view in self.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
                if isinstance(view.model(), QtWidgets.QFileSystemModel):
                    view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        
class Coordinate(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class MyCanvas(Canvas):
    #Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.).
    def __init__(self, parent=None, width=5, height=10, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig = fig
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
#        self.axes.hold(False)

        self.compute_initial_figure()
        Canvas.__init__(self, fig)
        self.setParent(parent)

        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)
       
    def compute_initial_figure(self):
        pass
    
def main():
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
    qApp.exec_()
   
if __name__ == "__main__":
    main()

####### For debugging only:    
#aw = ApplicationWindow()    
#aw.show()