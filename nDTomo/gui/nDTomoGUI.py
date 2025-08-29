# -*- coding: utf-8 -*-
"""

nDTomoGUI: A GUI for Chemical Imaging Data Visualization and Analysis

This application provides a PyQt5-based graphical user interface (GUI) 
for handling and visualizing chemical imaging data. It includes tools 
for exploring hyperspectral data, defining regions of interest (ROI), 
extracting local patterns, and performing peak fitting.

Features:
---------
- **Image and Spectrum Visualization:** 
  Display hyperspectral images and spectra interactively.
  
- **ROI Selection and Export:**
  Select regions of interest and export extracted images or spectra.

- **Segmentation and Pattern Extraction:** 
  Apply thresholding to segment images and extract local diffraction patterns.

- **Peak Fitting Module:**
  Perform single peak fitting (Gaussian, Lorentzian, or Pseudo-Voigt profiles) 
  across datasets.

- **File Handling:**
  Supports reading and writing `.hdf5` and `.h5` files, along with export to 
  `.asc` and `.xy` formats.

@author: Dr Antony Vamvakeros

"""

#%%

from __future__ import unicode_literals
from matplotlib import use as u
try:
    u('Qt5Agg')
except:
    print("Warning: Failed to set backend to Qt5Agg. Ensure PyQt5 is installed.")
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal, QThread
import h5py, sys
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import scipy.optimize as sciopt
from scipy.signal import find_peaks
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

class nDTomoGUI(QtWidgets.QMainWindow):
    
    """
    A PyQt5 GUI for visualizing and analyzing chemical imaging datasets.

    This interface enables interaction with hyperspectral or diffraction tomography data.
    Users can load datasets, explore image/spectrum slices, extract ROIs, perform peak fitting,
    and export results. Features include segmentation, dynamic cursor inspection, batch fitting,
    and in-GUI IPython console support.

    Attributes
    ----------
    volume : np.ndarray
        The loaded 3D data volume (X, Y, Channels).
    xaxis : np.ndarray
        The 1D array for the spectral or scattering axis.
    image : np.ndarray
        2D image slice shown in the main display.
    spectrum : np.ndarray
        1D spectrum shown in the spectral viewer.
    cmap : str
        The active colormap for image display.
    xaxislabel : str
        Label for the x-axis of the spectrum (e.g. 'Channel', '2theta').
    chi, chf : int
        Start and end channel indices for ROI selection.
    res : dict
        Dictionary storing the latest peak fitting results.
    mask : np.ndarray
        Binary mask (same XY shape as image) used for ROI operations.
    peaktype : str
        Type of peak profile used ('Gaussian', 'Lorentzian', or 'Pseudo-Voigt').
    """   
    
    def __init__(self):

        """
        Initialize the nDTomoGUI application.

        This method sets up the PyQt5 main window and initializes all necessary GUI elements,
        including menus, tabs, image and spectrum display areas, widgets for ROI selection,
        segmentation, peak fitting controls, and embedded IPython console support. It also
        initializes internal data containers and default parameters used throughout the GUI.

        Main Components Initialized
        ---------------------------
        - File menu: open, append, save, and quit actions for chemical imaging datasets.
        - Advanced menu: create phantom dataset and launch IPython console.
        - Help menu: about dialog with citation and licensing info.
        - Tabbed interface:
            • Tab 1: Image and spectrum explorer with colormap selector and export tools.
            • Tab 2: ROI image generation and visualization.
            • Tab 3: ROI segmentation and pattern extraction.
            • Tab 4: Single-peak fitting with configurable parameters and batch fitting tools.
        - Dock widgets: interactive Matplotlib canvases for image and spectrum display.
        - Controls for selecting colormaps, fitting profiles, and managing real-time cursor updates.

        Attributes Initialized
        ----------------------
        - self.volume : np.ndarray
            Loaded 3D chemical imaging data volume.
        - self.xaxis : np.ndarray
            1D axis corresponding to spectral or scattering dimension.
        - self.cmap : str
            Default colormap for image visualization.
        - self.peaktype : str
            Default peak profile ('Gaussian').
        - self.Area, self.FWHM : float
            Default initial values for area and FWHM fitting.
        - self.loaded_dataset_names : list
            Tracks loaded dataset filenames for display.

        Raises
        ------
        RuntimeError
            If PyQt5 is not installed or Qt5Agg backend cannot be set.
        """
        super(nDTomoGUI, self).__init__()
        
        self.volume = np.zeros(())
        self.xaxis = np.zeros(())
        self.image = np.zeros(())
        self.spectrum = np.zeros(())
        self.cbar = None
        self.im = None
        self.vline = None
        self.hdf_fileName = ''
        self.c = 3E8
        self.h = 6.620700406E-34        
        self.cmap = 'inferno'
        self.xaxislabel = 'Channel'
        self.chi = 0
        self.chf = 1        
        self.cmap_list = ['gray', 'inferno', 'viridis','plasma','magma','cividis','flag', 
            'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
        self.loaded_dataset_names = []
        self.mask = None
        self.peaktype = 'Gaussian'
        self.Area = 0.5; self.Areamin = 0.; self.Areamax = 10.
        self.FWHM = 1.; self.FWHMmin = 0.1; self.FWHMmax = 5.

        
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("nDTomoGUI")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Open data', self.fileOpen, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.file_menu.addAction('&Append data', self.append_file)
        self.file_menu.addAction('&Save data', self.savechemvol)
        self.file_menu.addAction('&Quit', self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        # Advanced menu
        self.advanced_menu = QtWidgets.QMenu('&Advanced', self)
        self.advanced_menu.addAction('Create Phantom Dataset', self.create_phantom)
        self.advanced_menu.addAction('Open IPython Console', self.init_console)
        self.menuBar().addMenu(self.advanced_menu)
        
        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.help_menu.addAction('&About', self.about)        
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)
        
        # Initialize tab screen
        self.tabs = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()
        self.tab4 = QtWidgets.QWidget()
        
        # Add tabs
        self.tabs.addTab(self.tab1,"Hyperexplorer")        
        self.tabs.addTab(self.tab2,"ROI image")        
        self.tabs.addTab(self.tab3,"ROI pattern")        
        self.tabs.addTab(self.tab4,"Peak fitting")        

        self.tab1.layout = QtWidgets.QGridLayout()       
        self.tab2.layout = QtWidgets.QGridLayout()
        self.tab3.layout = QtWidgets.QGridLayout()
        self.tab4.layout = QtWidgets.QGridLayout()
        
        ############ Tab1 - Hyperxplorer ############

        # Create left panel for the image display
        self.fig_image = Figure()
        self.ax_image = self.fig_image.add_subplot(111)
        self.canvas_image = FigureCanvas(self.fig_image)
        self.ax_image.set_title("Image")

        # Create right panel for the spectrum plot
        self.fig_spectrum = Figure()
        self.ax_spectrum = self.fig_spectrum.add_subplot(111)
        self.canvas_spectrum = FigureCanvas(self.fig_spectrum)
        self.ax_spectrum.set_title("Histogram")

        # set up the mapper
        self.mapperWidget = QtWidgets.QWidget(self)
        self.mapperExplorerDock = QtWidgets.QDockWidget("Image", self)
        self.mapperExplorerDock.setWidget(self.mapperWidget)
        self.mapperExplorerDock.setFloating(False)
        self.mapperToolbar = NavigationToolbar(self.canvas_image, self)
        vbox1 = QtWidgets.QVBoxLayout()
        vbox1.addWidget(self.mapperToolbar)
        vbox1.addWidget(self.canvas_image)
        self.mapperWidget.setLayout(vbox1)

        #set up the plotter
        self.plotterWidget = QtWidgets.QWidget(self)
        self.plotterExplorerDock = QtWidgets.QDockWidget("Histogram", self)
        self.plotterExplorerDock.setWidget(self.plotterWidget)
        self.plotterExplorerDock.setFloating(False)
        self.plotterToolbar = NavigationToolbar(self.canvas_spectrum, self)
        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.addWidget(self.plotterToolbar)        
        vbox2.addWidget(self.canvas_spectrum) # starting row, starting column, row span, column span
        self.plotterWidget.setLayout(vbox2)
        
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.mapperExplorerDock)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.plotterExplorerDock)

        self.DatasetLabel = QtWidgets.QLabel(self)
        self.DatasetLabel.setText('Dataset:')
        self.tab1.layout.addWidget(self.DatasetLabel,0,0)

        self.DatasetNameLabel = QtWidgets.QLabel(self)
        self.tab1.layout.addWidget(self.DatasetNameLabel,0,1)         

        self.ChooseColormap = QtWidgets.QLabel(self)
        self.ChooseColormap.setText('Choose colormap')
        self.tab1.layout.addWidget(self.ChooseColormap,0,2)
        
        self.ChooseColormapType = QtWidgets.QComboBox(self)
        self.ChooseColormapType.addItems(self.cmap_list)
        self.ChooseColormapType.currentIndexChanged.connect(self.changecolormap)
        self.ChooseColormapType.setMaximumWidth(170)
        self.tab1.layout.addWidget(self.ChooseColormapType,0,3)  
            
        self.ExportDPbutton = QtWidgets.QPushButton("Export local diffraction pattern",self)
        self.ExportDPbutton.clicked.connect(self.exportdp)
        self.tab1.layout.addWidget(self.ExportDPbutton,1,2)

        self.ExportIMbutton = QtWidgets.QPushButton("Export image of interest",self)
        self.ExportIMbutton.clicked.connect(self.exportim)
        self.tab1.layout.addWidget(self.ExportIMbutton,1,3)
        
        ############ Tab2 - ROI ############


        self.Labelbkg = QtWidgets.QLabel(self)
        self.Labelbkg.setText('Set the channels for the ROI image')
        self.tab2.layout.addWidget(self.Labelbkg,0,0)
        
        self.Channel1 = QtWidgets.QLabel(self)
        self.Channel1.setText('Initial channel')
        self.tab2.layout.addWidget(self.Channel1,1,0)        
                    
        self.crspinbox1 = QtWidgets.QSpinBox(self)
        self.crspinbox1.valueChanged.connect(self.channel_initial)
        self.crspinbox1.setMinimum(1)
        self.tab2.layout.addWidget(self.crspinbox1,1,1)    
        self.crspinbox1.valueChanged.connect(self.sync_peak_position_from_roi)
        
        self.Channel2 = QtWidgets.QLabel(self)
        self.Channel2.setText('Final channel')
        self.tab2.layout.addWidget(self.Channel2,1,2)        
                    
        self.crspinbox2 = QtWidgets.QSpinBox(self)
        self.crspinbox2.valueChanged.connect(self.channel_final)
        self.crspinbox2.setMinimum(1)
        self.tab2.layout.addWidget(self.crspinbox2,1,3)        
        self.crspinbox2.valueChanged.connect(self.sync_peak_position_from_roi)   
        
        self.pbutton1 = QtWidgets.QPushButton("Plot mean image", self)
        self.pbutton1.clicked.connect(self.plot_mean_image)
        self.tab2.layout.addWidget(self.pbutton1, 2, 0)
        
        self.pbutton2 = QtWidgets.QPushButton("Plot mean image with mean bkg subtraction", self)
        self.pbutton2.clicked.connect(self.plot_mean_image_mean_bkg)
        self.tab2.layout.addWidget(self.pbutton2, 2, 1)

        self.pbutton3 = QtWidgets.QPushButton("Plot mean image with linear bkg subtraction", self)
        self.pbutton3.clicked.connect(self.plot_mean_image_linear_bkg)
        self.tab2.layout.addWidget(self.pbutton3, 2, 2)

        self.expimroi = QtWidgets.QPushButton("Export image",self)
        self.expimroi.clicked.connect(self.export_roi_image)
        self.tab2.layout.addWidget(self.expimroi,2,3)  
        

        ############ Tab3 - Segmentation and extraction of local pattern ############


        self.Labelbkg = QtWidgets.QLabel(self)
        self.Labelbkg.setText('Set the threshold to segment the ROI image - range is between 0-100%')
        self.tab3.layout.addWidget(self.Labelbkg,0,0)
        
        self.Channel3 = QtWidgets.QLabel(self)
        self.Channel3.setText('Threshold')
        self.tab3.layout.addWidget(self.Channel3,1,0)        
                    
        self.crspinbox3 = QtWidgets.QSpinBox(self)
        self.crspinbox3.valueChanged.connect(self.set_thr)
        self.crspinbox3.setMinimum(0)
        self.crspinbox3.setMaximum(100)
        self.tab3.layout.addWidget(self.crspinbox3,1,1)              
        
        self.pbutton4 = QtWidgets.QPushButton("Apply the threshold", self)
        self.pbutton4.clicked.connect(self.segment_image)
        self.tab3.layout.addWidget(self.pbutton4, 2, 0)
        
        self.pbutton5 = QtWidgets.QPushButton("Use mask to extract ROI pattern from the volume", self)
        self.pbutton5.clicked.connect(self.plot_roi_pattern)
        self.tab3.layout.addWidget(self.pbutton5, 2, 1)

        self.exppatroi = QtWidgets.QPushButton("Export ROI pattern",self)
        self.exppatroi.clicked.connect(self.export_roi_pattern)
        self.tab3.layout.addWidget(self.exppatroi,2,3)  

        self.pbutton_suggest_peaks = QtWidgets.QPushButton("Suggest peak positions", self)
        self.pbutton_suggest_peaks.clicked.connect(self.suggest_peak_positions)
        self.tab3.layout.addWidget(self.pbutton_suggest_peaks, 3, 0)

        ############ Tab4 - Peak fitting ############

        self.Labelbkg = QtWidgets.QLabel(self)
        self.Labelbkg.setText('Single peak fitting')
        self.tab4.layout.addWidget(self.Labelbkg,0,0)
        
        # Define x-axis channel range for fitting
        self.label_range = QtWidgets.QLabel("Fit range (channels):")
        self.tab4.layout.addWidget(self.label_range, 0, 1)

        self.xfit_min_spin = QtWidgets.QSpinBox()
        self.xfit_min_spin.setMinimum(0)
        self.xfit_min_spin.setMaximum(1)  # will be updated after data load
        self.xfit_min_spin.setValue(0)
        self.tab4.layout.addWidget(self.xfit_min_spin, 0, 2)

        self.xfit_max_spin = QtWidgets.QSpinBox()
        self.xfit_max_spin.setMinimum(0)
        self.xfit_max_spin.setMaximum(1)  # will be updated after data load
        self.xfit_max_spin.setValue(1)
        self.tab4.layout.addWidget(self.xfit_max_spin, 0, 3)

        self.set_xfit_button = QtWidgets.QPushButton("Set fit range", self)
        self.set_xfit_button.clicked.connect(self.set_fit_range)
        self.tab4.layout.addWidget(self.set_xfit_button, 0, 4)
        
        self.LabelTypePeak = QtWidgets.QLabel(self)
        self.LabelTypePeak.setText('Function')
        self.tab4.layout.addWidget(self.LabelTypePeak,1,0)
        
        self.ChooseFunction = QtWidgets.QComboBox(self)
        self.ChooseFunction.addItems(["Gaussian", "Lorentzian", "Pseudo-Voigt"])
        self.ChooseFunction.currentIndexChanged.connect(self.profile_function)
        self.ChooseFunction.setEnabled(True)
        self.tab4.layout.addWidget(self.ChooseFunction,1,1)   

        # Peak Area controls
        self.label_area = QtWidgets.QLabel("Area")
        self.tab4.layout.addWidget(self.label_area, 2, 0)
        self.area_spin = QtWidgets.QDoubleSpinBox()
        self.area_spin.setRange(0.0, 100.0)
        self.area_spin.setValue(self.Area)
        self.area_spin.setSingleStep(0.1)
        self.tab4.layout.addWidget(self.area_spin, 2, 1)
        
        self.label_area_min = QtWidgets.QLabel("Min")
        self.tab4.layout.addWidget(self.label_area_min, 2, 2)
        self.area_min_spin = QtWidgets.QDoubleSpinBox()
        self.area_min_spin.setRange(0.0, 100.0)
        self.area_min_spin.setValue(self.Areamin)
        self.tab4.layout.addWidget(self.area_min_spin, 2, 3)

        self.label_area_max = QtWidgets.QLabel("Max")
        self.tab4.layout.addWidget(self.label_area_max, 2, 4)
        self.area_max_spin = QtWidgets.QDoubleSpinBox()
        self.area_max_spin.setRange(0.0, 100.0)
        self.area_max_spin.setValue(self.Areamax)
        self.tab4.layout.addWidget(self.area_max_spin, 2, 5)

        # FWHM controls
        self.label_fwhm = QtWidgets.QLabel("FWHM")
        self.tab4.layout.addWidget(self.label_fwhm, 3, 0)
        self.fwhm_spin = QtWidgets.QDoubleSpinBox()
        self.fwhm_spin.setRange(0.01, 20.0)
        self.fwhm_spin.setValue(self.FWHM)
        self.fwhm_spin.setSingleStep(0.1)
        self.tab4.layout.addWidget(self.fwhm_spin, 3, 1)
        
        self.label_fwhm_min = QtWidgets.QLabel("Min")
        self.tab4.layout.addWidget(self.label_fwhm_min, 3, 2)
        self.fwhm_min_spin = QtWidgets.QDoubleSpinBox()
        self.fwhm_min_spin.setRange(0.01, 20.0)
        self.fwhm_min_spin.setValue(self.FWHMmin)
        self.tab4.layout.addWidget(self.fwhm_min_spin, 3, 3)

        self.label_fwhm_max = QtWidgets.QLabel("Max")
        self.tab4.layout.addWidget(self.label_fwhm_max, 3, 4)
        self.fwhm_max_spin = QtWidgets.QDoubleSpinBox()
        self.fwhm_max_spin.setRange(0.01, 20.0)
        self.fwhm_max_spin.setValue(self.FWHMmax)
        self.tab4.layout.addWidget(self.fwhm_max_spin, 3, 5)        
        
        # Central position
        self.label_pos = QtWidgets.QLabel("Position")
        self.tab4.layout.addWidget(self.label_pos, 4, 0)
        self.pos_spin = QtWidgets.QDoubleSpinBox()
        self.pos_spin.setDecimals(2)
        self.pos_spin.setRange(0.0, 1e6)
        self.pos_spin.setSingleStep(0.1)
        self.tab4.layout.addWidget(self.pos_spin, 4, 1)

        # Position min
        self.label_pos_min = QtWidgets.QLabel("Min")
        self.tab4.layout.addWidget(self.label_pos_min, 4, 2)
        self.pos_min_spin = QtWidgets.QDoubleSpinBox()
        self.pos_min_spin.setDecimals(2)
        self.pos_min_spin.setRange(0.0, 1e6)
        self.tab4.layout.addWidget(self.pos_min_spin, 4, 3)

        # Position max
        self.label_pos_max = QtWidgets.QLabel("Max")
        self.tab4.layout.addWidget(self.label_pos_max, 4, 4)
        self.pos_max_spin = QtWidgets.QDoubleSpinBox()
        self.pos_max_spin.setDecimals(2)
        self.pos_max_spin.setRange(0.0, 1e6)
        self.tab4.layout.addWidget(self.pos_max_spin, 4, 5)

        # Pseudo-Voigt mixing parameter (gamma)
        self.label_fraction = QtWidgets.QLabel("Mixing γ")
        self.tab4.layout.addWidget(self.label_fraction, 5, 0)
        self.fraction_spin = QtWidgets.QDoubleSpinBox()
        self.fraction_spin.setRange(0.0, 1.0)
        self.fraction_spin.setSingleStep(0.05)
        self.fraction_spin.setValue(0.5)
        self.tab4.layout.addWidget(self.fraction_spin, 5, 1)
        self.label_fraction.setVisible(False)
        self.fraction_spin.setVisible(False)
        
        self.area_spin.setToolTip("Initial guess for peak area")
        self.fraction_spin.setToolTip("Mixing fraction between Gaussian (0) and Lorentzian (1)")
        
        self.label_area.setToolTip("Initial guess for peak area")
        self.label_area_min.setToolTip("Minimum allowed peak area")
        self.label_area_max.setToolTip("Maximum allowed peak area")
        
        self.pbutton_fit = QtWidgets.QPushButton("Perform batch peak fitting",self)
        self.pbutton_fit.clicked.connect(self.batchpeakfit)
        self.tab4.layout.addWidget(self.pbutton_fit,1,2)
        
        self.progressbar_fit = QtWidgets.QProgressBar(self)
        self.tab4.layout.addWidget(self.progressbar_fit,1,3)

        self.pbutton_stop = QtWidgets.QPushButton("Stop",self)
        self.pbutton_stop.clicked.connect(self.stopfit)
        self.tab4.layout.addWidget(self.pbutton_stop,1,4)
        
        
        self.LabelLive = QtWidgets.QLabel("Live view:")
        self.tab4.layout.addWidget(self.LabelLive, 1, 5)

        self.ChooseLive = QtWidgets.QComboBox(self)
        self.ChooseLive.addItems(['Area','Position','FWHM'])
        self.ChooseLive.setEnabled(True)
        self.tab4.layout.addWidget(self.ChooseLive, 1, 6)        
        
        self.LabelRes = QtWidgets.QLabel(self)
        self.LabelRes.setText('Display peak fitting results')
        self.tab4.layout.addWidget(self.LabelRes,2,6)
        
        self.ChooseRes = QtWidgets.QComboBox(self)
        self.ChooseRes.addItems(['Area','Position', 'FWHM', 'Slope', 'Intercept'])
        self.ChooseRes.currentIndexChanged.connect(self.plot_fit_results)
        self.ChooseRes.setEnabled(False)
        self.tab4.layout.addWidget(self.ChooseRes,2,7)   
        
        self.pbutton_expfit = QtWidgets.QPushButton("Export fit results",self)
        self.pbutton_expfit.clicked.connect(self.savefitresults)
        self.pbutton_expfit.setEnabled(False)
        self.tab4.layout.addWidget(self.pbutton_expfit,2,8)

        self.check_diag_mode = QtWidgets.QCheckBox("Inspect Fit Diagnostics (live overlay)", self)
        self.check_diag_mode.setChecked(False)
        self.tab4.layout.addWidget(self.check_diag_mode, 3, 6)

        ############

        self.tabs.setFocus()        
        self.tab1.setLayout(self.tab1.layout)     
        self.tab2.setLayout(self.tab2.layout)   
        self.tab3.setLayout(self.tab3.layout)   
        self.tab4.setLayout(self.tab4.layout)   
        self.setCentralWidget(self.tabs)
        self.show()
            
    ####################### Methods #######################
     
            
    def explore(self):
        
        """
        Display the mean image and mean spectrum from the loaded chemical imaging dataset.

        This method calculates the average projection of the 3D volume along the spectral axis
        and displays it in the image panel. It also computes the mean spectrum across all spatial
        positions and displays it in the spectrum panel. Interactive tools are initialized to allow
        real-time cursor-based inspection of local spectra and image slices.

        Functionality
        -------------
        - Computes and displays the mean image (`volume.mean(axis=2)`) using the current colormap.
        - Computes and displays the mean spectrum (`volume.mean(axis=(0, 1))`) as a 1D plot.
        - Adds a vertical cursor line to the spectrum panel (movable by mouse hover).
        - Enables interactive updates:
            • Hover over the image shows the corresponding spectrum at that pixel.
            • Hover over the spectrum shows the corresponding image slice at that channel.
            • Scroll wheel zooms on both image and spectrum panels.
            • Left-click toggles real-time updates on; right-click toggles them off.

        Raises
        ------
        ValueError
            If no data volume has been loaded (`self.volume.size == 0`).
        """

        if self.volume.size == 0:
            raise ValueError("No data loaded. Please load an HDF5 file first.")

        # Calculate mean image and spectrum
        mean_image = np.mean(self.volume, axis=2)
        mean_spectrum = np.mean(self.volume, axis=(0, 1))

        self.fig_image.clear()
        self.ax_image = self.fig_image.add_subplot(111)
        self.im = self.ax_image.imshow(mean_image.T, cmap=self.cmap)

        # Safely remove any existing colorbar
        try:
            if hasattr(self, 'cbar') and self.cbar is not None and self.cbar.ax:
                self.cbar.remove()
        except Exception as e:
            print("Warning: Failed to remove colorbar:", e)

        self.cbar = None

        self.cbar = self.fig_image.colorbar(self.im, ax=self.ax_image, fraction=0.046, pad=0.04)

        self.ax_spectrum.clear()
        self.ax_spectrum.plot(self.xaxis, mean_spectrum, color='b')
        self.ax_spectrum.set_xlabel(self.xaxislabel)

        # Add vertical line for spectrum cursor
        self.vline = self.ax_spectrum.axvline(x=self.xaxis[0], color='r', linestyle='--', lw=1)

        # Connect events
        self.canvas_image.mpl_connect('motion_notify_event', self.update_spectrum)
        self.canvas_spectrum.mpl_connect('motion_notify_event', self.update_image)
        self.canvas_image.mpl_connect('button_press_event', self.toggle_real_time_spectrum)
        self.canvas_spectrum.mpl_connect('button_press_event', self.toggle_real_time_image)
        self.canvas_image.mpl_connect('scroll_event', self.on_canvas_scroll)
        self.canvas_spectrum.mpl_connect('scroll_event', self.on_canvas_scroll)

        self.real_time_update_image = True
        self.real_time_update_spectrum = True

        self.canvas_image.draw()
        self.canvas_spectrum.draw()

    def on_canvas_scroll(self, event):
        """
        Handle zooming interactions triggered by mouse scroll events.

        This method responds to scroll wheel events within the image or spectrum axes.
        When scrolling up, it zooms in around the cursor location; when scrolling down,
        it zooms out. Zooming is applied symmetrically in both axes around the cursor.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The scroll event containing information about which axis the cursor is over,
            the direction of the scroll, and the data coordinates of the cursor.

        Notes
        -----
        - Scroll up zooms in (magnifies view around cursor).
        - Scroll down zooms out (restores a wider view).
        - Has no effect if the scroll occurs outside the plotting axes.
        """
        zoom_factor = 1.5  # Zoom factor

        if event.button == 'up':
            zoom_factor = 1 / zoom_factor  # Zoom in
        elif event.button == 'down':
            pass  # Zoom out
        else:
            return

        if event.inaxes == self.ax_image:
            self.ax_image.set_xlim(event.xdata - zoom_factor * (event.xdata - self.ax_image.get_xlim()[0]),
                                   event.xdata + zoom_factor * (self.ax_image.get_xlim()[1] - event.xdata))
            self.ax_image.set_ylim(event.ydata - zoom_factor * (event.ydata - self.ax_image.get_ylim()[0]),
                                   event.ydata + zoom_factor * (self.ax_image.get_ylim()[1] - event.ydata))
            self.canvas_image.draw()

        elif event.inaxes == self.ax_spectrum:
            self.ax_spectrum.set_xlim(event.xdata - zoom_factor * (event.xdata - self.ax_spectrum.get_xlim()[0]),
                                      event.xdata + zoom_factor * (self.ax_spectrum.get_xlim()[1] - event.xdata))
            self.ax_spectrum.set_ylim(event.ydata - zoom_factor * (event.ydata - self.ax_spectrum.get_ylim()[0]),
                                      event.ydata + zoom_factor * (self.ax_spectrum.get_ylim()[1] - event.ydata))
            self.canvas_spectrum.draw()


    def update_spectrum(self, event):
        """
        Update the 1D spectrum plot based on mouse hover over the 2D image.

        When the user moves the mouse over the image display (ax_image), this method 
        retrieves the spectrum at the corresponding pixel coordinates and plots it.

        If diagnostic mode is enabled and fitting results are available, the fitted
        peak model and residual (difference between raw and fitted spectra) are also shown.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event object containing position and axis metadata.

        Behavior
        --------
        - Plots the raw spectrum at (x, y).
        - If diagnostic overlay is enabled and the pixel is within a fitted mask:
            - Plots the model fit (red line) and residuals (green line).
            - Supports Gaussian, Lorentzian, and Pseudo-Voigt peak profiles.
        - Updates plot title and legend accordingly.

        Notes
        -----
        - Requires valid `self.volume`, `self.xaxis`, and axis handles to be initialized.
        - `self.mask` must be defined for diagnostics to overlay.
        - Fit parameters are read from `self.res`, produced after peak fitting.
        """
        if self.real_time_update_spectrum and event.inaxes == self.ax_image:
            self.x, self.y = int(event.xdata), int(event.ydata)

            if 0 <= self.x < self.image_width and 0 <= self.y < self.image_height:
                self.spectrum = self.volume[self.x, self.y, :]

                # Clear all lines except the vertical cursor
                for line in self.ax_spectrum.get_lines():
                    if line != self.vline:
                        line.remove()

                # Decide raw plot style
                show_diag = hasattr(self, 'check_diag_mode') and self.check_diag_mode.isChecked()
                has_fit = hasattr(self, 'res')


                # Overlay fit and residuals if diagnostics are on and fit exists
                if show_diag and has_fit and self.mask[self.x, self.y] > 0:
                    try:
                        ch_min = self.xfit_min_spin.value()
                        ch_max = self.xfit_max_spin.value()
                        fit_x_ch = np.arange(ch_min, ch_max)
                        fit_x_native = self.xaxis[ch_min:ch_max]
                        raw_subset = self.spectrum[ch_min:ch_max]

                        area = self.res['Area'][self.x, self.y]
                        pos = self.res['Position'][self.x, self.y]
                        fwhm = self.res['FWHM'][self.x, self.y]
                        bkg1 = self.res['Background1'][self.x, self.y]
                        bkg2 = self.res['Background2'][self.x, self.y]

                        if self.peaktype == "Gaussian":
                            fit_y = self.PeakFitData.gmodel(fit_x_ch, area, pos, fwhm, bkg1, bkg2)
                        elif self.peaktype == "Lorentzian":
                            fit_y = self.PeakFitData.lmodel(fit_x_ch, area, pos, fwhm, bkg1, bkg2)
                        elif self.peaktype == "Pseudo-Voigt":
                            frac = self.res['Fraction'][self.x, self.y]
                            fit_y = self.PeakFitData.pvmodel(fit_x_ch, area, pos, fwhm, bkg1, bkg2, frac)
                        else:
                            fit_y = None

                        if fit_y is not None:
                            self.ax_spectrum.plot(self.xaxis, self.spectrum, 'b--', label='Raw')
                            self.ax_spectrum.plot(fit_x_native, fit_y, 'r', label='Fit')
                            residual = raw_subset - fit_y
                            self.ax_spectrum.plot(fit_x_native, residual, 'g', label='Residual')

                    except Exception as e:
                        print(f"[Diagnostic overlay error]: {e}")
                        
                else:
                    self.ax_spectrum.plot(self.xaxis, self.spectrum, 'b', label='Raw')
                
                self.ax_spectrum.set_title(f"Spectrum at ({self.x}, {self.y})")
                self.ax_spectrum.set_xlabel(self.xaxislabel)
                self.ax_spectrum.legend()
                self.canvas_spectrum.draw()

    def valid_fit_for_pixel(self, x, y):
        """
        Check if the fitted peak parameters at pixel (x, y) are valid.

        A fit is considered valid if:
        - 'Area', 'Position', and 'FWHM' exist at the pixel,
        - All values are finite (not NaN or Inf),
        - The FWHM value is strictly positive.

        Parameters
        ----------
        x : int
            X-coordinate (row index) of the pixel.
        y : int
            Y-coordinate (column index) of the pixel.

        Returns
        -------
        bool
            True if the pixel has valid fit parameters, False otherwise.
        """
        
        try:
            return (
                np.isfinite(self.res['Area'][x, y]) and
                np.isfinite(self.res['Position'][x, y]) and
                np.isfinite(self.res['FWHM'][x, y]) and
                self.res['FWHM'][x, y] > 0
            )
        except:
            return False
        
    def update_image(self, event):
        """
        Update the 2D image view based on mouse hover over the spectrum plot.

        When the user moves the mouse cursor along the spectral axis (ax_spectrum),
        this method determines the corresponding index in the spectral axis and displays
        the corresponding 2D slice from the volume.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event object triggered by hover in the spectrum axis.

        Behavior
        --------
        - Translates the hovered x-axis position to the nearest spectral index.
        - Extracts the corresponding image slice and updates the image view.
        - Moves the red vertical cursor line in the spectrum plot to match.
        - Automatically adjusts color limits and updates the colorbar.

        Notes
        -----
        - Handles both default x-axis and inverted axes (e.g., `d-spacing`).
        - Ensures compatibility with mouse wheel zoom and interaction events.
        """
        if self.real_time_update_image and event.inaxes == self.ax_spectrum and event.xdata is not None:

            self.index = event.xdata

            if self.index >= 0 and self.index < self.nbins:
                # try:
                if self.xaxislabel == 'd':
                    self.index = len(self.xaxis) - np.searchsorted(self.xaxisd, [self.index])[0]
                else:
                    self.index = np.searchsorted(self.xaxis.flatten(), [self.index])[0] - 1

                self.index = max(0, min(self.index, len(self.xaxis) - 1))

                # Get image at selected index
                self.image = self.volume[:, :, self.index]

                # Ensure the image object exists
                if hasattr(self, 'im') and self.im is not None:
                    self.im.set_data(self.image.T)
                    self.im.set_clim(np.min(self.image), np.max(self.image))
                else:
                    self.im = self.ax_image.imshow(self.image.T, cmap=self.cmap)
                    self.cbar = self.fig_image.colorbar(self.im, ax=self.ax_image, fraction=0.046, pad=0.04)

                if hasattr(self, 'cbar') and self.cbar is not None and hasattr(self, 'im') and self.im is not None:
                    self.cbar.update_normal(self.im)

                if self.xaxislabel == 'Channel':
                    self.ax_image.set_title(f"Image: Channel {self.index}")
                else:
                    self.ax_image.set_title("Image: Channel %d, %s %.3f" %
                                            (self.index, self.xaxislabel, self.xaxis[self.index]))

                # Move vertical line on spectrum plot
                if hasattr(self, 'vline'):
                    self.vline.set_xdata([self.xaxis[self.index]])
                    self.canvas_spectrum.draw()

                self.canvas_image.draw()

                # except Exception as e:
                #     print(f"Error in update_image: {e}")

    def toggle_real_time_spectrum(self, event):
        """
        Toggle real-time spectrum updates when hovering over the image plot.

        Enables or disables live spectrum display depending on which mouse button is pressed
        while clicking on the image plot.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event object. Left-click enables, right-click disables real-time updates.
        """
        if event.button == 1:
            self.real_time_update_spectrum = True
        elif event.button == 3:
            self.real_time_update_spectrum = False

    def toggle_real_time_image(self, event):
        """
        Toggle real-time image updates when hovering over the spectrum plot.

        Enables or disables live image updates depending on which mouse button is pressed
        while clicking on the spectrum plot.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event object. Left-click enables, right-click disables real-time updates.
        """
        if event.button == 1:
            self.real_time_update_image = True
        elif event.button == 3:
            self.real_time_update_image = False
            
    def exportdp(self):
        """
        Export the current spectrum (diffraction pattern) to HDF5 and text-based formats.

        Saves the currently displayed 1D spectrum (from the selected pixel) and its corresponding
        x-axis to three separate files:
        - `.h5` file containing datasets `I` and `xaxis`
        - `.asc` text file with two columns: x-axis and intensity
        - `.xy` text file with the same format as `.asc`

        Output filenames include the pixel coordinates (x, y).

        Notes
        -----
        This function requires that `self.hdf_fileName`, `self.spectrum`, `self.xaxis`,
        and pixel coordinates `self.x`, `self.y` are defined.
        """
        # Check all prerequisites
        if not hasattr(self, 'hdf_fileName') or not self.hdf_fileName:
            print("Missing or invalid HDF5 filename.")
            return

        if not hasattr(self, 'spectrum') or not isinstance(self.spectrum, np.ndarray) or self.spectrum.size == 0:
            print("Spectrum not selected or empty.")
            return

        if not hasattr(self, 'xaxis') or not isinstance(self.xaxis, np.ndarray) or self.xaxis.size != self.spectrum.size:
            print("X-axis not defined or does not match spectrum size.")
            return

        if not all(hasattr(self, attr) for attr in ['x', 'y']):
            print("Pixel coordinates (x, y) are not defined.")
            return

        # Safe to proceed with export
        base = self.hdf_fileName.rsplit('.h5', 1)[0]
        tag = f"{base}_{self.x}_{self.y}"

        h5_name = f"{tag}.h5"
        asc_name = f"{tag}.asc"
        xy_name = f"{tag}.xy"

        print(h5_name)

        with h5py.File(h5_name, "w") as h5f:
            h5f.create_dataset('I', data=self.spectrum)
            h5f.create_dataset('xaxis', data=self.xaxis)

        xy_data = np.column_stack((self.xaxis, self.spectrum))
        np.savetxt(asc_name, xy_data, fmt="%.5f")
        np.savetxt(xy_name, xy_data, fmt="%.5f")
        
    def exportim(self):
        """
        Export the current spectral/scattering image to HDF5 and PNG formats.

        Saves the currently displayed 2D image (at a specific spectral index) to:
        - `.h5` file containing datasets `I` (image) and `Channel` (index)
        - `.png` file visualizing the image using the active colormap

        Notes
        -----
        This function requires `self.hdf_fileName`, `self.image`, and `self.index` to be defined and valid.
        """
        # Check required attributes exist and are valid
        if not hasattr(self, 'hdf_fileName') or not self.hdf_fileName:
            print("Missing or invalid HDF5 filename.")
            return

        if not hasattr(self, 'image') or not isinstance(self.image, np.ndarray) or self.image.size == 0:
            print("No image selected or image is empty.")
            return

        if not hasattr(self, 'index'):
            print("No spectral index defined yet.")
            return

        # Safe to export
        base = self.hdf_fileName.rsplit('.h5', 1)[0]
        h5_name = f"{base}_channel_{self.index}.h5"
        png_name = f"{base}_channel_{self.index}.png"

        print(h5_name)
        with h5py.File(h5_name, "w") as h5f:
            h5f.create_dataset('I', data=self.image)
            h5f.create_dataset('Channel', data=self.index)

        plt.imsave(png_name, self.image, cmap=self.cmap)
            
                
    def changecolormap(self, ind):
        """
        Change the active colormap used for image display.

        Parameters
        ----------
        ind : int
            Index of the selected colormap from `self.cmap_list`.

        Notes
        -----
        Updates the display immediately to reflect the new colormap.
        """
        self.cmap = self.cmap_list[ind]
        print(f"Colormap changed to: {self.cmap}")

        try:
            if hasattr(self, 'im') and self.im is not None:
                self.im.set_cmap(self.cmap)
                self.canvas_image.draw()
            
            if hasattr(self, 'cbar') and self.cbar is not None:
                self.cbar.update_normal(self.im)

        except Exception as e:
            print(f"Error updating colormap: {e}")
        

    def selXRDCTdata(self):

        """
        Open a file dialog to select one or more XRD-CT datasets.

        Extracts dataset names and base paths from the selected files,
        and populates the GUI list widget (`self.datalist`) and `self.pathslist`.

        Notes
        -----
        - Uses the custom `FileDialog` class with multi-selection enabled.
        - Expects filenames to end with `.hdf5`.
        - Silently ignores or logs any parsing errors.
        """        
        
        self.dialog = FileDialog()
        if self.dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.paths = (self.dialog.selectedFiles())
        
        for ii in range(0,len(self.paths)):
            oldpath = str(self.paths[ii])
            try:
                newpath = oldpath.split("/")
                self.datasetname = newpath[-1]
                
                try: 
                    newfile = self.datasetname.split(".hdf5")
                    newfile = newfile[0]
                    if len(self.datasetname)>0 and len(newfile)>0:
                        self.datalist.addItem(self.datasetname)    
                        
                        newpath = oldpath.split(self.datasetname)
                        newpath = newpath[0]
                        self.pathslist.append(newpath)  
                        
                except:
                    print('Something is wrong with the selected datasets')
            except:
                pass
                
    def fileOpen(self):
        
        """
        Open and load a chemical imaging dataset via file dialog.

        Opens a file browser to select an HDF5 (.h5 or .hdf5) file, loads the dataset into memory,
        updates the dataset label in the GUI, and calls `explore()` to display the data.

        Notes
        -----
        - Internally calls `self.loadchemvol()` to read the file contents.
        - Only one dataset is loaded at a time using this method.
        """
                
        self.hdf_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open data', "", "*.hdf5 *.h5")

        if len(self.hdf_fileName) > 0:
            self.loadchemvol()
            self.loaded_dataset_names = [self.hdf_fileName.split("/")[-1]]  # <-- track dataset name
            self.DatasetNameLabel.setText(" + ".join(self.loaded_dataset_names))
            self.explore()            
            
                        
    def loadchemvol(self):
        
        """
        Load a chemical imaging volume and spectral axis from the current HDF5 file.

        Reads the 3D dataset from `/data` and an associated axis (e.g., `/twotheta`, `/q`, etc.)
        from the file `self.hdf_fileName`, performing shape and compatibility checks.

        Updates GUI elements such as channel range spin boxes and sets the axis label.

        Raises
        ------
        FileNotFoundError
            If `self.hdf_fileName` is not set.
        KeyError
            If the expected dataset `/data` is not found in the file.
        """
                
        xaxis_labels = ['/d', '/q', '/twotheta', '/Energy', '/tth', '/energy']
        
        if not self.hdf_fileName:
            raise FileNotFoundError("No file selected. Please choose an HDF5 file.")

        with h5py.File(self.hdf_fileName,'r') as f:
            try:
                print("Available datasets:", list(f.keys()))
                if '/data' not in f:
                    raise KeyError("Dataset '/data' not found in HDF5 file.")                
                self.volume = f['/data'][:]
                self.check_and_transpose()
                self.image_width, self.image_height, self.nbins = self.volume.shape
                self.xaxis = np.arange(0, self.volume.shape[2])
                self.crspinbox1.setMaximum(self.nbins - 1)
                self.crspinbox2.setMaximum(self.nbins)

                for xaxis_label in xaxis_labels:
                    if xaxis_label in f and len(f[xaxis_label][:])==self.volume.shape[2]:
                        self.xaxis = f[xaxis_label][:]
                        self.xaxislabel = xaxis_label.lstrip('/')            
                        break  # Stop once we find a match             

                self.xfit_min_spin.setMaximum(self.volume.shape[2] - 1)
                self.xfit_max_spin.setMaximum(self.volume.shape[2])
                self.xfit_max_spin.setValue(self.volume.shape[2])       
                
            except KeyError as e:
                print("Error:", e)
            except Exception as e:
                print("Unexpected error:", e)                                
        print("Loaded data shape:", self.volume.shape)
        
 
                        
    def append_file(self):
        """
        Append an additional chemical imaging dataset to the current volume along the X-axis (axis=0).

        Opens a file dialog to select a new HDF5 dataset and checks for compatibility with the currently
        loaded volume. If the height and spectral dimensions match, the new volume is concatenated 
        column-wise. The GUI is updated accordingly.

        Notes
        -----
        - Only appends if the new volume has the same shape in dimensions 1 and 2 (height and spectral).
        - If necessary, the new volume is transposed to match the shape of the existing dataset.
        - Updates internal volume, dataset label, and spinbox limits.
        
        Raises
        ------
        KeyError
            If the '/data' dataset is not found in the appended file.
        ValueError
            If shape compatibility is not met (different height or spectral axis).
        """
        append_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Append data', "", "*.hdf5 *.h5")

        if not append_fileName or self.volume.size == 0:
            QtWidgets.QMessageBox.warning(self, "Append Error", "No file selected or no dataset loaded yet.")
            return

        try:
            with h5py.File(append_fileName, 'r') as f:
                if '/data' not in f:
                    raise KeyError("Dataset '/data' not found in file.")
                
                # Only check dimension 1 and 2 (height and spectral)
                shape_new = f['/data'].shape
                if shape_new[1] != self.volume.shape[1] or shape_new[2] != self.volume.shape[2]:
                    raise ValueError("Cannot append: mismatch in height or spectral dimension.")

                # Read and transpose only if necessary
                vol_new = f['/data'][:]
                if vol_new.shape[0] != vol_new.shape[1] and vol_new.shape[0] == vol_new.shape[2]:
                    vol_new = np.transpose(vol_new, (0, 2, 1))
                elif vol_new.shape[0] != vol_new.shape[1] and vol_new.shape[1] == vol_new.shape[2]:
                    vol_new = np.transpose(vol_new, (1, 2, 0))

            # Append column-wise (axis=1)
            self.volume = np.concatenate((self.volume, vol_new), axis=0)
            del vol_new  # free memory

            # Update GUI
            self.image_width, self.image_height, self.nbins = self.volume.shape
            self.crspinbox1.setMaximum(self.nbins - 1)
            self.crspinbox2.setMaximum(self.nbins)

            # Track and show appended filename
            appended_name = append_fileName.split("/")[-1]
            self.loaded_dataset_names.append(appended_name)
            self.DatasetNameLabel.setText(" + ".join(self.loaded_dataset_names))

            self.explore()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Append Failed", f"Error appending dataset:\n{e}")

    def check_and_transpose(self):
        """
        Check the shape of the loaded volume and apply a transpose if necessary.

        Ensures that the volume has shape (X, Y, Channels). If the first dimension matches the spectral 
        length instead of spatial axes, the array is transposed accordingly to bring it to the expected format.

        Notes
        -----
        - Transposes (0, 2, 1) if volume.shape[0] == volume.shape[2]
        - Transposes (1, 2, 0) if volume.shape[1] == volume.shape[2]
        """        
        dims = self.volume.shape
        if dims[0] != dims[1] and dims[0] == dims[2]:
            # Transpose the array so that the first dimension becomes the last dimension
            self.volume = np.transpose(self.volume, (0, 2, 1))
        elif dims[0] != dims[1] and dims[1] == dims[2]:
            # Transpose the array so that the second dimension becomes the last dimension
            self.volume = np.transpose(self.volume, (1, 2, 0))
            
    def savechemvol(self):

        """
        Save the current chemical imaging dataset to an HDF5 file.

        Prompts the user to select a save location and writes the current 3D volume
        and x-axis array to disk. Adds a `.h5` extension if not present.

        File contents
        -------------
        - 'data': The 3D chemical imaging volume.
        - 'xaxis': The spectral or scattering axis used for plotting.
        """
        self.fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save data', "", "*.h5")
	
        if len(self.fn)>0:
    
            st = self.fn.split(".h5")
            if len(st)<2:
                self.fn = "%s.h5" %self.fn
                print(self.fn)

            h5f = h5py.File(self.fn, "w")
            h5f.create_dataset('data', data=self.volume)
            h5f.create_dataset('xaxis', data=self.xaxis)
            h5f.close()

    def fileQuit(self):
        """
        Trigger the application close event.
        """        
        self.close()

    def closeEvent(self, ce):
        """
        Override the Qt closeEvent handler to ensure proper shutdown.

        Parameters
        ----------
        ce : QCloseEvent
            The close event triggered by the window manager.
        """        
        self.fileQuit()

    def about(self):
        message = '<b>nDTomoGUI<p>'
        message += '<p><i>Created by Antony Vamvakeros. Running under license under GPLv3'
        message += '<p>Please cite using the following:<p>'
        message += '<p>Vamvakeros, A. et al., nDTomo software suite, 2019, url: https://github.com/antonyvam/nDTomo<p>'
        message += '\t '
        d = QtWidgets.QMessageBox()
        d.setWindowTitle('About')
        d.setText(message)
        d.exec_()


    ####################### ROI image #######################

    def channel_initial(self, value):
        """
        Set the initial channel index for the ROI (Region of Interest) image.

        Parameters
        ----------
        value : int
            The starting spectral channel to define the ROI range.
        """        
        self.chi = value
        print(self.chi)

    def channel_final(self, value):
        """
        Set the final channel index for the ROI (Region of Interest) image.

        Parameters
        ----------
        value : int
            The ending spectral channel to define the ROI range.
        """        
        self.chf = value
        print(self.chf)
        
    def plot_mean_image(self):

        """
        Compute and display the mean ROI image based on selected channel range.

        The method:
        - Calculates a mean image from the 3D volume using channels from `chi` to `chf`.
        - Normalizes the image to a 0-100 range.
        - Automatically estimates a central peak position for fitting.
        - Generates a binary mask from the image using a fixed threshold (5%).
        - Updates the image canvas with the new ROI visualization.
        """
        
        # Ensure chi < chf and both are within volume bounds
        n_channels = self.volume.shape[2]

        if self.chi > self.chf:
            self.chi, self.chf = self.chf, self.chi
            if hasattr(self, 'crspinbox1') and hasattr(self, 'crspinbox2'):
                self.crspinbox1.setValue(self.chi)
                self.crspinbox2.setValue(self.chf)

        self.chi = max(0, min(self.chi, n_channels - 1))
        self.chf = max(self.chi + 1, min(self.chf, n_channels))
        
        roi = np.arange(self.chi,self.chf)

        # Compute central position as float
        pos = 0.5 * (self.chi + self.chf)

        # Set Tab 4 controls automatically
        self.pos_spin.setValue(pos)
        self.pos_min_spin.setValue(max(0.0, pos - 5.0))
        self.pos_max_spin.setValue(pos + 5.0)
        
        self.volroi = self.volume[:,:,roi]
        self.volroi = self.volroi/np.max(self.volroi)
        
        self.xroi = np.arange(self.chi, self.chf)
        if self.chf<self.chi:
            self.chf = self.chi + 1
            self.crspinbox2.setValue(self.chf)
        # Update the image display
        self.ax_image.clear()
        self.image = np.sum(self.volume[:,:,self.chi:self.chf], axis = 2)
        self.image = 100*(self.image/np.max(self.image))
        self.mask = np.copy(self.image)
        self.mask[self.mask<5] = 0
        self.mask[self.mask>0] = 1
        self.ax_image.imshow(self.image.T, cmap=self.cmap)
        self.ax_image.set_title("ROI image")
        self.canvas_image.draw()

    def plot_mean_image_mean_bkg(self):

        """
        Compute and display a mean ROI image with constant background subtraction.

        The background is estimated as the mean of the start (`chi`) and end (`chf`) channels,
        and subtracted from the sum across the selected ROI channel range.

        Steps:
        - Normalize the ROI volume.
        - Subtract mean of boundary channels from the summed signal.
        - Clip negative values and rescale to 0-100.
        - Generate a binary mask (thresholded at 5%).
        - Display the processed image in the main viewer.
        """
        
        # Ensure chi < chf and both are within volume bounds
        n_channels = self.volume.shape[2]

        if self.chi > self.chf:
            self.chi, self.chf = self.chf, self.chi
            if hasattr(self, 'crspinbox1') and hasattr(self, 'crspinbox2'):
                self.crspinbox1.setValue(self.chi)
                self.crspinbox2.setValue(self.chf)

        self.chi = max(0, min(self.chi, n_channels - 1))
        self.chf = max(self.chi + 1, min(self.chf, n_channels))
        
        # Update the image display
        self.ax_image.clear()
        roi = np.arange(self.chi,self.chf)
        # Compute central position as float
        pos = 0.5 * (self.chi + self.chf)

        # Set Tab 4 controls automatically
        self.pos_spin.setValue(pos)
        self.pos_min_spin.setValue(max(0.0, pos - 5.0))
        self.pos_max_spin.setValue(pos + 5.0)

        self.volroi = self.volume[:,:,roi]
        self.volroi = self.volroi/np.max(self.volroi)
        
        self.xroi = np.arange(self.chi, self.chf)
        self.image = np.sum(self.volume[:,:,roi],axis=2)-len(roi)*np.mean(self.volume[:,:,[self.chi,self.chf]],axis=2)
        self.image[self.image<0] = 0
        self.image = 100*(self.image/np.max(self.image))
        self.mask = np.copy(self.image)
        self.mask[self.mask<5] = 0
        self.mask[self.mask>0] = 1
        self.ax_image.imshow(self.image.T, cmap=self.cmap)
        self.ax_image.set_title("ROI image")
        self.canvas_image.draw()
        
    def plot_mean_image_linear_bkg(self):

        """
        Compute and display a mean ROI image with linear background subtraction.

        The background is estimated per-pixel as a linear interpolation between the values
        at the start (`chi`) and end (`chf`) channels. This estimated background is subtracted 
        from the raw ROI before summing.

        Steps:
        - Fit a linear background for each pixel across the ROI channel range.
        - Subtract the background from the ROI signal.
        - Clip negative values and rescale to 0-100.
        - Create a binary mask (values above 5%).
        - Display the result in the image viewer.
        """
        
        # Ensure chi < chf and both are within volume bounds
        n_channels = self.volume.shape[2]

        if self.chi > self.chf:
            self.chi, self.chf = self.chf, self.chi
            if hasattr(self, 'crspinbox1') and hasattr(self, 'crspinbox2'):
                self.crspinbox1.setValue(self.chi)
                self.crspinbox2.setValue(self.chf)

        self.chi = max(0, min(self.chi, n_channels - 1))
        self.chf = max(self.chi + 1, min(self.chf, n_channels))
        
        # Update the image display
        self.ax_image.clear()
        roi = np.arange(self.chi,self.chf)
        
        # Compute central position as float
        pos = 0.5 * (self.chi + self.chf)

        # Set Tab 4 controls automatically
        self.pos_spin.setValue(pos)
        self.pos_min_spin.setValue(max(0.0, pos - 5.0))
        self.pos_max_spin.setValue(pos + 5.0)        
        
        self.xroi = np.arange(self.chi, self.chf)
        self.volroi = self.volume[:,:,roi]
        slope = (self.volroi[:,:,-1] - self.volroi[:,:,0])/len(roi)
        slope = np.reshape(slope, (self.image_width*self.image_height,1))
        inter = self.volroi[:,:,0]
        inter = np.reshape(inter, (self.image_width*self.image_height,1))
        volroi = np.reshape(self.volroi, (self.image_width*self.image_height,len(roi)))    
        self.volroi = self.volroi/np.max(self.volroi)
        self.image = volroi - (slope*np.arange(0,len(roi)) + inter)
        self.image = np.sum(self.image, axis = 1)
        self.image = np.reshape(self.image, (self.image_width, self.image_height))
        self.image[self.image<0] = 0
        self.image = 100*(self.image/np.max(self.image))
        self.mask = np.copy(self.image)
        self.mask[self.mask<5] = 0
        self.mask[self.mask>0] = 1
        self.ax_image.imshow(self.image.T, cmap=self.cmap)
        self.ax_image.set_title("ROI image")
        self.canvas_image.draw()
        
 
    def export_roi_image(self):

        """
        Export the currently displayed ROI image to disk.

        This method saves both the raw image data and a rendered PNG:
        - HDF5 file containing the ROI image (`I`) and the selected channel range (`chi`, `chf`).
        - PNG image of the ROI visualized with the current colormap.

        The filenames are generated based on the original HDF5 filename and the ROI range.

        Notes
        -----
        This function assumes that `self.image` and `self.hdf_fileName` are valid.
        If no image or filename is available, it prints a warning and exits.
        """
        
        if len(self.hdf_fileName)>0 and len(self.image)>0:
            s = self.hdf_fileName.split('.h5'); s = s[0]
            sn = "%s_roi_%sto%s.h5" %(s,str(self.chi), str(self.chf))
            print(sn)
            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.image)
            h5f.create_dataset('chi', data=self.chi)            
            h5f.create_dataset('chf', data=self.chf)            
            h5f.close()
            sn = "%s_roi_%sto%s.png" %(s,str(self.chi), str(self.chf))
            plt.imsave(sn,self.image,cmap=self.cmap)
        else:
            print("Something is wrong with the data")
            
    ####################### ROI pattern #######################

    def set_thr(self, value):
        """
        Set the threshold value used for segmenting the ROI image.

        Parameters
        ----------
        value : int
            Threshold value (0–100) used to binarize the ROI image.
        """        
        self.thr = value
        print(self.thr)

    def segment_image(self):
        """
        Segment the ROI image using the current threshold value.

        Applies a simple threshold to the currently displayed ROI image to generate a binary mask.
        Pixels below the threshold are set to 0, others to 1. The resulting mask is displayed in the image view.
        """        
        # Update the image display
        self.ax_image.clear()
        self.mask = np.copy(self.image)
        self.mask[self.mask<self.thr] = 0
        self.mask[self.mask>0] = 1
        self.ax_image.imshow(self.mask.T, cmap=self.cmap)
        self.ax_image.set_title("Mask")
        self.canvas_image.draw()
    
    def plot_roi_pattern(self):
        
        """
        Compute and display the ROI-averaged pattern based on the current segmentation mask.

        Multiplies each spectral slice by the binary mask to isolate the ROI region,
        then integrates across all pixels to generate a 1D spectrum. The spectrum is
        normalized and plotted in the spectral view.
        """        
        voln = np.zeros_like(self.volume)
        for ii in range(self.volume.shape[2]):
            voln[:,:,ii] = self.volume[:,:,ii]*self.mask

        self.spectrum = np.sum(voln, axis = (0,1))
        self.spectrum = self.spectrum/np.max(self.spectrum)
        self.ax_spectrum.clear()
        self.ax_spectrum.plot(self.xaxis, self.spectrum, color='b')
        self.ax_spectrum.set_title("ROI pattern")
        self.ax_spectrum.set_xlabel(self.xaxislabel)
        self.canvas_spectrum.draw()
            
    def export_roi_pattern(self):
        
        """
        Export the segmented ROI-averaged pattern to multiple file formats.

        Saves the 1D ROI pattern (`self.spectrum`) and corresponding x-axis values (`self.xaxis`)
        to an HDF5 file (`.h5`) and two plain text formats (`.asc` and `.xy`).
        Filenames include the current threshold value used for segmentation.

        Notes
        -----
        This method will print a warning if `self.spectrum` is empty or no HDF5 file was previously loaded.
        """        
        if len(self.hdf_fileName)>0 and len(self.spectrum)>0:
            
            s = self.hdf_fileName.split('.h5'); s = s[0]
            sn = "%s_ROI_thr%s.h5" %(s,str(self.thr))
            print(sn)

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.spectrum)
            h5f.create_dataset('xaxis', data=self.xaxis)
            h5f.close()
        
            xy = np.column_stack((self.xaxis,self.spectrum))
            sn = "%s_ROI_thr%s.asc" %(s,str(self.thr))
            np.savetxt(sn,xy, fmt="%.5f")
            
            xy = np.column_stack((self.xaxis,self.spectrum))
            sn = "%s_ROI_thr%s.xy" %(s,str(self.thr))
            np.savetxt(sn,xy, fmt="%.5f")
                
        else:
            print("Something is wrong with the data")

    def suggest_peak_positions(self):
        """
        Automatically detect and visualize candidate peak positions in the ROI-averaged spectrum.

        Uses `scipy.signal.find_peaks` to identify prominent peaks based on height and spacing thresholds.
        Detected peaks are displayed as vertical lines overlaid on the spectrum plot:
            - The first peak is highlighted in red.
            - All other peaks are shown in green.

        The image viewer is updated to show the spectral slice corresponding to the first detected peak.
        Detected peak positions are also stored internally for future reference.

        Raises
        ------
        Warning dialog if no spectrum is available or no peaks are found.
        """
        if self.spectrum is None or len(self.spectrum) == 0:
            QtWidgets.QMessageBox.warning(self, "No Spectrum", "Please extract the ROI pattern first.")
            return

        # Detect peaks
        peaks, _ = find_peaks(self.spectrum, height=0.1 * np.max(self.spectrum), distance=3)

        if len(peaks) == 0:
            QtWidgets.QMessageBox.information(self, "No Peaks Found", "No peaks could be detected in the spectrum.")
            return

        # Display peak list
        msg = "Suggested peak positions:\n"
        for p in peaks:
            msg += f"  {self.xaxis[p]:.2f}\n"
        QtWidgets.QMessageBox.information(self, "Peak Suggestions", msg)

        # Rebuild spectrum figure
        self.fig_spectrum.clear()
        self.ax_spectrum = self.fig_spectrum.add_subplot(111)
        self.ax_spectrum.plot(self.xaxis, self.spectrum, color='b')
        self.ax_spectrum.set_title("ROI pattern with suggested peaks")
        self.ax_spectrum.set_xlabel(self.xaxislabel)

        # Add red vline (first peak)
        self.vline = self.ax_spectrum.axvline(self.xaxis[peaks[0]], color='r', linestyle='--', lw=1)

        # Add green lines for all peaks
        for p in peaks:
            self.ax_spectrum.axvline(x=self.xaxis[p], color='g', linestyle='--', lw=1.2)

        # Reconnect spectrum panel interactivity
        self.canvas_spectrum.mpl_connect('motion_notify_event', self.update_image)
        self.canvas_spectrum.mpl_connect('button_press_event', self.toggle_real_time_image)
        self.canvas_spectrum.mpl_connect('scroll_event', self.on_canvas_scroll)
        self.canvas_spectrum.draw()

        # Force image update to match first peak
        self.index = peaks[0]
        self.image = self.volume[:, :, self.index]

        # Recreate image plot fully to ensure canvas update
        self.fig_image.clear()
        self.ax_image = self.fig_image.add_subplot(111)

        self.im = self.ax_image.imshow(self.image.T, cmap=self.cmap)
        self.ax_image.set_title(f"Image: Channel {self.index}")
        self.cbar = self.fig_image.colorbar(self.im, ax=self.ax_image, fraction=0.046, pad=0.04)

        self.canvas_image.mpl_connect('motion_notify_event', self.update_spectrum)
        self.canvas_image.mpl_connect('button_press_event', self.toggle_real_time_spectrum)
        self.canvas_image.mpl_connect('scroll_event', self.on_canvas_scroll)

        self.canvas_image.draw()

        # Store peaks for future use
        self.detected_peaks = peaks
        
    ####################### Peak fitting #######################
    
    def set_fit_range(self):
        """
        Define the spectral fitting region and normalize the data volume accordingly.

        This method updates the region of interest (`xroi`, `volroi`) based on the 
        spinbox values for initial and final channels. It also normalizes the ROI data 
        and updates the peak position spinboxes to reflect the new fitting range.

        Raises
        ------
        QtWidgets.QMessageBox
            If the selected channel range is invalid (i.e., final ≤ initial).
        """
        ch_min = self.xfit_min_spin.value()
        ch_max = self.xfit_max_spin.value()

        if ch_max <= ch_min:
            QtWidgets.QMessageBox.warning(self, "Invalid Range", "Final channel must be greater than initial channel.")
            return

        self.xroi = np.arange(ch_min, ch_max)
        self.volroi = self.volume[:, :, ch_min:ch_max]
        self.volroi = self.volroi / np.max(self.volroi)

        # Set peak position controls based on the range
        pos = 0.5 * (ch_min + ch_max)
        pos_min = ch_min
        pos_max = ch_max

        self.pos_spin.setValue(pos)
        self.pos_min_spin.setValue(pos_min)
        self.pos_max_spin.setValue(pos_max)

        print(f"Fitting range set to channels {ch_min} to {ch_max} ({ch_max - ch_min} channels)")
        print(f"Updated peak position guess: {pos:.2f} [{pos_min:.2f}, {pos_max:.2f}]")
        
    def sync_peak_position_from_roi(self):
        """
        Automatically update peak position guess based on ROI channel selection.

        This method sets the peak position (`pos_spin`) and bounds (`pos_min_spin`, `pos_max_spin`)
        using the midpoint of the selected channel range in the ROI tab. It helps keep peak fitting
        parameters in sync with image generation settings.
        """        
        chi = self.crspinbox1.value()
        chf = self.crspinbox2.value()
        if chf <= chi:
            return
        pos = 0.5 * (chi + chf)
        self.pos_spin.setValue(pos)
        self.pos_min_spin.setValue(max(0.0, pos - 5.0))
        self.pos_max_spin.setValue(pos + 5.0)
        
    def profile_function(self, ind):
        """
        Set the peak profile type for curve fitting based on user selection.

        Updates internal `peaktype` and toggles the visibility of the mixing fraction
        (γ) parameter used for Pseudo-Voigt profiles.

        Parameters
        ----------
        ind : int
            Index from the dropdown selector:
            - 0: Gaussian
            - 1: Lorentzian
            - 2: Pseudo-Voigt
        """        
        if ind == 0:
            self.peaktype = "Gaussian"
            print("Gaussian profile")
            self.label_fraction.setVisible(False)
            self.fraction_spin.setVisible(False)
        elif ind == 1:
            self.peaktype = "Lorentzian"
            print("Lorentzian profile")
            self.label_fraction.setVisible(False)
            self.fraction_spin.setVisible(False)
        elif ind == 2:
            self.peaktype = "Pseudo-Voigt"
            print("Pseudo-Voigt profile")
            self.label_fraction.setVisible(True)
            self.fraction_spin.setVisible(True)

    def batchpeakfit(self):     
        
        """
        Perform batch single-peak fitting over the masked region of the dataset.

        This method collects all peak fitting parameters from the GUI (initial guesses and bounds)
        and applies them to the currently selected volume region (`volroi`) using a user-defined
        peak model (Gaussian, Lorentzian, or Pseudo-Voigt). It applies the segmentation mask to restrict
        the fitting region, initializes a `FitData` thread for asynchronous processing, and connects
        the progress and result signals for GUI updates.

        Notes
        -----
        - If the fitting type is Pseudo-Voigt, the mixing fraction is also passed to the fitting class.
        - The fitting thread is non-blocking and results are streamed progressively.
        - Disables interactive diagnostic display and the "Fit" button until completion.
        """
        
        self.check_diag_mode.setEnabled(False)
        
        self.pbutton_fit.setEnabled(False)

        # Read GUI values
        self.Area = self.area_spin.value()
        self.Areamin = self.area_min_spin.value()
        self.Areamax = self.area_max_spin.value()

        self.FWHM = self.fwhm_spin.value()
        self.FWHMmin = self.fwhm_min_spin.value()
        self.FWHMmax = self.fwhm_max_spin.value()

        self.Pos = self.pos_spin.value()
        self.Posmin = self.pos_min_spin.value()
        self.Posmax = self.pos_max_spin.value()

        # Override Posmin/Posmax if custom set
        if self.pos_min_spin.value() < self.pos_max_spin.value():
            self.Posmin = self.pos_min_spin.value()
            self.Posmax = self.pos_max_spin.value()

        if self.mask is None or not isinstance(self.mask, np.ndarray):
            print("No mask found, using default mask (all 1s)")
            self.mask = np.ones((self.volume.shape[0], self.volume.shape[1]))   
                    
        for ii in range(self.volroi.shape[2]):
            self.volroi[:,:,ii] = self.volroi[:,:,ii]*self.mask
        
        fraction = self.fraction_spin.value() if self.peaktype == "Pseudo-Voigt" else None
    
        self.PeakFitData = FitData(self.peaktype, self.volroi, self.xroi,
                                self.Area, self.Areamin, self.Areamax,
                                self.Pos, self.Posmin, self.Posmax,
                                self.FWHM, self.FWHMmin, self.FWHMmax)

        if self.peaktype == "Pseudo-Voigt":
            self.PeakFitData.fraction_init = fraction
        self.PeakFitData.start()            
        self.PeakFitData.progress_fit.connect(self.progressbar_fit.setValue)
        self.PeakFitData.result_partial.connect(self.update_live_fit_image)
        self.PeakFitData.fitdone.connect(self.updatefitdata)        

    def update_live_fit_image(self, live_data):
        
        """
        Update the image display with intermediate peak fitting results.

        This method is triggered during batch fitting to provide real-time visual feedback
        of the parameter currently selected in the "Live view" dropdown (Area, Position, or FWHM).
        It redraws the image canvas using the latest fitting results for the selected parameter
        and reconnects all interactive events to maintain responsiveness.

        Parameters
        ----------
        live_data : np.ndarray
            A 2D array representing the latest fitting result for a single parameter (typically area),
            used if no predefined mapping is selected.
        """        
        param = self.ChooseLive.currentText()

        if param == "Area":
            img = self.PeakFitData.phase
            vmin, vmax = None, None  # autoscale
        elif param == "Position":
            img = self.PeakFitData.cen
            vmin, vmax = self.Posmin, self.Posmax
        elif param == "FWHM":
            img = self.PeakFitData.wid
            vmin, vmax = None, None  # autoscale
        else:
            img = live_data
            vmin, vmax = np.nanmin(img), np.nanmax(img)

        # Recreate the figure to avoid layout shrinking
        self.fig_image.clear()
        self.ax_image = self.fig_image.add_subplot(111)

        if vmin is not None and vmax is not None:
            self.im = self.ax_image.imshow(img.T, cmap='jet', vmin=vmin, vmax=vmax)
        else:
            self.im = self.ax_image.imshow(img.T, cmap='jet')

        self.ax_image.set_title(f"Live fit: {param}")
        self.cbar = self.fig_image.colorbar(self.im, ax=self.ax_image, fraction=0.046, pad=0.04)

        # IMPORTANT: reconnect interactions here to restore hover after each update
        self.canvas_image.mpl_connect('motion_notify_event', self.update_spectrum)
        self.canvas_image.mpl_connect('button_press_event', self.toggle_real_time_spectrum)
        self.canvas_image.mpl_connect('scroll_event', self.on_canvas_scroll)

        self.canvas_image.draw()
        
    def updatefitdata(self):

        """
        Finalize and display peak fitting results after batch processing.

        This method is called when the fitting thread signals completion. It updates the internal
        `res` dictionary with the final results (e.g. Area, Position, FWHM), re-enables UI controls,
        and redraws the image panel to show the fitted peak area map. It also reconnects interactive
        mouse events to restore full responsiveness of the GUI.
        """
                
        if not hasattr(self.PeakFitData, 'res'):
            print("Peak fitting was stopped before completion.")
            return

        try:
            self.PeakFitData.result_partial.disconnect()
        except TypeError:
            pass

        self.res = self.PeakFitData.res
        self.pbutton_fit.setEnabled(True)
        self.ChooseRes.setEnabled(True)
        self.pbutton_expfit.setEnabled(True)

        # Recreate figure and axis
        self.fig_image.clear()
        self.ax_image = self.fig_image.add_subplot(111)

        img = self.res['Area']
        self.im = self.ax_image.imshow(img.T, cmap='jet', vmin=self.Areamin, vmax=self.Areamax)
        self.ax_image.set_title("Peak area")

        self.cbar = self.fig_image.colorbar(self.im, ax=self.ax_image, fraction=0.046, pad=0.04)

        # Reconnect event handlers (so update_image still works)
        self.canvas_image.mpl_connect('motion_notify_event', self.update_spectrum)
        self.canvas_image.mpl_connect('button_press_event', self.toggle_real_time_spectrum)
        self.canvas_image.mpl_connect('scroll_event', self.on_canvas_scroll)

        self.canvas_image.draw()

        self.check_diag_mode.setEnabled(True)
        
    def stopfit(self):
        """
        Stop the ongoing peak fitting process.

        Requests the fitting thread to stop gracefully, resets the progress bar,
        and re-enables the fit button for user interaction.
        """        
        if hasattr(self, 'PeakFitData'):
            self.PeakFitData.request_stop()
        self.progressbar_fit.setValue(0)
        self.pbutton_fit.setEnabled(True)        
        
    def plot_fit_results(self, ind):
        """
        Plot a selected parameter map from the peak fitting results.

        Parameters
        ----------
        ind : int
            Index of the parameter to display:
            - 0: Peak area
            - 1: Peak position
            - 2: FWHM
            - 3: Background 1 slope
            - 4: Background 2 intercept

        The method updates the image panel with the selected map and reconnects
        interactive controls.
        """        
        self.fig_image.clear()
        self.ax_image = self.fig_image.add_subplot(111)

        # Map selection to data and optional bounds
        if ind == 0:
            label, data = "Area", self.res['Area']
            vmin, vmax = None, None  # autoscale
        elif ind == 1:
            label, data = "Position", self.res['Position']
            vmin, vmax = self.Posmin, self.Posmax  # fixed
        elif ind == 2:
            label, data = "FWHM", self.res['FWHM']
            vmin, vmax = None, None  # autoscale
        elif ind == 3:
            label, data = "Background 1", self.res['Background1']
            vmin, vmax = None, None  # autoscale
        elif ind == 4:
            label, data = "Background 2", self.res['Background2']
            vmin, vmax = None, None  # autoscale

        # Plot with optional scaling
        if vmin is not None and vmax is not None:
            self.im = self.ax_image.imshow(data.T, cmap='jet', vmin=vmin, vmax=vmax)
        else:
            self.im = self.ax_image.imshow(data.T, cmap='jet')

        self.ax_image.set_title(label)

        self.cbar = self.fig_image.colorbar(self.im, ax=self.ax_image, fraction=0.046, pad=0.04)

        # Reconnect interactivity
        self.canvas_image.mpl_connect('motion_notify_event', self.update_spectrum)
        self.canvas_image.mpl_connect('button_press_event', self.toggle_real_time_spectrum)
        self.canvas_image.mpl_connect('scroll_event', self.on_canvas_scroll)

        self.canvas_image.draw()
        
    def savefitresults(self):
        
        """
        Export the peak fitting results to an HDF5 file.

        Saves the fitted parameter maps (area, position, FWHM, background slope and intercept)
        to a new `.h5` file using the current dataset filename as a base. If the Pseudo-Voigt
        profile is used, the mixing parameter ('gamma') is also saved.

        The resulting file contains:
            - 'Area': Peak area map
            - 'Position': Peak center position map
            - 'FWHM': Full width at half maximum map
            - 'bkga': Background slope map
            - 'bkgb': Background intercept map
            - 'gamma' (if applicable): Mixing fraction between Gaussian and Lorentzian

        Notes
        -----
        This method assumes a dataset has been previously loaded and fitting has been completed.
        """
        if len(self.hdf_fileName)>0:
            s = self.hdf_fileName.split('.h5'); s = s[0]
            sn = "%s_fit_results.h5" %(s)
            with h5py.File(sn, 'w') as h5f:            
                h5f.create_dataset('Area', data=self.res['Area'])
                h5f.create_dataset('Position', data=self.res['Position'])
                h5f.create_dataset('FWHM', data=self.res['FWHM'])
                h5f.create_dataset('bkga', data=self.res['Background1'])
                h5f.create_dataset('bkgb', data=self.res['Background2'])
                if self.peaktype == "Pseudo-Voigt":                
                    h5f.create_dataset('gamma', data=self.res['Fraction'])            
        else:
            print("Something is wrong with the data")
                    

    ####################### Create synthetic phantom #######################

    def create_phantom(self):
        
        """
        Generate and load a synthetic phantom XRD-CT dataset into the GUI.

        This method creates a 3D synthetic chemical imaging volume based on known diffraction
        patterns of common elements (Al, Cu, Fe, Pt, Zn). It uses pre-defined 2D spatial templates
        and overlays them with their respective spectral signatures to construct a realistic
        simulated XRD-CT dataset.

        The generated volume is loaded into the viewer, and internal data structures are updated
        (including axis labels, spinbox limits, and dataset name). An initial visualization is
        triggered via `self.explore()`.

        If the process fails (e.g. missing dependencies or runtime errors), a critical message box is shown.

        Notes
        -----
        This function relies on modules within the `nDTomo.sim` and `nDTomo.methods` namespaces.
        """        
        
        try:
            from nDTomo.sim.phantoms import load_example_patterns, nDTomophantom_2D, nDTomophantom_3D
            from nDTomo.methods.plots import showspectra, showim

            # Load example patterns
            dpAl, dpCu, dpFe, dpPt, dpZn, tth, q = load_example_patterns()
            spectra = [dpAl, dpCu, dpFe, dpPt, dpZn]

            # Generate 2D images
            npix = 200
            imAl, imCu, imFe, imPt, imZn = nDTomophantom_2D(npix, nim='Multiple')
            iml = [imAl, imCu, imFe, imPt, imZn]

            # Create synthetic 3D dataset
            chemct = nDTomophantom_3D(npix, use_spectra='Yes', spectra=spectra, imgs=iml, indices='All', norm='No')

            self.volume = chemct
            self.xaxis = tth
            self.xaxislabel = '2theta'
            self.image_width, self.image_height, self.nbins = chemct.shape
            self.DatasetNameLabel.setText("Synthetic Phantom")
            self.crspinbox1.setMaximum(self.nbins - 1)
            self.crspinbox2.setMaximum(self.nbins)
            self.explore()

            QtWidgets.QMessageBox.information(self, "Phantom Created", "Synthetic phantom XRD-CT dataset has been created.")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to create phantom:\n{e}")
            
    ######################## IPython console #######################
    
    def init_console(self):
        
        """
        Launch an embedded IPython console inside the GUI.

        This method creates a dockable widget hosting a fully interactive IPython
        (Jupyter) console. The console allows the user to inspect and manipulate
        GUI-level variables such as `volume`, `image`, `spectrum`, and `xaxis`
        using NumPy and Matplotlib interactively within the same application.

        If the console has already been initialized, this method brings it back into view.

        Notes
        -----
        The console is powered by `qtconsole` and runs an in-process kernel,
        meaning it shares memory with the main application context.
        """        
        if hasattr(self, 'console_dock') and self.console_dock is not None:
            self.console_dock.show()
            return

        self.console_dock = QtWidgets.QDockWidget("IPython Console", self)
        self.console_dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        self.console_widget = RichJupyterWidget()
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt'
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        self.kernel.shell.push({
            'gui': self,
            'volume': self.volume,
            'image': self.image,
            'spectrum': self.spectrum,
            'xaxis': self.xaxis,
            'np': np,
            'plt': plt
        })

        self.console_widget.kernel_manager = self.kernel_manager
        self.console_widget.kernel_client = self.kernel_client
        self.console_widget.exit_requested.connect(self.stop_console)

        self.console_dock.setWidget(self.console_widget)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.console_dock)

    def stop_console(self):
        """
        Shut down the embedded IPython console and its kernel.

        This method stops the running IPython kernel and disconnects the console client.
        It is triggered when the user exits the docked console interface.
        """        
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()        


class FileDialog(QtWidgets.QFileDialog):
    
    """
    Initialize the custom file dialog with multi-selection support.

    This constructor configures the dialog to:
    - Use a non-native file dialog (to allow enhanced widget access).
    - Enable multi-selection mode for both QListView and QTreeView components,
      allowing the user to select multiple files at once.
    
    Parameters
    ----------
    *args : tuple
        Arguments passed to the base QFileDialog constructor.
    """

    def __init__(self, *args):
        QtWidgets.QFileDialog.__init__(self, *args)
        self.setOption(self.DontUseNativeDialog, True)
        for view in self.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
            if isinstance(view.model(), QtWidgets.QFileSystemModel):
                view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)


class FitData(QThread):
    
    """
    Worker thread for batch single-peak fitting on hyperspectral data.

    This class performs pixel-wise non-linear fitting of a single peak model (with linear background)
    using SciPy's curve_fit. Runs asynchronously and emits progress and results.

    Parameters
    ----------
    peaktype : str
        Type of peak profile to fit ('Gaussian', 'Lorentzian', 'Pseudo-Voigt').
    data : np.ndarray
        3D hyperspectral data to fit (X, Y, Channels).
    x : np.ndarray
        1D array of x-axis values corresponding to spectral channels.
    Area, Areamin, Areamax : float
        Initial, minimum, and maximum values for peak area.
    Pos, Posmin, Posmax : float
        Initial, minimum, and maximum values for peak position.
    FWHM, FWHMmin, FWHMmax : float
        Initial, minimum, and maximum values for full width at half maximum.

    Signals
    -------
    fitdone : pyqtSignal
        Emitted when the fitting is complete.
    progress_fit : pyqtSignal(int)
        Emits fitting progress (0–100).
    result_partial : pyqtSignal(np.ndarray)
        Emits a partial result image (area) during processing.
    """
    
    fitdone = pyqtSignal()
    progress_fit = pyqtSignal(int)
    result_partial = pyqtSignal(np.ndarray)
    
    def __init__(self, peaktype, data, x, Area, Areamin, Areamax, Pos, Posmin, Posmax, FWHM, FWHMmin, FWHMmax):
        
        """
        Initialize the FitData worker thread for single-peak fitting.

        This sets up the initial parameters and allocates output arrays for the
        batch fitting of hyperspectral or tomographic data. The fitting is done
        using SciPy's non-linear curve fitting (`curve_fit`) on a per-pixel basis.

        Parameters
        ----------
        peaktype : str
            Type of peak to fit; must be one of: "Gaussian", "Lorentzian", "Pseudo-Voigt".
        data : np.ndarray
            3D array of shape (X, Y, Channels) representing the volume to fit.
        x : np.ndarray
            1D array of length Channels representing the spectral axis.
        Area : float
            Initial guess for peak area.
        Areamin : float
            Minimum allowed value for peak area.
        Areamax : float
            Maximum allowed value for peak area.
        Pos : float
            Initial guess for peak position (in x units).
        Posmin : float
            Minimum allowed value for peak position.
        Posmax : float
            Maximum allowed value for peak position.
        FWHM : float
            Initial guess for full width at half maximum.
        FWHMmin : float
            Minimum allowed FWHM.
        FWHMmax : float
            Maximum allowed FWHM.
        """        
        QThread.__init__(self)
        self._stop_requested = False
        
        self.peaktype = peaktype
        self.fitdata = data  
        
        shape = (self.fitdata.shape[0], self.fitdata.shape[1])
        self.phase = np.zeros(shape, dtype='float32')
        self.cen = np.full(shape, Pos, dtype='float32')
        self.wid = np.full(shape, FWHMmin, dtype='float32')
        self.bkg1 = np.zeros(shape, dtype='float32')
        self.bkg2 = np.zeros(shape, dtype='float32')
        self.fr = np.full(shape, 0.5, dtype='float32')     
        
        self.xroi = x
        self.Area = Area; self.Areamin = Areamin; self.Areamax = Areamax
        self.Pos = Pos; self.Posmin = Posmin; self.Posmax = Posmax
        self.FWHM = FWHM; self.FWHMmin = FWHMmin; self.FWHMmax = FWHMmax
        
        msk = np.sum(self.fitdata,axis = 2)
        msk[msk>0] = 1
        self.i, self.j = np.where(msk > 0)

    def request_stop(self):
        """
        Signal the fitting thread to stop.

        Sets an internal flag that is checked periodically during fitting.
        Allows the fitting process to be gracefully interrupted without killing the thread.
        """
                
        self._stop_requested = True
    
    def run(self):
        
        """
        Entry point for the QThread.

        Invokes the batch fitting routine (`batchfit`) when the thread is started.
        """
        self.batchfit()
        
    def batchfit(self):
                
        """
        Perform batch pixel-wise single-peak fitting over a 3D hyperspectral dataset.

        The method fits a Gaussian, Lorentzian, or pseudo-Voigt peak with a linear background
        to each spectrum in the dataset where the mask is non-zero. It uses SciPy's `curve_fit`
        for non-linear least squares optimization with parameter bounds. Progress is emitted
        after each row to support responsive GUI updates.

        For each valid pixel:
        - Attempts to fit the selected peak model.
        - On success, stores peak parameters (area, position, FWHM, background terms, and Voigt fraction if applicable).
        - On failure, stores midpoint values for peak position and width as fallback.

        Emits
        -----
        progress_fit : pyqtSignal(int)
            Signal emitting fitting progress in percentage (0–100).
        result_partial : pyqtSignal(np.ndarray)
            Signal emitting current 'Area' map (transposed) for live updating.
        fitdone : pyqtSignal()
            Signal emitted upon completion of fitting.

        Notes
        -----
        - Uses `request_stop()` flag to allow asynchronous termination from GUI.
        - Results are stored in the `self.res` dictionary.
        - The method ensures fitting is done only where data is present (non-zero mask).
        """
                        
        if self.peaktype == "Pseudo-Voigt":
            x0 = np.array([float(self.Area), float(self.Pos), float(self.FWHM), 0., 0., 0.5], dtype=float)
        else:
            x0 = np.array([float(self.Area), float(self.Pos), float(self.FWHM), 0., 0.], dtype=float)
        
        if self.peaktype == "Pseudo-Voigt":
            param_bounds=([float(self.Areamin),float(self.Posmin),float(self.FWHMmin),-0.5,-1.0,0],
                          [float(self.Areamax),float(self.Posmax),float(self.FWHMmax),0.5,1.0,1])
        else:
            param_bounds=([float(self.Areamin),float(self.Posmin),float(self.FWHMmin),-0.5,-1.0],
                          [float(self.Areamax),float(self.Posmax),float(self.FWHMmax),0.5,1.0])
        
        rows = np.unique(self.i)

        if len(rows) == 0:
            print("No data to fit (mask may be empty)")
            self.progress_fit.emit(0)
            return

        for irow, row in enumerate(rows):
            
            if self._stop_requested:
                break                
            cols = self.j[self.i == row]

            for jj in cols:
                
                if self._stop_requested:
                    break                
                dp = self.fitdata[row, jj, :]
                
                try:
                    if self.peaktype == "Gaussian":
                        params, _ = sciopt.curve_fit(self.gmodel, self.xroi, dp, p0=x0, bounds=param_bounds)
                    elif self.peaktype == "Lorentzian":
                        params, _ = sciopt.curve_fit(self.lmodel, self.xroi, dp, p0=x0, bounds=param_bounds)
                    elif self.peaktype == "Pseudo-Voigt":
                        params, _ = sciopt.curve_fit(self.pvmodel, self.xroi, dp, p0=x0, bounds=param_bounds)

                    self.phase[row, jj] = params[0]
                    self.cen[row, jj] = params[1]
                    self.wid[row, jj] = params[2]
                    self.bkg1[row, jj] = params[3]
                    self.bkg2[row, jj] = params[4]
                    if self.peaktype == "Pseudo-Voigt":
                        self.fr[row, jj] = params[5]

                except Exception:
                    self.cen[row, jj] = (param_bounds[0][1] + param_bounds[1][1]) / 2
                    self.wid[row, jj] = (param_bounds[0][2] + param_bounds[1][2]) / 2

            # Emit the current partial result image after this row
            self.result_partial.emit(self.phase.T.copy())

            # Progress bar update
            progress = int(100 * (irow + 1) / len(rows))
            self.progress_fit.emit(progress)
        

        self.phase = np.where(self.phase<0,0,self.phase)
        
        if self.peaktype == "Pseudo-Voigt":
            self.res = {'Area':self.phase, 'Position':self.cen, 'FWHM':self.wid, 'Background1':self.bkg1, 'Background2':self.bkg2, 'Fraction':self.fr}
        else:
            self.res = {'Area':self.phase, 'Position':self.cen, 'FWHM':self.wid, 'Background1':self.bkg1, 'Background2':self.bkg2}
        
        self.fitdone.emit()

    def gmodel(self, x, A, m, w, a, b):
        
        """
        Gaussian peak model with linear background.

        Parameters
        ----------
        x : np.ndarray
            The x-axis values (e.g., channel, energy, or 2θ).
        A : float
            Area under the Gaussian peak.
        m : float
            Peak center (mean).
        w : float
            Full width at half maximum (FWHM) of the peak.
        a : float
            Linear background slope.
        b : float
            Linear background intercept.

        Returns
        -------
        np.ndarray
            The evaluated Gaussian function with background.
        """
        return (A / (np.sqrt(2 * np.pi) * w)) * np.exp(- (x - m)**2 / (2 * w**2)) + a * x + b    
    
    def lmodel(self, x, A, m, w, a, b):
        
        """
        Lorentzian peak model with linear background.

        Parameters
        ----------
        x : np.ndarray
            The x-axis values.
        A : float
            Area under the Lorentzian peak.
        m : float
            Peak center.
        w : float
            FWHM of the peak.
        a : float
            Linear background slope.
        b : float
            Linear background intercept.

        Returns
        -------
        np.ndarray
            The evaluated Lorentzian function with background.
        """
        return (A / (1 + ((x - m) / w)**2)) / (np.pi * w) + a * x + b
    
    def pvmodel(self, x, A, m, w, a, b, fr):
        
        """
        Pseudo-Voigt peak model with linear background.

        This is a linear combination of a Gaussian and Lorentzian profile.

        Parameters
        ----------
        x : np.ndarray
            The x-axis values.
        A : float
            Total area under the peak.
        m : float
            Peak center.
        w : float
            FWHM of the peak.
        a : float
            Linear background slope.
        b : float
            Linear background intercept.
        fr : float
            Fraction of Lorentzian contribution (0 = pure Gaussian, 1 = pure Lorentzian).

        Returns
        -------
        np.ndarray
            The evaluated Pseudo-Voigt function with background.
        """
        gauss = (A / (np.sqrt(2 * np.pi) * w)) * np.exp(- (x - m)**2 / (2 * w**2))
        lorentz = (A / (1 + ((x - m) / w)**2)) / (np.pi * w)
        return (1 - fr) * gauss + fr * lorentz + a * x + b


def main():
    qApp = QtWidgets.QApplication(sys.argv)
    aw = nDTomoGUI()
    aw.show()
    sys.exit(qApp.exec_())
    qApp.exec_()
    
if __name__ == "__main__":
    main()
    