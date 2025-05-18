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
u('Qt5Agg')
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
        Initialize the main GUI window.
        This method sets up the main window, initializes various data structures, and creates the file and help menus.
        """
        super(nDTomoGUI, self).__init__()
        
        self.volume = np.zeros(())
        self.xaxis = np.zeros(())
        self.image = np.zeros(())
        self.spectrum = np.zeros(())
        self.hdf_fileName = ''
        self.c = 3E8
        self.h = 6.620700406E-34        
        self.cmap = 'jet'
        self.xaxislabel = 'Channel'
        self.chi = 0
        self.chf = 1        
        self.cmap_list = ['viridis','plasma','inferno','magma','cividis','flag', 
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
        self.file_menu.addAction('&Open Chemical imaging data', self.fileOpen, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.file_menu.addAction('&Append Chemical imaging data', self.append_file)
        self.file_menu.addAction('&Save Chemical imaging data', self.savechemvol)
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
        self.tabs.addTab(self.tab1,"Chemical imaging data")        
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
        self.Labelbkg.setText('Set the threshold to segment the ROI image - range is between 0 and 100')
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
        Compute and display the mean image and mean spectrum from the loaded dataset.

        This method plots the average projection across all channels and overlays the
        average spectrum on the spectrum panel. Enables interactive mouse-based inspection.
        """

        if self.volume.size == 0:
            raise ValueError("No data loaded. Please load an HDF5 file first.")

        # Calculate mean image and spectrum
        mean_image = np.mean(self.volume, axis=2)
        mean_spectrum = np.mean(self.volume, axis=(0, 1))

        self.fig_image.clear()
        self.ax_image = self.fig_image.add_subplot(111)
        self.im = self.ax_image.imshow(mean_image.T, cmap=self.cmap)

        # Add colorbar (or update if it already exists)
        try:
            if self.cbar and self.cbar.ax:
                self.cbar.remove()
                self.cbar = None
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
        Handle the mouse scroll event for zooming in the plots.
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

    # def update_spectrum(self, event):
    #     """
    #     Update the spectrum plot based on the mouse hover event on the image plot.
    #     """
    #     if self.real_time_update_spectrum and event.inaxes == self.ax_image:
    #         self.x, self.y = int(event.xdata), int(event.ydata)

    #         # Check if the mouse position is within the image dimensions
    #         if self.x >= 0 and self.x < self.image_width and self.y >= 0 and self.y < self.image_height:
    #             # Get the spectrum from the volume
    #             self.spectrum = self.volume[self.x, self.y, :]

    #             # Remove existing non-vline lines
    #             for line in self.ax_spectrum.get_lines():
    #                 if line != self.vline:
    #                     line.remove()

    #             # Plot the new spectrum
    #             self.ax_spectrum.plot(self.xaxis, self.spectrum, color='b')
    #             self.ax_spectrum.set_title(f"Histogram: ({self.x}, {self.y})")
    #             self.ax_spectrum.set_xlabel(self.xaxislabel)
    #             self.canvas_spectrum.draw()

    def update_spectrum(self, event):
        """
        Update the spectrum plot based on the mouse hover event on the image plot.
        Overlay fitted peak and residual if diagnostic mode is enabled.
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
        Update the image plot based on the mouse hover event on the spectrum plot.
        """
        if self.real_time_update_image and event.inaxes == self.ax_spectrum:
            self.index = event.xdata

            if self.index >= 0 and self.index < self.nbins:
                try:
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

                    if hasattr(self, 'cbar') and self.cbar is not None:
                        self.cbar.update_normal(self.im)

                    if self.xaxislabel == 'Channel':
                        self.ax_image.set_title(f"Image: Channel {self.index}")
                    else:
                        self.ax_image.set_title("Image: Channel %d, %s %.3f" %
                                                (self.index, self.xaxislabel, self.xaxis[self.index]))

                    # Move vertical line on spectrum plot
                    if hasattr(self, 'vline'):
                        self.vline.set_xdata(self.xaxis[self.index])
                        self.canvas_spectrum.draw()

                    self.canvas_image.draw()

                except Exception as e:
                    print(f"Error in update_image: {e}")

    def toggle_real_time_spectrum(self, event):
        """
        Toggle the real-time update of the spectrum plot based on the mouse button press event on the image plot.
        """
        if event.button == 1:
            self.real_time_update_spectrum = True
        elif event.button == 3:
            self.real_time_update_spectrum = False

    def toggle_real_time_image(self, event):
        """
        Toggle the real-time update of the image plot based on the mouse button press event on the spectrum plot.
        """
        if event.button == 1:
            self.real_time_update_image = True
        elif event.button == 3:
            self.real_time_update_image = False
            
    def exportdp(self):
        
        """
        Method to export spectra/diffraction patterns of interest
        """
        
        if len(self.hdf_fileName)>0 and len(self.spectrum)>0:
            
            s = self.hdf_fileName.split('.h5'); s = s[0]
            sn = "%s_%s_%s.h5" %(s,str(self.x),str(self.y))
            print(sn)

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.spectrum)
            h5f.create_dataset('xaxis', data=self.xaxis)
            h5f.close()
        
            xy = np.column_stack((self.xaxis,self.spectrum))
            sn = "%s_%s_%s.asc" %(s,str(self.x),str(self.y))
            np.savetxt(sn,xy)
            
            xy = np.column_stack((self.xaxis,self.spectrum))
            sn = "%s_%s_%s.xy" %(s,str(self.x),str(self.y))
            np.savetxt(sn,xy)
                
        else:
            print("Something is wrong with the data")
        
    def exportim(self):
        
        """
        Method to export spectral/scattering image of interest
        """
        
        if len(self.hdf_fileName)>0 and len(self.image)>0:
            s = self.hdf_fileName.split('.h5'); s = s[0]
            sn = "%s_channel_%s.h5" %(s,str(self.index))
            print(sn)

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.image)
            h5f.create_dataset('Channel', data=self.index)            
            h5f.close()
        
            sn = "%s_channel_%s.png" %(s,str(self.index))
            plt.imsave(sn,self.image,cmap=self.cmap)
                        
        else:
            print("Something is wrong with the data")
            
                
    def changecolormap(self,ind):
        
        self.cmap = self.cmap_list[ind]
        print(self.cmap)
        try:
            self.update()
        except: 
            pass
        

    def selXRDCTdata(self):
        
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
        Open and load a chemical imaging dataset from an HDF5 file.

        Launches a file dialog, reads the dataset and axis from the selected file,
        updates internal data structures, and triggers the initial visualization.
        """
                
        self.hdf_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Chemical imaging data', "", "*.hdf5 *.h5")

        if len(self.hdf_fileName) > 0:
            self.loadchemvol()
            self.loaded_dataset_names = [self.hdf_fileName.split("/")[-1]]  # <-- track dataset name
            self.DatasetNameLabel.setText(" + ".join(self.loaded_dataset_names))
            self.explore()            
            
                        
    def loadchemvol(self):
        
        """
        Load chemical imaging data from an HDF5 file.

        This function reads an HDF5 file containing hyperspectral imaging data, 
        attempts to identify the correct dataset structure, and updates the GUI.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        KeyError
            If the required dataset ('/data') is not found in the HDF5 file.
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
        Appends another dataset to the currently loaded volume along axis=1 (column-wise),
        ensuring minimal memory usage and displaying combined dataset names.
        """
        append_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Append Chemical imaging data', "", "*.hdf5 *.h5")

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
        dims = self.volume.shape
        if dims[0] != dims[1] and dims[0] == dims[2]:
            # Transpose the array so that the first dimension becomes the last dimension
            self.volume = np.transpose(self.volume, (0, 2, 1))
        elif dims[0] != dims[1] and dims[1] == dims[2]:
            # Transpose the array so that the second dimension becomes the last dimension
            self.volume = np.transpose(self.volume, (1, 2, 0))
            
    def savechemvol(self):

        self.fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Chemical imaging data', "", "*.h5")
	
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
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        message = '<b>nDTomoGUI<p>'
        message += '<p><i>Created by Antony Vamvakeros. Running under license under GPLv3'
        message += '<p>Please cite using the following:<p>'
        message += '<p>Vamvakeros, A. et al., nDTomo software suite, 2019, DOI: https://doi.org/10.5281/zenodo.7139214, url: https://github.com/antonyvam/nDTomo<p>'
        message += '\t '
        d = QtWidgets.QMessageBox()
        d.setWindowTitle('About')
        d.setText(message)
        d.exec_()


    ####################### ROI image #######################

    def channel_initial(self, value):
        self.chi = value
        print(self.chi)

    def channel_final(self, value):
        self.chf = value
        print(self.chf)
        
    def plot_mean_image(self):

        """
        Compute a mean ROI image with linear background subtraction.

        Subtracts a pixel-wise linear estimate of background (based on start and end channels)
        before summing over the ROI range. Applies masking and displays the result.
        """
        
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

        if self.chf<self.chi:
            self.chf = self.chi + 1
            self.crspinbox2.setValue(self.chf)
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

        if self.chf<self.chi:
            self.chf = self.chi + 1
            self.crspinbox2.setValue(self.chf)
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
        Method to export spectral/scattering image of interest
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
        self.thr = value
        print(self.thr)

    def segment_image(self):
        
        # Update the image display
        self.ax_image.clear()
        self.mask = np.copy(self.image)
        self.mask[self.mask<self.thr] = 0
        self.mask[self.mask>0] = 1
        self.ax_image.imshow(self.mask.T, cmap=self.cmap)
        self.ax_image.set_title("Mask")
        self.canvas_image.draw()
    
    def plot_roi_pattern(self):
        
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
            np.savetxt(sn,xy)
            
            xy = np.column_stack((self.xaxis,self.spectrum))
            sn = "%s_ROI_thr%s.xy" %(s,str(self.thr))
            np.savetxt(sn,xy)
                
        else:
            print("Something is wrong with the data")

    def suggest_peak_positions(self):
        """
        Suggest initial peak positions using scipy.signal.find_peaks
        on the currently loaded ROI pattern, and overlay vertical lines.
        Also updates the left image panel to match the first suggested peak.
        """
        if self.spectrum is None or len(self.spectrum) == 0:
            QtWidgets.QMessageBox.warning(self, "No Spectrum", "Please extract the ROI pattern first.")
            return

        from scipy.signal import find_peaks

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
        Sets xroi and volroi based on selected channel range (integer indices).
        Also updates peak position spinboxes based on this range.
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
        chi = self.crspinbox1.value()
        chf = self.crspinbox2.value()
        if chf <= chi:
            return
        pos = 0.5 * (chi + chf)
        self.pos_spin.setValue(pos)
        self.pos_min_spin.setValue(max(0.0, pos - 5.0))
        self.pos_max_spin.setValue(pos + 5.0)
        
    def profile_function(self, ind):
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
        if hasattr(self, 'PeakFitData'):
            self.PeakFitData.request_stop()
        self.progressbar_fit.setValue(0)
        self.pbutton_fit.setEnabled(True)        
        
    def plot_fit_results(self, ind):
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
        Method to export the peak fitting results
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
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()        


class FileDialog(QtWidgets.QFileDialog):
    
    """
    A custom file dialog that supports multi-file selection.

    This overrides the default QFileDialog to use a non-native dialog with multi-selection enabled
    for QListView and QTreeView widgets.
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
        QThread.__init__(self)
        self._stop_requested = False
        
        self.peaktype = peaktype
        self.fitdata = data  
        
        shape = (self.fitdata.shape[0], self.fitdata.shape[1])
        self.phase = np.full(shape, Area, dtype='float32')
        self.cen = np.full(shape, Pos, dtype='float32')
        self.wid = np.full(shape, FWHM, dtype='float32')
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
        self._stop_requested = True
    
    def run(self):
        
        """
        Initialise the single peak batch fitting process
        """  
        self.batchfit()
        
    def batchfit(self):
                
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
        Gaussian model with linear background: (A/(sqrt(2*pi)*w) )* exp( - (x-m)**2 / (2*w**2)) + a*x + b
        """
        return (A / (np.sqrt(2 * np.pi) * w)) * np.exp(- (x - m)**2 / (2 * w**2)) + a * x + b    
    
    def lmodel(self, x, A, m, w, a, b):
        
        """
        Lorentzian model with linear background: (A/(1 + ((1.0*x-m)/w)**2)) / (pi*w) + a*x + b   
        """
        return (A / (1 + ((x - m) / w)**2)) / (np.pi * w) + a * x + b
    
    def pvmodel(self, x, A, m, w, a, b, fr):
        
        """
        pseudo-Voigt model with linear background: ((1-fr)*gaumodel(x, A, m, s) + fr*lormodel(x, A, m, s))
        """
        gauss = (A / (np.sqrt(2 * np.pi) * w)) * np.exp(- (x - m)**2 / (2 * w**2))
        lorentz = (A / (1 + ((x - m) / w)**2)) / (np.pi * w)
        return (1 - fr) * gauss + fr * lorentz + a * x + b

def main():
    qApp = QtWidgets.QApplication(sys.argv)
    aw = nDTomoGUI()
    aw.show()
    sys.exit(qApp.exec_())
   
if __name__ == "__main__":
    main()
    