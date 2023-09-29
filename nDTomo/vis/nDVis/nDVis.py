# -*- coding: utf-8 -*-
"""

nDTomoGUI for chemical imaging data visualization

@author: A. Vamvakeros

"""

#%

from __future__ import unicode_literals
from matplotlib import use as u
u('Qt5Agg')
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal, QThread
import h5py, sys
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.optimize as sciopt

class nDTomoGUI(QtWidgets.QMainWindow):
    
    """
    nDTomoGUI
    """
    
    def __init__(self):

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
        
        self.peaktype = 'Gaussian'
        self.Area = 0.5; self.Areamin = 0.; self.Areamax = 10.
        self.FWHM = 1.; self.FWHMmin = 0.1; self.FWHMmax = 5.
        
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("nDTomoGUI")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Open Chemical imaging data', self.fileOpen, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.file_menu.addAction('&Save Chemical imaging data', self.savechemvol)
        self.file_menu.addAction('&Quit', self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

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
        self.ax_spectrum.set_title("Spectrum")

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
        
        self.Channel2 = QtWidgets.QLabel(self)
        self.Channel2.setText('Final channel')
        self.tab2.layout.addWidget(self.Channel2,1,2)        
                    
        self.crspinbox2 = QtWidgets.QSpinBox(self)
        self.crspinbox2.valueChanged.connect(self.channel_final)
        self.crspinbox2.setMinimum(1)
        self.tab2.layout.addWidget(self.crspinbox2,1,3)           
        
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


        ############ Tab4 - Peak fitting ############

        self.Labelbkg = QtWidgets.QLabel(self)
        self.Labelbkg.setText('Single peak fitting')
        self.tab4.layout.addWidget(self.Labelbkg,0,0)
        
        self.LabelTypePeak = QtWidgets.QLabel(self)
        self.LabelTypePeak.setText('Function')
        self.tab4.layout.addWidget(self.LabelTypePeak,1,0)
        
        self.ChooseFunction = QtWidgets.QComboBox(self)
        self.ChooseFunction.addItems(["Gaussian", "Lorentzian", "Pseudo-Voigt"])
        self.ChooseFunction.currentIndexChanged.connect(self.profile_function)
        self.ChooseFunction.setEnabled(True)
        self.tab4.layout.addWidget(self.ChooseFunction,1,1)   

        self.pbutton_fit = QtWidgets.QPushButton("Perform batch peak fitting",self)
        self.pbutton_fit.clicked.connect(self.batchpeakfit)
        self.tab4.layout.addWidget(self.pbutton_fit,1,2)
        
        self.progressbar_fit = QtWidgets.QProgressBar(self)
        self.tab4.layout.addWidget(self.progressbar_fit,1,3)

        self.pbutton_stop = QtWidgets.QPushButton("Stop",self)
        self.pbutton_stop.clicked.connect(self.stopfit)
        self.tab4.layout.addWidget(self.pbutton_stop,1,4)
        
        self.LabelRes = QtWidgets.QLabel(self)
        self.LabelRes.setText('Display peak fitting results')
        self.tab4.layout.addWidget(self.LabelRes,2,0)
        
        self.ChooseRes = QtWidgets.QComboBox(self)
        self.ChooseRes.addItems(['Area','Position', 'FWHM', 'Slope', 'Intercept'])
        self.ChooseRes.currentIndexChanged.connect(self.plot_fit_results)
        self.ChooseRes.setEnabled(False)
        self.tab4.layout.addWidget(self.ChooseRes,2,1)   
        
        self.pbutton_expfit = QtWidgets.QPushButton("Export fit results",self)
        self.pbutton_expfit.clicked.connect(self.savefitresults)
        self.pbutton_expfit.setEnabled(False)
        self.tab4.layout.addWidget(self.pbutton_expfit,2,2)

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
        # Calculate mean image and mean spectrum
        mean_image = np.mean(self.volume, axis=2)
        mean_spectrum = np.mean(self.volume, axis=(0, 1))

        # Update the image display and spectrum plot
        self.ax_image.imshow(mean_image.T, cmap=self.cmap)
        self.ax_spectrum.plot(self.xaxis, mean_spectrum, color='b')
        self.ax_spectrum.set_xlabel(self.xaxislabel)

        # Connect mouse hover events
        self.canvas_image.mpl_connect('motion_notify_event', self.update_spectrum)
        self.canvas_spectrum.mpl_connect('motion_notify_event', self.update_image)
        self.canvas_image.mpl_connect('button_press_event', self.toggle_real_time_spectrum)
        self.canvas_spectrum.mpl_connect('button_press_event', self.toggle_real_time_image)
        # Connect mouse scroll events for zooming
        self.canvas_image.mpl_connect('scroll_event', self.on_canvas_scroll)
        self.canvas_spectrum.mpl_connect('scroll_event', self.on_canvas_scroll)

        self.real_time_update_image = True  # Flag to enable real-time update for image
        self.real_time_update_spectrum = True  # Flag to enable real-time update for spectrum

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

    def update_spectrum(self, event):
        """
        Update the spectrum plot based on the mouse hover event on the image plot.
        """
        if self.real_time_update_spectrum and event.inaxes == self.ax_image:
            self.x, self.y = int(event.xdata), int(event.ydata)

            # Check if the mouse position is within the image dimensions
            if self.x >= 0 and self.x < self.image_width and self.y >= 0 and self.y < self.image_height:
                # Get the spectrum from the volume
                self.spectrum = self.volume[self.x, self.y, :]

                # Update the spectrum plot
                self.ax_spectrum.clear()
                self.ax_spectrum.plot(self.xaxis, self.spectrum, color='b')
                self.ax_spectrum.set_title(f"Spectrum: ({self.x}, {self.y})")
                self.ax_spectrum.set_xlabel(self.xaxislabel)
                self.canvas_spectrum.draw()

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
                        self.index = np.searchsorted(self.xaxis.flatten(), [self.index])[0] -1
                    
                    if self.index<0:
                        self.index = 0
                    elif self.index>len(self.xaxis):
                        self.index = len(self.xaxis)-1                
    
                    # Get the image from the volume
                    self.image = self.volume[:, :, self.index]
        
                    # Update the image display
                    self.ax_image.clear()
                    self.ax_image.imshow(self.image.T, cmap=self.cmap)
                    
                    if self.xaxislabel == 'Channel':
                        self.ax_image.set_title(f"Image: Channel {self.index}")
                    else:
                        self.ax_image.set_title("Image: Channel %d, %s %.3f" %(self.index, self.xaxislabel, self.xaxis[self.index]))
    
                    self.canvas_image.draw()
                    
                except:
                    pass

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
        
        self.hdf_fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Chemical imaging data', "", "*.hdf5 *.h5")
        
        if len(self.hdf_fileName)>0:
            
            self.loadchemvol()
       
            datasetname = self.hdf_fileName.split("/")
            self.datasetname = datasetname[-1]
            self.DatasetNameLabel.setText(self.datasetname)
            self.explore()
                        
    def loadchemvol(self):
        
        xaxis_labels = ['/d', '/q', '/twotheta', '/Energy', '/tth', '/energy']
        
        if len(self.hdf_fileName)>0:

            with h5py.File(self.hdf_fileName,'r') as f:
                try:
                    print(f.keys())
                    self.volume = f['/data'][:]
                    self.check_and_transpose()
                    self.image_width, self.image_height, self.nbins = self.volume.shape
                    self.xaxis = np.arange(0, self.volume.shape[2])
                    self.crspinbox1.setMaximum(self.nbins - 1)
                    self.crspinbox2.setMaximum(self.nbins)
                except:
                    print('Seems like there is a problem accessing the data')
                for xaxis_label in xaxis_labels:
                    if xaxis_label in f:
                        self.xaxis = f[xaxis_label][:]
                        self.xaxislabel = xaxis_label.lstrip('/')                         
        print(self.volume.shape)
                        
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

        roi = np.arange(self.chi,self.chf)
        self.xroi = np.arange(0, len(roi))
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
        self.xroi = np.arange(0, len(roi))
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
        self.xroi = np.arange(0, len(roi))
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
        for ii in tqdm(range(self.volume.shape[2])):
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

    ####################### Peak fitting #######################

    def profile_function(self,ind):
        if ind == 0:
            self.peaktype = "Gaussian"
            print("Gaussian profile")
        elif ind == 1:
            self.peaktype = "Lorentzian"
            print("Lorentzian profile")
        elif ind == 2:
            self.peaktype = "Pseudo-Voigt"
            print("Pseudo-Voigt profile")

    def batchpeakfit(self):

        self.pbutton_fit.setEnabled(False)
        self.Pos = (int(self.volroi.shape[2]/2))
        self.Posmin = (int(self.volroi.shape[2]/2) - 5)
        self.Posmax = (int(self.volroi.shape[2]/2) + 5)
        
        for ii in range(self.volroi.shape[2]):
            self.volroi[:,:,ii] = self.volroi[:,:,ii]*self.mask
            
        self.PeakFitData = FitData(self.peaktype, self.volroi, self.xroi,
                                   self.Area, self.Areamin, self.Areamax,
                                   self.Pos, self.Posmin, self.Posmax,
                                   self.FWHM, self.FWHMmin, self.FWHMmax)
        self.PeakFitData.start()            
        self.PeakFitData.progress_fit.connect(self.progressbar_fit.setValue)
        self.PeakFitData.fitdone.connect(self.updatefitdata)
        
    def updatefitdata(self):
        
        self.res = self.PeakFitData.res #### need to think about this
        self.pbutton_fit.setEnabled(True)
        self.ChooseRes.setEnabled(True)
        self.pbutton_expfit.setEnabled(True)
        
        self.ax_image.clear()
        self.ax_image.imshow(self.res['Area'].T,cmap='jet')
        self.ax_image.set_title("Peak area")

    def stopfit(self):
        self.PeakFitData.terminate()
        self.progressbar_fit.setValue(0)
        self.pbutton_fit.setEnabled(True)
        
    def plot_fit_results(self,ind):
        
        self.ax_image.clear()
        if ind == 0:
            self.ax_image.imshow(self.res['Area'].T,cmap='jet')
            self.ax_image.set_title("Peak area")
        elif ind == 1:
            self.ax_image.imshow(self.res['Position'].T,cmap='jet')
            self.ax_image.set_title("Peak position")        
        elif ind == 2:
            self.ax_image.imshow(self.res['FWHM'].T,cmap='jet')
            self.ax_image.set_title("FWHM")  
        elif ind == 3:
            self.ax_image.imshow(self.res['Background1'].T,cmap='jet')
            self.ax_image.set_title("Background 1")  
        elif ind == 4:
            self.ax_image.imshow(self.res['Background2'].T,cmap='jet')
            self.ax_image.set_title("Background 2")  
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
                    

class FileDialog(QtWidgets.QFileDialog):
        def __init__(self, *args):
            QtWidgets.QFileDialog.__init__(self, *args)
            self.setOption(self.DontUseNativeDialog, True)
#            self.setFileMode(self.DirectoryOnly)
            for view in self.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
                if isinstance(view.model(), QtWidgets.QFileSystemModel):
                    view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)


class FitData(QThread):
    
    '''
    Single peak batch fitting class   
    
    :data: the spectral/scattering data
    :roi: bin number of interest
    :Area: initial value for peak area
    :Areamin: minimum value for peak area
    :Areamax: maximum value for peak area
    :Pos: initial value for peak position
    :Posmin: minimum value for peak position
    :Posmax: maximum value for peak position
    :FWHM: initial value for peak full width at half maximum (FWHM)
    :FWHMmin: minimum value for peak FWHM
    :FWHMmax: maximum value for peak FWHM

    '''
    
    fitdone = pyqtSignal()
    progress_fit = pyqtSignal(int)
    
    def __init__(self, peaktype, data, x, Area, Areamin, Areamax, Pos, Posmin, Posmax, FWHM, FWHMmin, FWHMmax):
        QThread.__init__(self)
        
        self.peaktype = peaktype
        self.fitdata = data  
        self.phase = np.zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.cen = np.zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.wid = np.zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.bkg1 = np.zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.bkg2 = np.zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.fr = np.zeros((self.fitdata.shape[0],self.fitdata.shape[1]))
        self.xroi = x
        self.Area = Area; self.Areamin = Areamin; self.Areamax = Areamax; 
        self.Pos = Pos; self.Posmin = Posmin; self.Posmax = Posmax; 
        self.FWHM = FWHM; self.FWHMmin = FWHMmin; self.FWHMmax = FWHMmax;
        
        msk = np.sum(self.fitdata,axis = 2)
        msk[msk>0] = 1
        self.i, self.j = np.where(msk > 0)

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
        
        for ind in tqdm(np.arange(0,len(self.i))):
                
                ii = self.i[ind]
                jj = self.j[ind]
                dp = self.fitdata[ii,jj,:]
                try:
                    if self.peaktype == "Gaussian":
                        params, params_covariance = sciopt.curve_fit(self.gmodel, self.xroi, dp[self.xroi], p0=x0, bounds=param_bounds)
                    elif self.peaktype == "Lorentzian":
                        params, params_covariance = sciopt.curve_fit(self.lmodel, self.xroi, dp[self.xroi], p0=x0, bounds=param_bounds)
                    elif self.peaktype == "Pseudo-Voigt":
                        params, params_covariance = sciopt.curve_fit(self.pvmodel, self.xroi, dp[self.xroi], p0=x0, bounds=param_bounds)
                    
                    self.phase[ii,jj] = params[0]
                    self.cen[ii,jj] = params[1]
                    self.wid[ii,jj] = params[2]         
                    self.bkg1[ii,jj] = params[3]   
                    self.bkg2[ii,jj] = params[4]      
                    if self.peaktype == "Pseudo-Voigt":
                        self.fr[ii,jj] = params[5]  
           
                except: 
                    self.cen[ii,jj] = (param_bounds[0][1] + param_bounds[1][1])/2
                    self.wid[ii,jj] = (param_bounds[0][2] + param_bounds[1][2])/2     
                        
                v = (100.*(ind+1))/(len(self.i))
                self.progress_fit.emit(v)
                
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
        return (A/(np.sqrt(2*np.pi)*w) )* np.exp( - (x-m)**2 / (2*w**2)) + a*x + b
    
    def lmodel(self, x, A, m, w, a, b):
        
        """
        Lorentzian model with linear background: (A/(1 + ((1.0*x-m)/w)**2)) / (pi*w) + a*x + b   
        """
        return (A/(1 + ((x-m)/w)**2)) / (np.pi*w) + a*x + b    

    def pvmodel(self, x, A, m, w, a, b, fr):
        
        """
        pseudo-Voigt model with linear background: ((1-fr)*gaumodel(x, A, m, s) + fr*lormodel(x, A, m, s))
        """
        return ((1-fr)*(A/(np.sqrt(2*np.pi)*w) )*np.exp( - (x-m)**2 / (2*w**2)) + fr*(A/(1 + ((x-m)/w)**2)) / (np.pi*w) + a*x + b)


def main():
    qApp = QtWidgets.QApplication(sys.argv)
    aw = nDTomoGUI()
    aw.show()
    sys.exit(qApp.exec_())
    qApp.exec_()
   
if __name__ == "__main__":
    main()
    
#aw = nDTomoGUI()    
#aw.show()
   
