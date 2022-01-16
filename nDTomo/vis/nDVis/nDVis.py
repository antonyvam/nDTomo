# -*- coding: utf-8 -*-
"""

nDVis GUI for chemical imaging data visualization

@author: A. Vamvakeros

"""

#%%

from __future__ import unicode_literals

from matplotlib import use as u
u('Qt5Agg')

from PyQt5 import QtCore, QtWidgets, QtGui

import sys, os, h5py
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import matplotlib.pyplot as plt



class ApplicationWindow(QtWidgets.QMainWindow):
    
    """
    
    nDVis GUI
    
    """
    
    def __init__(self):

        super(ApplicationWindow, self).__init__()
        
        self.data = np.zeros(())
        self.xaxis = np.zeros(())
        self.imoi = np.zeros(())
        self.hdf_fileName = ''
        self.dproi = 0
        self.c = 3E8
        self.h = 6.620700406E-34        
        self.cmap = 'jet'
        self.xaxislabel = 'Channel'
        
        self.cmap_list = ['viridis','plasma','inferno','magma','cividis','flag', 
            'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
        
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("MultiTool")

#        self.left = 100
#        self.top = 100
#        self.width = 640
#        self.height = 480
#        self.setGeometry(self.left, self.top, self.width, self.height)

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

        self.tab1.layout = QtWidgets.QGridLayout()       

        # set up the mapper
        self.mapperWidget = QtWidgets.QWidget(self)
        self.mapper = MyCanvas()
        self.mapperExplorerDock = QtWidgets.QDockWidget("Image", self)
        self.mapperExplorerDock.setWidget(self.mapperWidget)
        self.mapperExplorerDock.setFloating(False)
        self.mapperToolbar = NavigationToolbar(self.mapper, self)
        
        vbox1 = QtWidgets.QVBoxLayout()
#        vbox1 = QtWidgets.QGridLayout()
        
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
#        vbox2 = QtWidgets.QGridLayout()
        
        vbox2.addWidget(self.plotterToolbar)        
        vbox2.addWidget(self.plotter) # starting row, starting column, row span, column span
        self.plotterWidget.setLayout(vbox2)
        
        self.setCentralWidget(self.tabs)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.mapperExplorerDock)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.plotterExplorerDock)

        # set up the Chemical imaging image mapper
        self.mapperWidget_xrdct = QtWidgets.QWidget(self)
        self.mapper_xrdct = MyCanvas()
        self.mapperExplorerDock_xrdct = QtWidgets.QDockWidget("Image", self)
        self.mapperExplorerDock_xrdct.setWidget(self.mapperWidget_xrdct)
        self.mapperExplorerDock_xrdct.setFloating(False)
        self.mapperToolbar_xrdct = NavigationToolbar(self.mapper_xrdct, self)
        vbox3 = QtWidgets.QVBoxLayout()        
        vbox3.addWidget(self.mapperToolbar_xrdct)
        vbox3.addWidget(self.mapper_xrdct)
        self.mapperWidget_xrdct.setLayout(vbox3)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.mapperExplorerDock_xrdct)
        self.mapperExplorerDock_xrdct.setVisible(False)
    
        
        ############### Chemical imaging panel ############### 
        
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
#        self.ExportDPbutton.setMaximumWidth(150)
        self.tab1.layout.addWidget(self.ExportDPbutton,1,2)

        self.ExportIMbutton = QtWidgets.QPushButton("Export image of interest",self)
        self.ExportIMbutton.clicked.connect(self.exportim)
#        self.ExportDPbutton.setMaximumWidth(150)
        self.tab1.layout.addWidget(self.ExportIMbutton,1,3)
        
                
#        self.main_widget.setFocus()
#        self.setCentralWidget(self.main_widget)

        self.tabs.setFocus()
        self.setCentralWidget(self.tabs)
                
        self.tab1.setLayout(self.tab1.layout)     
        
#        self.setLayout(layout)
        self.show()
        
        self.selectedVoxels = np.empty(0,dtype=object)
        self.selectedChannels = np.empty(0,dtype=object)
    
    
    ####################### Chemical imaging #######################

    def exportdp(self):
        
        """
        
        Method to export spectra/diffraction patterns of interest
        
        """
        
        if len(self.hdf_fileName)>0 and len(self.dproi)>0:
            
            s = self.hdf_fileName.split('.hdf5'); s = s[0]
            sn = "%s_%s_%s.h5" %(s,str(self.row),str(self.col))
            print(sn)

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.dproi)
            h5f.create_dataset('xaxis', data=self.xaxis)
            h5f.close()
        
            perm = 'chmod 777 %s' %sn
            os.system(perm)    
            
            xy = np.column_stack((self.xaxis,self.dproi))
            sn = "%s_%s_%s.asc" %(s,str(self.row),str(self.col))
            np.savetxt(sn,xy)
            perm = 'chmod 777 %s' %sn
            os.system(perm) 
            
            xy = np.column_stack((self.xaxis,self.dproi))
            sn = "%s_%s_%s.xy" %(s,str(self.row),str(self.col))
            np.savetxt(sn,xy)
            perm = 'chmod 777 %s' %sn
            os.system(perm) 

                
        else:
            print("Something is wrong with the data")
        
    def exportim(self):
        
        """
        
        Method to export spectral/scattering image of interest
        
        """
        
        if len(self.hdf_fileName)>0 and len(self.imoi)>0:
            s = self.hdf_fileName.split('.hdf5'); s = s[0]
            sn = "%s_channel_%s.h5" %(s,str(self.nx))
            print(sn)

            h5f = h5py.File(sn, "w")
            h5f.create_dataset('I', data=self.imoi)
            h5f.create_dataset('Channel', data=self.nx)            
            h5f.close()
        
            perm = 'chmod 777 %s' %sn
            os.system(perm)    
            
            sn = "%s_channel_%s.png" %(s,str(self.nx))
            plt.imsave(sn,self.imoi,cmap=self.cmap)
                        
        else:
            print("Something is wrong with the data")
            
                
    def changecolormap(self,ind):
        
        self.cmap = self.cmap_list[ind]
        print(self.cmap)
        try:
            self.update()
        except: 
            pass
            
    def explore(self):
                
        self.imoi = np.mean(self.data,axis=2)
        self.map_data = self.mapper.axes.imshow(self.imoi, cmap=self.cmap)
        title = 'Mean image'
        self.mapper.axes.set_title(title, fontstyle='italic')
        self.mapper.fig.canvas.mpl_connect('button_press_event', self.onMapClick)
        self.mapper.fig.canvas.mpl_connect('motion_notify_event', self.onMapMoveEvent)        
        
        
        # self.cb = self.mapper.fig.colorbar(self.map_data)
        
        self.mapper.show()
        self.mapper.draw()  
        
        self.mdp = np.mean(np.mean(self.data,axis=1),axis=0)
        self.sdp = np.sum(np.sum(self.data,axis=1),axis=0)
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
                
        
    def onMapClick(self, event): # SDMJ version
        if event.button == 1:
            if event.inaxes:
                self.col = int(event.xdata.round())
                self.row = int(event.ydata.round())
                
                
                
                if self.xaxislabel == 'd':
                    self.dproi = self.data[self.row,self.col,::-1]
                else:
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
                self.selectedVoxels = np.empty(0,dtype=object)
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
                self.plotter.axes.set_ylim(0,np.max(self.dproi))
                self.plotter.show()        
                self.plotter.draw()

    def onMapMoveEvent(self, event): # SDMJ version
        if event.inaxes:
            
            col = int(event.xdata.round())
            row = int(event.ydata.round())
            
            if self.xaxislabel == 'd':
                dproi = self.data[row,col,::-1]
            else:
                dproi = self.data[row,col,:]            
            
            
            self.activeCurve[0].set_data(self.xaxis, dproi) 
            self.mapper.axes.relim()
            self.activeCurve[0].set_label(str(row)+','+str(col))
#            self.mapper.axes.autoscale(enable=True, axis='both', tight=True)#.axes.autoscale_view(False,True,True)
            self.activeCurve[0].set_visible(True)
            if np.max(dproi)>0:
                self.plotter.axes.set_ylim(0,np.max(dproi))

        else:
            self.activeCurve[0].set_visible(False)
            self.activeCurve[0].set_label('')
            
#            self.plotter.axes.set_xlim(self.xaxis[0],self.xaxis[-1])
            self.plotter.axes.set_ylim(0,np.max(self.dproi))
            
        self.plotter.axes.legend()
        self.plotter.draw()
    
    
    def onPlotMoveEvent(self, event):
        if event.inaxes:
            
            x = event.xdata
            
            if self.xaxislabel == 'd':
                self.nx = len(self.xaxis) - np.searchsorted(self.xaxisd, [x])[0]
            else:
                self.nx = np.searchsorted(self.xaxis, [x])[0] -1
            
            if self.nx<0:
                self.nx = 0
            elif self.nx>len(self.xaxis):
                self.nx = len(self.xaxis)-1

            
            self.selectedChannels = self.nx;
            self.mapper.axes.clear() # not fast
            self.imoi = self.data[:,:,self.nx]
#            self.map_data = self.mapper.axes.imshow(self.imoi,cmap='jet')
            self.map_data = self.mapper.axes.imshow(self.imoi,cmap = self.cmap)
            title = "Channel = %d; %s = %.3f" % (self.nx, self.xaxislabel, self.xaxis[self.nx])

            try:
                self.cb.remove()
            except:
                pass
            
            # self.cb = self.mapper.fig.colorbar(self.map_data)

            self.mapper.axes.set_title(title)
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
        self.mapper.axes.clear() # not fast
        # this bit is messy
        
        self.imoi = np.mean(self.data, axis = 2)
        if (not self.selectedChannels):
            self.mapper.axes.imshow(self.imoi ,cmap=self.cmap)
            title = 'Mean image'
        else:
            if self.selectedChannels.size == 1:
                self.mapper.axes.imshow(self.data[:,:,self.selectedChannels],cmap=self.cmap)
                title = "Channel = %d; %s = %.3f" % (self.nx, self.xaxislabel, self.xaxis[self.nx])
            if self.selectedChannels.size > 1:
                self.mapper.axes.imshow(np.mean(self.data[:,:,self.selectedChannels],2),cmap=self.cmap)
                title = self.name+' '+'mean of selected channels'
        self.mapper.axes.set_title(title)
        self.mapper.show()
        self.mapper.draw()  
            

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
    
    
            try:
                self.histogramCurve.pop(0).remove()
                self.activeCurve.pop(0).remove()
                self.vCurve.remove()
#                self.cb.remove()
            except:
                pass
    
            self.explore()
        
    def loadchemvol(self):
        
        if len(self.hdf_fileName)>0:
            
            with h5py.File(self.hdf_fileName,'r') as f:
                
                try:
                    self.data = f['/data'][:]
                    self.data = np.transpose(self.data, (2,1,0))
                    self.xaxis = np.arange(0, self.data.shape[2])
                    
                except:
                    try:
                        self.q = f['/q'][:]
                        self.xaxis = self.q
                        self.xaxislabel = 'Q'
                        
                    except:
                        pass

        
        print(self.data.shape)
                        
	       
        
    def savechemvol(self):

        self.fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Chemical imaging data', "", "*.hdf5")
	
        if len(self.fn)>0:
    
            st = self.fn.split(".hdf5")
            if len(st)<2:
                self.fn = "%s.hdf5" %self.fn
                print(self.fn)


            h5f = h5py.File(self.fn, "w")
            h5f.create_dataset('data', data=self.data)
            h5f.create_dataset('xaxis', data=self.xaxis)
            
            h5f.close()
        
            perm = 'chmod 777 %s' %self.fn
            os.system(perm)    
        

    def fileQuit(self):
#        plt.close('all')
        self.close()

    def closeEvent(self, ce):
#        plt.close('all')
        self.fileQuit()

    def about(self):
        message = '<b>EasyVis<p>'
        message += '<p><i>Created by Antony Vamvakeros. Running under license under GPLv3'
        message += '\t '
        d = QtWidgets.QMessageBox()
        d.setWindowTitle('About')
        d.setText(message)
        d.exec_()


class Coordinate(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class MyCanvas(Canvas):
    #Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.).
    def __init__(self, parent=None, width=5, height=4, dpi=100):
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

class FileDialog(QtWidgets.QFileDialog):
        def __init__(self, *args):
            QtWidgets.QFileDialog.__init__(self, *args)
            self.setOption(self.DontUseNativeDialog, True)
#            self.setFileMode(self.DirectoryOnly)
            for view in self.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
                if isinstance(view.model(), QtWidgets.QFileSystemModel):
                    view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
                    
def main():
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
    qApp.exec_()
    
if __name__ == "__main__":
    main()
    
# aw = ApplicationWindow()    
# aw.show()
   
