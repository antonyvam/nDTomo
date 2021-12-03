# -*- coding: utf-8 -*-
"""
Some test classes for visualization

@authors: S.D.M. Jacques and A. Vamvakeros
"""

from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
        
class Coordinate(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class HyperDataBaseStruct():
    def __init__(self):
        self.name = ''
        self.data = []
        self._order = 0;
        self.nbin = 0;
        self.x = []
        self.meanSpectra = []
        self.sumSpectra = []
        self.spectraXLabel = 'undefined x'
        self.spectraYLabel = 'undefined y'

class HyperLineDataStruct(HyperDataBaseStruct):
    def __init__(self):
        HyperDataBaseStruct.__init__(self)
        self._order = 1;
        self.n = 0;
        self.nLabel = 'undefined'

class HyperSliceDataStruct(HyperDataBaseStruct):
    def __init__(self):
        HyperDataBaseStruct.__init__(self)
        self._order = 2;
        self.nrow = 0;
        self.ncol = 0;
        self.rowLabel = 'undefined'
        self.colLabel = 'undefined'

class SpectrumStats(HyperDataBaseStruct):
    def __init__(self):
        HyperDataBaseStruct.__init__(self)
    # Methods for calculating properties        
    def computeMeanSpectra(self):
        self.meanSpectra = self.data.mean(axis=tuple(range(0, self._order)))
        return self.meanSpectra
    def computeSumSpectra(self):
        self.sumSpectra = self.data.sum(axis=tuple(range(0, self._order)))


class HyperSlice(SpectrumStats, HyperSliceDataStruct):
    def __init__(self):
        HyperSliceDataStruct.__init__(self)
    # Methods for transforming data    

class HyperLine(SpectrumStats, HyperLineDataStruct):
    def __init__(self):
        HyperLineDataStruct.__init__(self)
       
class HyperSliceExplorer():
    def __init__(self):
        self.mapper = []
        self.map_fig = []
        self.map_axes = []
        self.map_data = []
        self.mapHoldState= 0;
        self.plotter = []
        self.plot_fig = []
        self.plot_axes = []
        self.currentCurve = 0;
        self.plot = []
        self.plotHoldState= 0;
        self.selectedChannels = []
        self.selectedVoxels = np.empty(0,dtype=object)

    def explore(self):
        self.mapper = plt;
        self.map_fig = self.mapper.figure()
        self.map_axes = self.map_fig.add_subplot(111)
        self.map_data = self.mapper.imshow(np.mean(self.data,2),cmap='jet')
        title = self.name+' '+'mean image'
        self.mapper.title(title, fontstyle='italic')
        self.map_fig.canvas.mpl_connect('button_press_event',self.onMapClick)
        self.map_fig.canvas.mpl_connect('motion_notify_event',self.onMapMoveEvent)

        self.plotter = plt
        self.plot_fig = self.plotter.figure()
        self.plot_axes = self.plot_fig.add_subplot(111)
        self.plot_axes.plot(self.x,self.meanSpectra, label='mean spectra')
#        self.plotter.get_current_fig_manager().toolbar.zoom()
        self.plotter.legend()
#        plot_cid = self.plot_fig.canvas.mpl_connect('button_press_event',self.onPlotClick)
        self.plot_cid = self.plot_fig.canvas.mpl_connect('motion_notify_event',self.onPlotMoveEvent)

#        self.mapper.ion() #added by av

    def onMapMoveEvent(self, event):
        if event.inaxes:
            x = int(event.xdata.round())
            y = int(event.ydata.round())
            
            if self.selectedVoxels.size == 0:
                self.selectedVoxels = np.append(self.selectedVoxels,Coordinate(x,y))
            else:
                self.selectedVoxels[-1] = Coordinate(x,y)
            self.plot_axes.lines[self.currentCurve].remove()
            self.plot_axes.plot(self.x,self.data[y,x,:], 'C0', label=str(y)+','+str(x))

            self.plotter.legend()
            self.plotter.draw_all()
            #self.plotter.show(block=False)#self.plotter.ion()#self.plotter.pause(0.0001)#self.plotter.ion()
        else:
            # need to not plot the active line here
            return

    def onPlotMoveEvent(self, event):
        if event.inaxes:
            nx = np.argmin(np.abs(self.x-event.xdata))
            self.selectedChannels = nx;
            self.map_axes.clear() # not fast
            self.map_axes.imshow(self.data[:,:,nx],cmap='jet')
            title = "%s: ch = %d; x = %.3f" % (self.name, nx, self.x[nx])
            self.map_axes.set_title(title)
            self.mapper.draw_all()
            #self.mapper.show(block=False)#self.mapper.ion()#self.mapper.pause(0.0001)

    def onMapClick(self, event):
        if event.inaxes:
            x = int(event.xdata.round())
            y = int(event.ydata.round())
            self.selectedVoxels = np.append(self.selectedVoxels,Coordinate(x,y))
            self.plot_axes.plot(self.x,self.data[y,x,:], 'C3', label=str(y)+','+str(x))
            self.currentCurve += 1
            self.plotter.legend()
            self.plotter.draw_all()
            #self.plotter.show(block=False)#self.plotter.ion()#self.plotter.pause(0.0001)
        else:
            self.selectedVoxels = []
            self.plot_axes.clear() # not fast
     
    def onPlotClick(self, event):
        print('Plot')
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
      
    def update(self):
        self.map_axes.clear() # not fast
        # this bit is messy
        if (not self.selectedChannels):
            self.map_axes.imshow(np.mean(self.data,2),cmap='jet')
            title = self.name+' '+'mean image'
        else:
            if self.selectedChannels.size == 1:
                self.map_axes.imshow(self.data[:,:,self.selectedChannels],cmap='jet')
                title = "%s: ch = %d; x = %.3f" % (self.name, self.selectedChannels, self.x[self.selectedChannels])
            if self.selectedChannels.size > 1:
                self.map_axes.imshow(np.mean(self.data[:,:,self.selectedChannels],2),cmap='jet')
                title = self.name+' '+'mean of selected channels'
        self.map_axes.set_title(title)
        self.mapper.draw_all()
	#self.mapper.show(block=False)#self.mapper.ion()#self.mapper.pause(0.0001)
        return