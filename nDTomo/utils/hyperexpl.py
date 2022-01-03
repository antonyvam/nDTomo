# -*- coding: utf-8 -*-
"""
HyperSliceExplorer

@author: Antony Vamvakeros
"""

import matplotlib.pyplot as plt
import numpy as np


class HyperSliceExplorer():
    
    
    '''
    HyperSliceExplorer is used to visualise hyperspectral imaging data
    '''
    
    def __init__(self, data, xaxis, xaxislabel):


        self.data = data
        self.xaxis = xaxis
        self.xaxislabel = xaxislabel
        
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
        self.meanSpectrum = np.mean(np.mean(data, axis = 0), axis = 0)
        self.cmap = 'jet'

        self.imoi = np.mean(self.data,axis=2)
        self.mdp = np.mean(np.mean(self.data,axis=1),axis=0)
        self.sdp = np.sum(np.sum(self.data,axis=1),axis=0)
        self.dproi = self.mdp

    def explore(self):
                
        self.mapper = plt;
        self.map_fig = self.mapper.figure()
        self.map_axes = self.map_fig.add_subplot(111)        
        self.map_data = self.mapper.imshow(self.imoi, cmap=self.cmap)
        title = 'Mean image'
        self.mapper.title(title, fontstyle='italic')
        self.map_fig.canvas.mpl_connect('button_press_event', self.onMapClick)
        self.map_fig.canvas.mpl_connect('motion_notify_event', self.onMapMoveEvent)        
        
        self.cb = self.map_fig.colorbar(self.map_data)
        
        self.mapper.show()
        self.mapper.draw()  
        
        
        self.plotter = plt
        self.plot_fig = self.plotter.figure()
        self.plot_axes = self.plot_fig.add_subplot(111)        
        self.plotter.legend()
        self.plotter.title("Mean diffraction pattern", fontstyle='italic')
        
        self.plot_fig.canvas.mpl_connect('motion_notify_event', self.onPlotClick)       
        self.plot_cid = self.plot_fig.canvas.mpl_connect('motion_notify_event', self.onPlotMoveEvent)   
        self.plotter.show()        
        self.plotter.draw()        

        self.histogramCurve = self.plotter.plot(self.xaxis, self.mdp, label='Mean diffraction pattern', color="b")
        self.activeCurve = self.plotter.plot(self.xaxis, self.mdp, label='Mean diffraction pattern', color="r")
        
        ########
        self.vCurve = self.plotter.axvline(x=0, color="k")
        #######
                
        
    def onMapClick(self, event): # SDMJ version
    
        if event.button == 1:
            if event.inaxes:
                self.col = int(event.xdata.round())
                self.row = int(event.ydata.round())
                
                self.dproi = self.data[self.row,self.col,:]
                
                self.histogramCurve[0].set_data(self.xaxis, self.dproi) 
                self.histogramCurve[0].set_label(str(self.row)+','+str(self.col))
                self.histogramCurve[0].set_visible(True)          
                self.plotter.legend()
                
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
                self.plot_axes.clear() # not fast
                self.plotter.plot(self.xaxis, self.mdp, label='Mean diffraction pattern')
#                self.selectedDataSetList.addItem('mean')
                self.plotter.legend()                
                self.plotter.draw() 

        if event.button == 3:
            if event.inaxes:
                self.histogramCurve[0].set_visible(False)          
                
                self.plot_axes.set_xlim(self.xaxis[0],self.xaxis[-1])
                self.plot_axes.set_ylim(0,np.max(self.dproi))
                self.plotter.show()        
                self.plotter.draw()

    def onMapMoveEvent(self, event): # SDMJ version
        if event.inaxes:
            
            col = int(event.xdata.round())
            row = int(event.ydata.round())
            
            dproi = self.data[row,col,:]            
            
            
            self.activeCurve[0].set_data(self.xaxis, dproi) 
            # self.mapper.relim()
            self.activeCurve[0].set_label(str(row)+','+str(col))
#            self.mapper.axes.autoscale(enable=True, axis='both', tight=True)#.axes.autoscale_view(False,True,True)
            self.activeCurve[0].set_visible(True)
            if np.max(dproi)>0:
                self.plot_axes.set_ylim(0,np.max(dproi))

        else:
            self.activeCurve[0].set_visible(False)
            self.activeCurve[0].set_label('')
            
#            self.plotter.axes.set_xlim(self.xaxis[0],self.xaxis[-1])
            self.plot_axes.set_ylim(0,np.max(self.dproi))
            
        self.plotter.legend()
        self.plotter.draw()
    
            
    
    def onPlotMoveEvent(self, event):
        
        if event.inaxes:
                        
            nx = np.argmin(np.abs(self.xaxis-event.xdata))
            
            if nx<0:
                nx = 0
            elif nx>len(self.xaxis):
                nx = len(self.xaxis)-1

            self.selectedChannels = nx
            self.map_axes.clear() # not fast
            self.imoi = self.data[:,:,nx]
            self.map_data = self.map_axes.imshow(self.imoi, cmap = 'jet')
            title = "Channel = %d; %s = %.3f" % (nx, self.xaxislabel, self.xaxis[nx])


            try:
                self.cb.remove()
            except:
                pass
            
            self.cb = self.map_fig.colorbar(self.map_data)

            
            self.map_axes.set_title(title)
            self.mapper.draw_all()             

            
            self.vCurve.set_xdata(event.xdata) 
                
            self.plotter.draw()
     
    def onPlotClick(self, event):

        if event.button == 1:
            self.plot_fig.canvas.mpl_disconnect(self.plot_cid)   

        elif event.button == 3:
            self.plot_cid = self.plot_fig.canvas.mpl_connect('motion_notify_event', self.onPlotMoveEvent)
            self.plotter.show()  
            self.plotter.draw()   
            
    def update(self):
        
        self.map_axes.clear() # not fast
        # this bit is messy
        
        self.imoi = np.mean(self.data, axis = 2)
        if (not self.selectedChannels):
            self.map_axes.imshow(self.imoi ,cmap=self.cmap)
            title = 'Mean image'
        else:
            if self.selectedChannels.size == 1:
                self.map_axes.imshow(self.data[:,:,self.selectedChannels],cmap=self.cmap)
                title = "Channel = %d; %s = %.3f" % (self.nx, self.xaxislabel, self.xaxis[self.nx])
            if self.selectedChannels.size > 1:
                self.map_axes.imshow(np.mean(self.data[:,:,self.selectedChannels],2),cmap=self.cmap)
                title = self.name+' '+'mean of selected channels'
        self.map_axes.set_title(title)
        self.mapper.show()
        self.mapper.draw_all() 










