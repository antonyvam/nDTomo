# -*- coding: utf-8 -*-
"""
Created on Sun May 20 15:25:19 2018

@author: Antony
"""

from __future__ import unicode_literals
import os, time, h5py, json
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt 

try:
	import fabio, pyFAI
except:
	"Problem importing pyFAI and/or fabio"

from PyQt5 import QtCore

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

from threading import Timer, Lock

class Periodic(object):
    # code by user fdb on stack overflow
    def __init__(self, interval, function, *args, **kwargs):
        self._lock = Lock()
        self._timer = None
        self.function = function
        self.interval = interval
        self.args = args
        self.kwargs = kwargs
        self._stopped = True
        if kwargs.pop('autostart', True):
            self.start()

    def start(self, from_run=False):
        self._lock.acquire()
        if from_run or self._stopped:
            self._stopped = False
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self._lock.release()

    def _run(self):
        self.start(from_run=True)
        self.function(*self.args, **self.kwargs)

    def stop(self):
        self._lock.acquire()
        self._stopped = True
        self._timer.cancel()
        self._lock.release()

        
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

class SliceTransformations():
    def __init__(self):
        pass
    def rowFlip(self):
        self.data[1::2,:,:] = self.data[1::2,::-1,:]
        
    def colFlip(self):
        self.data[:,1::2,:] = self.data[::-1,1::2,:]

    def transpose2D(self):
	self.data = np.transpose(self.data,(1,0))

    def transpose3D(self):
	self.data = np.transpose(self.data,(1,0,2))

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
    
class XRDCT_Squeeze(QtCore.QThread):
    squeeze = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int)
    
    def __init__(self,prefix,dataset,xrdctpath,maskname,poniname,na,nt,npt_rad,filt,procunit,units,prc,thres,datatype,savepath,scantype,energy,omega,trans,dio,etime):
        QtCore.QThread.__init__(self)
        self.prefix = prefix; self.dataset=dataset; self.xrdctpath = xrdctpath; self.maskname = maskname; self.poniname = poniname
        self.filt=filt; self.procunit=procunit;self.units =units;self.E = energy
        self.prc=prc;self.thres = thres;self.datatype = datatype;self.savepath = savepath
        self.na = int(na); self.nt = int(nt); self.npt_rad = int(npt_rad); self.scantype = scantype; #self.jsonname = jsonname
        self.omega = omega; self.trans = trans
        self.dio = dio; self.etime = etime
        dpath = os.path.join( self.xrdctpath, self.dataset)
        print dpath
        try:
            os.chdir(dpath)		

            self.mask = fabio.open(self.maskname)
            self.mask = np.array(self.mask.data)
#            self.data = np.zeros((self.nt,self.na,self.npt_rad-10));
            
        except:
            print "Cannot open mask file or the xrd-ct dataset directory is wrong"
            
    def run(self):

        try:
            self.data = np.zeros((self.nt*self.na,self.npt_rad-10));
            
            ai = pyFAI.load(self.poniname)        
#            ii = 0
#            for a in range(0,int(self.na)):
#                for t in range(0,int(self.nt)):
            print self.xrdctpath, self.dataset, self.prefix
            for ii in range(0,self.nt*self.na):
                    start=time.time()

                    if self.datatype == 'cbf':
                        s = '%s/%s_%.4d.cbf' % (self.dataset,self.prefix,ii)
                    else:
                        s = '%s/%s_%.4d.edf' % (self.dataset,self.prefix,ii)
                    pat = os.path.join(self.xrdctpath, s)
                    if os.path.exists(pat) and ii>0:
                        f = fabio.open(pat)
                        d = np.array(f.data)
    
                        if self.filt == "No":
                            if self.procunit == "CPU":
                                r, I = ai.integrate1d(data=d, npt=self.npt_rad, mask=self.mask, unit=self.units, method="cython",correctSolidAngle=False, polarization_factor=0.95)
                            elif self.procunit == "GPU":
                                r, I = ai.integrate1d(data=d, npt=self.npt_rad, mask=self.mask, unit=self.units, method="ocl_CSR_gpu",correctSolidAngle=False, polarization_factor=0.95)
                        elif self.filt == "Median":
                            if self.procunit == "CPU":
                                r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=50, unit=self.units, mask=self.mask, method="splitpixel", correctSolidAngle=False, polarization_factor=0.95);
                            elif self.procunit == "GPU":
                                r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=50, unit=self.units, mask=self.mask, method="ocl_CSR_gpu", correctSolidAngle=False, polarization_factor=0.95);
                        elif self.filt == "trimmed_mean":
                            if self.procunit == "CPU":
                                r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=(np.round(float(self.prc)/2),100-np.round(float(self.prc)/2)), unit=self.units, mask=self.mask, method="cython", correctSolidAngle=False, polarization_factor=0.95); #"splitpixel"
                            elif self.procunit == "GPU":
                                r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=(np.round(float(self.prc)/2),100-np.round(float(self.prc)/2)), unit=self.units, mask=self.mask, method="ocl_CSR_gpu", correctSolidAngle=False, polarization_factor=0.95); 
                        elif self.filt == "sigma":
                            if self.procunit == "CPU":
                                r, I, std = ai.sigma_clip(data=d, npt_rad=int(self.npt_rad), thres=float(self.thres), max_iter=5, unit=self.units, mask=self.mask, method="cython", correctSolidAngle=False, polarization_factor=0.95);
                            elif self.procunit == "GPU":
                                r, I, std = ai.sigma_clip(data=d, npt_rad=int(self.npt_rad), thres=float(self.thres), max_iter=5, unit=self.units, mask=self.mask, method="ocl_CSR_gpu", correctSolidAngle=False, polarization_factor=0.95);
    #                    self.data[t,a,:] = I[0:self.npt_rad-10]
                        self.data[ii,:] = I[0:self.npt_rad-10]
                        
                        v = np.round((100.*(ii+1))/(int(self.na)*int(self.nt)))
                        self.progress.emit(v)
                        ii += 1;
                        print s, time.time()-start
                    
            print "Integration done, now saving the data"
            self.r = r[0:self.npt_rad-10]
            self.tth = self.r
            self.tth2q()

            self.writehdf5()
            
            print "Saving data done"   
            self.squeeze.emit()
            
        except:
            print "Something is wrong with the integration..."

    def tth2q(self):
        self.h = 6.620700406E-34;self.c = 3E8
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1E3)
        self.d = self.wavel/(2*np.sin(np.deg2rad(0.5*self.tth)))
        self.q = np.pi*2/self.d
                
    def writehdf5(self):


        if self.filt == "No":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "Median":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "trimmed_mean":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.prc), self.filt,self.procunit)
        elif self.filt == "sigma":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.thres), self.filt,self.procunit)                

#        fn = "%s/%s_squeezed_%s.hdf5" % (self.savepath, self.dataset, self.filt)
        h5f = h5py.File(fn, "w")
        h5f.create_dataset('data', data=self.data)
        h5f.create_dataset('slow_axis_steps', data=self.na)
        h5f.create_dataset('fast_axis_steps', data=self.nt)
        h5f.create_dataset('q', data=self.q)
        h5f.create_dataset('twotheta', data=self.tth)
        h5f.create_dataset('d', data=self.d)
        h5f.create_dataset('omega', data=self.omega)
        h5f.create_dataset('y', data=self.trans)
        h5f.create_dataset('energy', data=self.E)
        h5f.create_dataset('scantype', data=self.scantype)
        h5f.create_dataset('diode', data=self.dio)
        h5f.create_dataset('exptime', data=self.etime)
        h5f.close()
    	
        os.chdir(self.savepath)
        perm = 'chmod 777 %s' %fn
        os.system(perm)
        

class XRDCT_ID15ASqueeze(QtCore.QThread, SliceTransformations, HyperSlice): 
    progress = QtCore.pyqtSignal(int)

    
    def __init__(self,prefix,dataset,xrdctpath,maskname,poniname,na,nt,npt_rad,filt,procunit,units,prc,thres,datatype,savepath,scantype,energy,jsonname,omega,trans,dio,etime):
        QtCore.QThread.__init__(self)
        self.prefix = prefix; self.dataset=dataset; self.xrdctpath = xrdctpath; self.maskname = maskname; self.poniname = poniname
        self.filt=filt; self.procunit=procunit;self.units =units;self.E = energy
        self.prc=prc;self.thres = thres;self.datatype = datatype;self.savepath = savepath
        self.na = int(na); self.nt = int(nt); self.npt_rad = int(npt_rad); self.scantype = scantype; self.jsonname = jsonname
        self.omega = omega; self.trans = trans;self.dio = dio; self.etime = etime
        self.gpuxrdctpath = '/gz%s' %self.xrdctpath
        
                
        dpath = os.path.join( self.xrdctpath, self.dataset)
        try:
            os.chdir(dpath)		

            self.mask = fabio.open(self.maskname)
            self.mask = np.array(self.mask.data)
            
#            self.sinos = np.zeros((self.nt*self.na,2000));

            if self.filt == "No":
#	            self.sinos = np.zeros((self.nt*self.na,2000));
                self.sinos = np.zeros((self.nt*self.na,self.npt_rad));
            else:
                self.sinos = np.zeros((self.nt*self.na,self.npt_rad-10));
            print self.sinos.shape
            
        except:
            print "Cannot open mask file or the xrd-ct dataset directory is wrong"
            

    def run(self):
        self.previousLine = 0
        self.nextLine = 1        
        self.nextimageFile  =  "%s/%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nt)     
        print self.nextimageFile    
          
        self.periodEventraw = Periodic(1, self.checkForImageFile)

    def checkForImageFile(self):
        
        if os.path.exists(self.nextimageFile) & (self.nextLine == 1):
            print('%s exists' % self.nextimageFile)
            self.periodEventraw.stop()
            self.writeLineData()
            self.setnextimageFile()       
        elif os.path.exists(self.nextimageFile) & (self.nextLine < self.na+1) & os.path.exists(self.previoush5File):
            print('%s exists' % self.nextimageFile)
            self.periodEventraw.stop()
            self.writeLineData()
            self.setnextimageFile()     
        else: # keep looking
            print('%s' %self.nextimageFile ,'does not exist')
            print('or %s does not exist' % self.previoush5File)

    def writeLineData(self):

        self.json = "%s%s_%.4d.json" % (self.savepath, self.dataset, self.nextLine)
        print(self.json)
	
        self.creategpujson() # write the .json files
        self.gpuproc() # integrate the 2D diffraction patterns
            
    def setnextimageFile(self):
        self.nextLine += 1
        self.previoush5File =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLine-1)
        if self.nextLine < self.na:
            self.nextimageFile =  "%s%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.nt)
            print self.nextimageFile
            self.periodEventraw.start()
        elif self.nextLine == self.na: #might need a sleep to give time to write the last image
            self.nextimageFile =  "%s%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.nt-1)
            print self.nextimageFile
            self.periodEventraw.start()            
        else:
            print('Integration done')
#            self.squeeze.emit()
#            self.tth2q()
#            self.writesinos()            
#            print "Saving data done"  
#            self.removedata()            
#            print "Deleted integrated linescans"  
#            
        v = np.round((100*self.nextLine/self.na))
        self.progress.emit(v)
            
    def tth2q(self):
        self.h = 6.620700406E-34;self.c = 3E8
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1E3)
        self.d = self.wavel/(2*np.sin(np.deg2rad(0.5*self.tth)))
        self.q = np.pi*2/self.d
            
    def liveRead(self):
        self.nextLineNumber = 1
        
    #### For GPU machine
        self.nextFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)
        self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber+1)
        self.periodEvent = Periodic(1, self.checkForFile)
        
    def checkForFile(self):
        if os.path.exists(self.nextFile) & os.path.exists(self.targetFile):
            print('%s exists' % self.nextFile)
            self.periodEvent.stop()
            self.hdfLineScanRead()
            self.setNextFile()
            self.setTargetFile()
        else: # keep looking
            print('%s or %s does not exist' %(self.nextFile,self.targetFile))

    def hdfLineScanRead(self):
        with h5py.File(self.nextFile,'r') as f:
        #### For GPU  
            if self.filt == "No":
                data = f['/entry_0000/PyFAI/process_integrate1d/I'][:]
                self.tth = f['/entry_0000/PyFAI/process_integrate1d/2th'][:]
            elif self.filt == "Median":
                data = f['/entry_0000/PyFAI/process_medfilt1d/I'][:,0:self.npt_rad-10]
                self.tth = f['/entry_0000/PyFAI/process_medfilt1d/2th'][0:self.npt_rad-10]             
            elif self.filt == "trimmed_mean":
                data = f['/entry_0000/PyFAI/process_medfilt1d/I'][:,0:self.npt_rad-10]     
                self.tth = f['/entry_0000/PyFAI/process_medfilt1d/2th'][0:self.npt_rad-10]
            elif self.filt == "sigma":
                data = f['/entry_0000/PyFAI/process_sigma_clip/I'][:,0:self.npt_rad-10]
                self.tth = f['/entry_0000/PyFAI/process_sigma_clip/2th'][0:self.npt_rad-10]        
        f.close
        
        filei = self.nt*(self.nextLineNumber-1)				
        filef = self.nt*self.nextLineNumber
        self.sinos[range(filei, filef),:] = data
        
    def setNextFile(self):
        self.nextLineNumber += 1
        
        #### For GPU
        if self.nextLineNumber < self.na+1:
            self.nextFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)

            print self.nextFile
            self.periodEvent.start()
        else:
            print('All done')
            self.tth2q()
            self.writesinos()            
            print "Saving data done"  
            self.removedata()            
            print "Deleted integrated linescans"     

    def setTargetFile(self):
        #### For GPU        
        if self.nextLineNumber < self.na:
            self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber+1)
        elif self.nextLineNumber == self.na:
            self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)

    def writesinos(self):

        if self.filt == "No":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "Median":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "trimmed_mean":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.prc), self.filt,self.procunit)
        elif self.filt == "sigma":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.thres), self.filt,self.procunit)                
        
        h5f = h5py.File(fn, "w")
        h5f.create_dataset('data', data=self.sinos)
        h5f.create_dataset('twotheta', data=self.tth)
        h5f.create_dataset('q', data=self.q)
        h5f.create_dataset('d', data=self.d)        
        h5f.create_dataset('slow_axis_steps', data=self.na)
        h5f.create_dataset('fast_axis_steps', data=self.nt)
        h5f.create_dataset('diode', data=self.dio)
        h5f.create_dataset('exptime', data=self.etime)
        h5f.create_dataset('omega', data=self.omega)
        h5f.create_dataset('translations', data=self.trans)      
        h5f.create_dataset('scantype', data=self.scantype)       
        h5f.close()
    	
        os.chdir(self.savepath)
        perm = 'chmod 777 %s' %fn
        os.system(perm)
        
    def removedata(self):
        fn = "%s/%s*.h5" % (self.savepath, self.dataset)
        cmd = 'rm %s' %fn
        os.system(cmd)        

        fn = "%s/%s*.json" % (self.savepath, self.dataset)
        cmd = 'rm %s' %fn
        os.system(cmd)   
        
    def creategpujson(self):
        
        start=time.time()
        
        try:
            perm = 'chmod 777 %s.h5' %self.of
            os.system(perm)
        except:
            pass
        
        with open(self.jsonname, 'r') as f:
            data = json.load(f)
            data['poni_file'] = data['poni']
            data['do_SA'] = 0
            data['method'] = "ocl_csr_nosplit_gpu"
            
            data['npt'] = int(self.npt_rad)
#            data['npt_azim'] = 256
#            data['do_SA'] = False
#            data['do_polarziation'] = True
#            data['polarization_factor'] = 0.97
            data['dummy'] = -10.0
            data['delta_dummy'] = 9.5
			
            ####### Filters #########
            if self.filt == "No":
                data['plugin_name'] = 'id15.IntegrateManyFrames'
            elif self.filt == "Median":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "medfilt1d"
                data['percentile'] = 50
            elif self.filt == "trimmed_mean":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "medfilt1d"
                data['percentile'] = (np.round(float(self.prc)/2),100-np.round(float(self.prc)/2))         
            elif self.filt == "sigma":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "sigma_clip"                
                data['sigma_clip_thresold'] = float(self.thres)
                data['sigma_clip_max_iter'] = 5
            
    			# Number of xrd-ct dataset in the 3D scan
            for d in range(0, 1):  #range(self.ntomos, self.ntomos+1): 
                    #	da = '%s%.2d' %(self.dataset,d)
                da = '%s' %(self.dataset)
                cbfs = []
        
                filei = self.nt*(self.nextLine-1)				
                filef = self.nt*self.nextLine
                for ii in range(filei, filef):
                    if self.datatype == "cbf":
                        f = '%s_%.4d.cbf' % (da,ii)
                    elif self.datatype == "edf":
                        f = '%s_%.4d.edf' % (da,ii)
        
                    cbf = os.path.join(self.xrdctpath, da, f)
                    cbfs.append(cbf)
        
            data['input_files'] = cbfs
            of = '%s/%s_%.4d.azim' % (self.savepath, da, self.nextLine)
            data['output_file'] = of
            self.of = of
            print data['output_file']
        
        self.jsonfile = '%s_%.4d.json' % (da, self.nextLine)
        job = os.path.join( self.savepath, self.jsonfile)
        with open(job, 'w') as outfile:  
            json.dump(data, outfile)

        os.chdir(self.savepath)
        
        perm = 'chmod 777 %s' %self.jsonfile
        os.system(perm) 
        
        print time.time()-start
        
    def gpuproc(self):
        dahu = 'dahu-reprocess %s &' %self.jsonfile #### try with &
        print 'processing dataset %s' %self.jsonfile	
        os.system(dahu)

class Fast_XRDCT_ID15ASqueeze(QtCore.QThread, SliceTransformations, HyperSlice): 
    progress = QtCore.pyqtSignal(int)

    
    def __init__(self,prefix,dataset,xrdctpath,maskname,poniname,na,nt,npt_rad,filt,procunit,units,prc,thres,datatype,savepath,scantype,energy,jsonname,omega,trans,dio,etime):
        QtCore.QThread.__init__(self)
        self.prefix = prefix; self.dataset=dataset; self.xrdctpath = xrdctpath; self.maskname = maskname; self.poniname = poniname
        self.filt=filt; self.procunit=procunit;self.units =units;self.E = energy
        self.prc=prc;self.thres = thres;self.datatype = datatype;self.savepath = savepath
        self.na = int(na); self.nt = int(nt); self.npt_rad = int(npt_rad); self.scantype = scantype; self.jsonname = jsonname
        self.omega = omega; self.trans = trans;self.dio = dio; self.etime = etime
        self.gpuxrdctpath = '/gz%s' %self.xrdctpath
        
                
        dpath = os.path.join( self.xrdctpath, self.dataset)
        try:
            os.chdir(dpath)		

            self.mask = fabio.open(self.maskname)
            self.mask = np.array(self.mask.data)
            
#            self.sinos = np.zeros((self.nt*self.na,2000));

            if self.filt == "No":
#	            self.sinos = np.zeros((self.nt*self.na,2000));
                self.sinos = np.zeros((self.nt*self.na,self.npt_rad));
            else:
                self.sinos = np.zeros((self.nt*self.na,self.npt_rad-10));
            print self.sinos.shape
            
        except:
            print "Cannot open mask file or the xrd-ct dataset directory is wrong"
            

    def run(self):
        self.previousLine = 0
        self.nextLine = 1        
        self.nextimageFile  =  "%s/%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.na)     
        print self.nextimageFile    
          
        self.periodEventraw = Periodic(1, self.checkForImageFile)

    def checkForImageFile(self):
        
        if os.path.exists(self.nextimageFile) & (self.nextLine == 1):
            print('%s exists' % self.nextimageFile)
            self.periodEventraw.stop()
            self.writeLineData()
            self.setnextimageFile()       
        elif os.path.exists(self.nextimageFile) & (self.nextLine < self.nt+1) & os.path.exists(self.previoush5File):
            print('%s exists' % self.nextimageFile)
            self.periodEventraw.stop()
            self.writeLineData()
            self.setnextimageFile()     
        else: # keep looking
            print('%s' %self.nextimageFile ,'does not exist')
            print('or %s does not exist' % self.previoush5File)

    def writeLineData(self):

        self.json = "%s%s_%.4d.json" % (self.savepath, self.dataset, self.nextLine)
        print(self.json)
	
        self.creategpujson() # write the .json files
        self.gpuproc() # integrate the 2D diffraction patterns
            
    def setnextimageFile(self):
        self.nextLine += 1
        self.previoush5File =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLine-1)
        if self.nextLine < self.nt:
            self.nextimageFile =  "%s%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.na)
            print self.nextimageFile
            self.periodEventraw.start()
        elif self.nextLine == self.nt: #might need a sleep to give time to write the last image
            self.nextimageFile =  "%s%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.na-1)
            print self.nextimageFile
            self.periodEventraw.start()            
        else:
            print('Integration done')
#            self.squeeze.emit()
#            self.tth2q()
#            self.writesinos()            
#            print "Saving data done"  
#            self.removedata()            
#            print "Deleted integrated linescans"  
#            
        v = np.round((100*self.nextLine/self.nt))
        self.progress.emit(v)
            
    def tth2q(self):
        self.h = 6.620700406E-34;self.c = 3E8
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1E3)
        self.d = self.wavel/(2*np.sin(np.deg2rad(0.5*self.tth)))
        self.q = np.pi*2/self.d
            
    def liveRead(self):
        self.nextLineNumber = 1
        
    #### For GPU machine
        self.nextFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)
        self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber+1)
        self.periodEvent = Periodic(1, self.checkForFile)
        
    def checkForFile(self):
        if os.path.exists(self.nextFile) & os.path.exists(self.targetFile):
            print('%s exists' % self.nextFile)
            self.periodEvent.stop()
            self.hdfLineScanRead()
            self.setNextFile()
            self.setTargetFile()
        else: # keep looking
            print('%s or %s does not exist' %(self.nextFile,self.targetFile))

    def hdfLineScanRead(self):
        with h5py.File(self.nextFile,'r') as f:
        #### For GPU  
            if self.filt == "No":
                data = f['/entry_0000/PyFAI/process_integrate1d/I'][:]
                self.tth = f['/entry_0000/PyFAI/process_integrate1d/2th'][:]
            elif self.filt == "Median":
                data = f['/entry_0000/PyFAI/process_medfilt1d/I'][:,0:self.npt_rad-10]
                self.tth = f['/entry_0000/PyFAI/process_medfilt1d/2th'][0:self.npt_rad-10]             
            elif self.filt == "trimmed_mean":
                data = f['/entry_0000/PyFAI/process_medfilt1d/I'][:,0:self.npt_rad-10]     
                self.tth = f['/entry_0000/PyFAI/process_medfilt1d/2th'][0:self.npt_rad-10]
            elif self.filt == "sigma":
                data = f['/entry_0000/PyFAI/process_sigma_clip/I'][:,0:self.npt_rad-10]
                self.tth = f['/entry_0000/PyFAI/process_sigma_clip/2th'][0:self.npt_rad-10]        
        f.close
        
        filei = self.na*(self.nextLineNumber-1)				
        filef = self.na*self.nextLineNumber
        self.sinos[range(filei, filef),:] = data
        
    def setNextFile(self):
        self.nextLineNumber += 1
        
        #### For GPU
        if self.nextLineNumber < self.nt+1:
            self.nextFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)

            print self.nextFile
            self.periodEvent.start()
        else:
            print('All done')
            self.tth2q()
            self.writesinos()            
            print "Saving data done"  
            self.removedata()            
            print "Deleted integrated linescans"     

    def setTargetFile(self):
        #### For GPU        
        if self.nextLineNumber < self.nt:
            self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber+1)
        elif self.nextLineNumber == self.nt:
            self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)

    def writesinos(self):

        if self.filt == "No":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "Median":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "trimmed_mean":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.prc), self.filt,self.procunit)
        elif self.filt == "sigma":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.thres), self.filt,self.procunit)                
        
        h5f = h5py.File(fn, "w")
        h5f.create_dataset('data', data=self.sinos)
        h5f.create_dataset('twotheta', data=self.tth)
        h5f.create_dataset('q', data=self.q)
        h5f.create_dataset('d', data=self.d)        
        h5f.create_dataset('slow_axis_steps', data=self.na)
        h5f.create_dataset('fast_axis_steps', data=self.nt)
        h5f.create_dataset('diode', data=self.dio)
        h5f.create_dataset('exptime', data=self.etime)
        h5f.create_dataset('omega', data=self.omega)
        h5f.create_dataset('translations', data=self.trans)      
        h5f.create_dataset('scantype', data=self.scantype)       
        h5f.close()
    	
        os.chdir(self.savepath)
        perm = 'chmod 777 %s' %fn
        os.system(perm)
        
    def removedata(self):
        fn = "%s/%s*.h5" % (self.savepath, self.dataset)
        cmd = 'rm %s' %fn
        os.system(cmd)        

        fn = "%s/%s*.json" % (self.savepath, self.dataset)
        cmd = 'rm %s' %fn
        os.system(cmd)   
        
    def creategpujson(self):
        
        start=time.time()
        
        try:
            perm = 'chmod 777 %s.h5' %self.of
            os.system(perm)
        except:
            pass
        
        with open(self.jsonname, 'r') as f:
            data = json.load(f)
            data['poni_file'] = data['poni']
            data['do_SA'] = 0
            data['method'] = "ocl_csr_nosplit_gpu"
            
            data['npt'] = int(self.npt_rad)
#            data['npt_azim'] = 256
#            data['do_SA'] = False
#            data['do_polarziation'] = True
#            data['polarization_factor'] = 0.97
            data['dummy'] = -10.0
            data['delta_dummy'] = 9.5
			
            ####### Filters #########
            if self.filt == "No":
                data['plugin_name'] = 'id15.IntegrateManyFrames'
            elif self.filt == "Median":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "medfilt1d"
                data['percentile'] = 50
            elif self.filt == "trimmed_mean":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "medfilt1d"
                data['percentile'] = (np.round(float(self.prc)/2),100-np.round(float(self.prc)/2))         
            elif self.filt == "sigma":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "sigma_clip"                
                data['sigma_clip_thresold'] = float(self.thres)
                data['sigma_clip_max_iter'] = 5
            
    			# Number of xrd-ct dataset in the 3D scan
            for d in range(0, 1):  #range(self.ntomos, self.ntomos+1): 
                    #	da = '%s%.2d' %(self.dataset,d)
                da = '%s' %(self.dataset)
                cbfs = []
        
                filei = self.na*(self.nextLine-1)				
                filef = self.na*self.nextLine
                for ii in range(filei, filef):
                    if self.datatype == "cbf":
                        f = '%s_%.4d.cbf' % (da,ii)
                    elif self.datatype == "edf":
                        f = '%s_%.4d.edf' % (da,ii)
        
                    cbf = os.path.join(self.xrdctpath, da, f)
                    cbfs.append(cbf)
        
            data['input_files'] = cbfs
            of = '%s/%s_%.4d.azim' % (self.savepath, da, self.nextLine)
            data['output_file'] = of
            self.of = of
            print data['output_file']
        
        self.jsonfile = '%s_%.4d.json' % (da, self.nextLine)
        job = os.path.join( self.savepath, self.jsonfile)
        with open(job, 'w') as outfile:  
            json.dump(data, outfile)

        os.chdir(self.savepath)
        
        perm = 'chmod 777 %s' %self.jsonfile
        os.system(perm) 
        
        print time.time()-start
        
    def gpuproc(self):
        dahu = 'dahu-reprocess %s &' %self.jsonfile #### try with &
        print 'processing dataset %s' %self.jsonfile	
        os.system(dahu)
        
class XRDCT_LiveSqueeze(QtCore.QThread): 
    progress = QtCore.pyqtSignal(int)

    
    def __init__(self,prefix,dataset,xrdctpath,maskname,poniname,na,nt,npt_rad,filt,procunit,units,prc,thres,datatype,savepath,scantype,energy,jsonname,omega,trans,dio,etime):
        QtCore.QThread.__init__(self)
        self.prefix = prefix; self.dataset=dataset; self.xrdctpath = xrdctpath; self.maskname = maskname; self.poniname = poniname
        self.filt=filt; self.procunit=procunit;self.units =units;self.E = energy
        self.prc=prc;self.thres = thres;self.datatype = datatype;self.savepath = savepath
        self.na = int(na); self.nt = int(nt); self.npt_rad = int(npt_rad); self.scantype = scantype; self.jsonname = jsonname
        self.omega = omega; self.trans = trans;self.dio = dio; self.etime = etime
        self.gpuxrdctpath = '/gz%s' %self.xrdctpath
        
        self.data = np.zeros((1,1,1))
                
        dpath = os.path.join(self.xrdctpath, self.dataset)
        try:
            os.chdir(dpath)		

            self.mask = fabio.open(self.maskname)
            self.mask = np.array(self.mask.data)
                        
        except:
            print "Cannot open mask file or the xrd-ct dataset directory is wrong"

    def run(self):
        self.previousLine = 0
        self.nextLine = 1        
        self.nextimageFile  =  "%s/%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nt)     
        print self.nextimageFile    
          
        
        if self.procunit == "MultiGPU":
            self.periodEventraw = Periodic(1, self.checkForImageFile)   
        else:
            self.periodEventraw = Periodic(1, self.checkForImageFileOar)

    def checkForImageFileOar(self):
            
        if os.path.exists(self.nextimageFile) & (self.nextLine == 1):
            print('%s exists' % self.nextimageFile)
            self.periodEventraw.stop()
            self.integrateLineDataOar()
            self.setnextimageFileOar()       
        elif os.path.exists(self.nextimageFile) & (self.nextLine < self.na+1) & os.path.exists(self.hdf5_fn):
            print('%s exists' % self.nextimageFile)
            self.periodEventraw.stop()
            self.integrateLineDataOar()
            self.setnextimageFileOar()     
        else: # keep looking
            print('%s' %self.nextimageFile ,'does not exist')
            print('or %s does not exist' % self.previousdatFile)
#            self.periodEventraw.stop()

    def setnextimageFileOar(self):

        self.previousLine += 1
        self.nextLine += 1
        if self.nextLine < self.na:

            self.nextimageFile =  "%s/%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.nt)
            print self.nextimageFile
            self.periodEventraw.start()

        elif self.nextLine == self.na: #might need a sleep to give time to write the last image
            
            self.nextimageFile =  "%s/%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.nt-1)
            print self.nextimageFile
            self.periodEventraw.start()   
        else:
            print('All done')
            self.isLive = False 
            self.squeeze.emit()
            
    def integrateLineDataOar(self):
        
        try:

            self.Int = np.zeros((self.nt,self.npt_rad-10));
         
            ai = pyFAI.load(self.poniname)       
            kk = 0
            for ii in range(int(self.nt)*self.previousLine,int(self.nt)*self.nextLine):             
            
                start=time.time()
                if self.datatype == 'cbf':
                    s = '%s/%s_%.4d.cbf' % (self.dataset,self.prefix,ii)
                else:
                    s = '%s/%s_%.4d.edf' % (self.dataset,self.prefix,ii)
                pat = os.path.join(self.xrdctpath, s)
                f = fabio.open(pat)
                d = np.array(f.data)
                if self.filt == "No":
                    if self.procunit == "CPU":
                        r, I = ai.integrate1d(data=d, npt=self.npt_rad, mask=self.mask, unit=self.units, method="cython",correctSolidAngle=False, polarization_factor=0.95)
                    elif self.procunit == "GPU":
                        r, I = ai.integrate1d(data=d, npt=self.npt_rad, mask=self.mask, unit=self.units, method="ocl_CSR_gpu",correctSolidAngle=False, polarization_factor=0.95)
                elif self.filt == "Median":
                    if self.procunit == "CPU":
                        r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=50, unit=self.units, mask=self.mask, method="splitpixel", correctSolidAngle=False, polarization_factor=0.95);
                    elif self.procunit == "GPU":
                        r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=50, unit=self.units, mask=self.mask, method="ocl_CSR_gpu", correctSolidAngle=False, polarization_factor=0.95);
                elif self.filt == "trimmed_mean":
                    if self.procunit == "CPU":
                        r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=(np.round(float(self.prc)/2),100-np.round(float(self.prc)/2)), unit=self.units, mask=self.mask, method="cython", correctSolidAngle=False, polarization_factor=0.95); #"splitpixel"
                    elif self.procunit == "GPU":
                        r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=(np.round(float(self.prc)/2),100-np.round(float(self.prc)/2)), unit=self.units, mask=self.mask, method="ocl_CSR_gpu", correctSolidAngle=False, polarization_factor=0.95); 
                elif self.filt == "sigma":
                    if self.procunit == "CPU":
                        r, I, std = ai.sigma_clip(data=d, npt_rad=int(self.npt_rad), thres=float(self.thres), max_iter=5, unit=self.units, mask=self.mask, method="cython", correctSolidAngle=False, polarization_factor=0.95);
                    elif self.procunit == "GPU":
                        r, I = ai.sigma_clip(data=d, npt_rad=int(self.npt_rad), thres=float(self.thres), max_iter=5, unit=self.units, mask=self.mask, method="ocl_CSR_gpu", correctSolidAngle=False, polarization_factor=0.95);
                self.Int[kk,:] = I[0:self.npt_rad-10]
                r = r[0:self.npt_rad-10]
                v = (np.round((100.*(ii+1))/(int(self.na)*int(self.nt))))
                self.progress.emit(v)
                kk += 1;
                print s, time.time()-start
                    
    
            print "Integration done, now saving the data"
            self.r = r
            if self.units=='q_A^-1':
                self.q = self.r
            elif self.units=='2th_deg':
                self.tth = self.r
                self.tth2q()
            self.writehdf5()

        except:
            print "Something is wrong with the actual integration..." 
        
    def tth2q(self):
        self.h = 6.620700406E-34;self.c = 3E8
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1E3)
        self.d = self.wavel/(2*np.sin(np.deg2rad(0.5*self.tth)))
        self.q = np.pi*2/self.d
            
    def writehdf5(self):
        self.hdf5_fn = "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLine)
        h5f = h5py.File(self.hdf5_fn, "w")
        h5f.create_dataset('I', data=self.Int)
        h5f.create_dataset('q', data=self.q)
        if self.units=='2th_deg':
            h5f.create_dataset('twotheta', data=self.tth)
            h5f.create_dataset('d', data=self.d)
        h5f.close()
    	
        os.chdir(self.savepath)
        os.system('chmod 777 *.hdf5')

    def checkForImageFile(self):
          
        if os.path.exists(self.nextimageFile) & (self.nextLine == 1):
            print('%s exists' % self.nextimageFile)
            self.periodEventraw.stop()
            self.writeLineData()
            self.setnextimageFile()       
        elif os.path.exists(self.nextimageFile) & (self.nextLine < self.na+1) & os.path.exists(self.previoush5File):
            print('%s exists' % self.nextimageFile)
            self.periodEventraw.stop()
            self.writeLineData()
            self.setnextimageFile()     
        else: # keep looking
            print('%s' %self.nextimageFile ,'does not exist')
            print('or %s does not exist' % self.previoush5File)

    def writeLineData(self):

        self.json = "%s%s_%.4d.json" % (self.savepath, self.dataset, self.nextLine)
        print(self.json)
	
        self.creategpujson() # write the .json files
        self.gpuproc() # integrate the 2D diffraction patterns
            
    def setnextimageFile(self):
        self.nextLine += 1
        self.previoush5File =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLine-1)
        if self.nextLine < self.na:
            self.nextimageFile =  "%s%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.nt)
            print self.nextimageFile
            self.periodEventraw.start()
        elif self.nextLine == self.na: #might need a sleep to give time to write the last image
            self.nextimageFile =  "%s%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.nt-1)
            print self.nextimageFile
            self.periodEventraw.start()            
        else:
            print('All done')
         
        v = np.round((100*self.nextLine/self.na))
        self.progress.emit(v)
        
    def creategpujson(self):
        
        start=time.time()
        
        try:
            perm = 'chmod 777 %s.h5' %self.of
            os.system(perm)
        except:
            pass
        
        with open(self.jsonname, 'r') as f:
            data = json.load(f)
            data['poni_file'] = data['poni']
            data['do_SA'] = 0
            data['method'] = "ocl_csr_nosplit_gpu"
            
            data['npt'] = int(self.npt_rad)
            data['npt_azim'] = 256
            data['do_SA'] = False
            data['do_polarziation'] = True
            data['polarization_factor'] = 0.97
            data['dummy'] = -10.0
            data['delta_dummy'] = 9.5
			
            ####### Filters #########
            if self.filt == "No":
                data['plugin_name'] = 'id15.IntegrateManyFrames'
            elif self.filt == "Median":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "medfilt1d"
                data['percentile'] = 50
            elif self.filt == "trimmed_mean":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "medfilt1d"
                data['percentile'] = (np.round(float(self.prc)/2),100-np.round(float(self.prc)/2))         
            elif self.filt == "sigma":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "sigma_clip"                
                data['sigma_clip_thresold'] = float(self.thres)
                data['sigma_clip_max_iter'] = 5
            
    			# Number of xrd-ct dataset in the 3D scan
            for d in range(0, 1):  #range(self.ntomos, self.ntomos+1): 
                    #	da = '%s%.2d' %(self.dataset,d)
                da = '%s' %(self.dataset)
                cbfs = []
        
                filei = (self.nt)*(self.nextLine-1)				
                filef = (self.nt)*self.nextLine
                for ii in range(filei, filef):
                    if self.datatype == "cbf":
                        f = '%s_%.4d.cbf' % (da,ii)
                    elif self.datatype == "edf":
                        f = '%s_%.4d.edf' % (da,ii)
        
                    cbf = os.path.join(self.xrdctpath, da, f)
                    cbfs.append(cbf)
        
            data['input_files'] = cbfs
            of = '%s/%s_%.4d.azim' % (self.savepath, da, self.nextLine)
            data['output_file'] = of
            self.of = of
            print data['output_file']
        
        self.jsonfile = '%s_%.4d.json' % (da, self.nextLine)
        job = os.path.join( self.savepath, self.jsonfile)
        with open(job, 'w') as outfile:  
            json.dump(data, outfile)

        os.chdir(self.savepath)
        
        perm = 'chmod 777 %s' %self.jsonfile
        os.system(perm) 
        
        print time.time()-start
        
    def gpuproc(self):
        dahu = 'dahu-reprocess %s &' %self.jsonfile #### try with &
        print 'processing dataset %s' %self.jsonfile	
        os.system(dahu)

class XRDCT_LiveRead(QtCore.QThread): 
    updatedata = QtCore.pyqtSignal()
    exploredata = QtCore.pyqtSignal()
    
    def __init__(self,prefix,dataset,xrdctpath,maskname,poniname,na,nt,npt_rad,filt,procunit,units,prc,thres,datatype,savepath,scantype,energy,jsonname,omega,trans,dio,etime):
        QtCore.QThread.__init__(self)
        self.prefix = prefix; self.dataset=dataset; self.xrdctpath = xrdctpath; self.maskname = maskname; self.poniname = poniname
        self.filt=filt; self.procunit=procunit;self.units =units;self.E = energy
        self.prc=prc;self.thres = thres;self.datatype = datatype;self.savepath = savepath
        self.na = int(na); self.nt = int(nt); self.npt_rad = int(npt_rad); self.scantype = scantype; self.jsonname = jsonname
        self.omega = omega; self.trans = trans;self.dio = dio; self.etime = etime
        self.gpuxrdctpath = '/gz%s' %self.xrdctpath
        
        self.data = np.zeros((1,1,1))
                
        dpath = os.path.join(self.xrdctpath, self.dataset)
        try:
            os.chdir(dpath)		

            self.mask = fabio.open(self.maskname)
            self.mask = np.array(self.mask.data)
            
            if self.filt == "No":
	            self.sinos = np.zeros((self.nt*self.na,2000));
            else:
                self.sinos = np.zeros((self.nt*self.na,self.npt_rad-10));
            print self.sinos.shape
            
        except:
            print "Cannot open mask file or the xrd-ct dataset directory is wrong"

    def run(self):
        self.nextLineNumber = 1

#        self.name = 'Live Sinogram'
#        self.isLive = True
#        self.explore()
        
        if self.procunit == "MultiGPU":
        #### For GPU machine
            self.nextFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)
            self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber+1)
            self.periodEvent = Periodic(1, self.checkForFile)
        else:
        #### For OAR
            self.nextFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber)
            self.targetFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber+1)
            self.periodEvent = Periodic(1, self.checkForFile)


    def checkForFile(self):
        if os.path.exists(self.nextFile) & os.path.exists(self.targetFile):
            print('%s exists' % self.nextFile)
            self.periodEvent.stop()
            self.hdfLineScanRead()
            self.setNextFile()
            self.setTargetFile()
            ### I should be sending a signal here, this should be caught by the GUI and update the figures
            self.updatedata.emit()
        else: # keep looking
            print('%s or %s does not exist' %(self.nextFile,self.targetFile))
           
    def hdfLineScanRead(self):
        with h5py.File(self.nextFile,'r') as f:
            if self.procunit == "MultiGPU":
            #### For GPU  
                if self.filt == "No":
                    data = f['/entry_0000/PyFAI/process_integrate1d/I'][:]
                    self.tth = f['/entry_0000/PyFAI/process_integrate1d/2th'][:]
                elif self.filt == "Median":
                    data = f['/entry_0000/PyFAI/process_medfilt1d/I'][:,0:self.npt_rad-10]
                    self.tth = f['/entry_0000/PyFAI/process_medfilt1d/2th'][0:self.npt_rad-10]             
                elif self.filt == "trimmed_mean":
                    data = f['/entry_0000/PyFAI/process_medfilt1d/I'][:,0:self.npt_rad-10]     
                    self.tth = f['/entry_0000/PyFAI/process_medfilt1d/2th'][0:self.npt_rad-10]
                elif self.filt == "sigma":
                    data = f['/entry_0000/PyFAI/process_sigma_clip/I'][:,0:self.npt_rad-10]
                    self.tth = f['/entry_0000/PyFAI/process_sigma_clip/2th'][0:self.npt_rad-10]           

            else:
            #### For OAR
                data = f['/I'][:]
                self.tth = f['/twotheta'][:]
            f.close
            self.tth2q()
            
        if self.nextLineNumber == 1: # this is a fix as we dont know the number of channels or the x scale
            self.nbin = data.shape[1] 
            self.nrow = self.nt
            self.ncol = self.na
            self.data = np.zeros((self.nrow, self.ncol, self.nbin))
            print self.data.shape

        self.data[:,self.nextLineNumber-1,:] = data
        if self.scantype == "zigzag":
            self.colFliplive()            
        self.meanSpectra = self.data.mean(axis=tuple(range(0, 2)))

        if self.nextLineNumber == 1:
            self.exploredata.emit()

    def tth2q(self):
        self.h = 6.620700406E-34;self.c = 3E8
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1E3)
        self.d = self.wavel/(2*np.sin(np.deg2rad(0.5*self.tth)))
        self.q = np.pi*2/self.d

    def colFliplive(self):
        if np.mod(self.nextLineNumber-1,2) == 0:
            self.data[:,self.nextLineNumber-1,:] = self.data[::-1,self.nextLineNumber-1,:]
            
    def setNextFile(self):
        self.nextLineNumber += 1
        
        if self.procunit == "MultiGPU":
            #### For GPU
            if self.nextLineNumber < self.na+1:
                self.nextFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)

                print self.nextFile
                self.periodEvent.start()
            else:
                print('All done')
                self.isLive = False     
                self.tth2q()
                self.writesinos()            
                print "Saving data done"  
                self.removedata()            
                print "Deleted integrated linescans"     
                
        else: 
            #### For OAR
            if self.nextLineNumber < self.na+1:
                self.nextFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber)

                print self.nextFile
                self.periodEvent.start()
            else:
                print('All done')
                self.isLive = False     
                self.tth2q()
                self.writesinos()            
                print "Saving data done"  
                self.removedata()            
                print "Deleted integrated linescans"                

    def setTargetFile(self):
        
        if self.procunit == "MultiGPU":
            #### For GPU        
            if self.nextLineNumber < self.na:
                self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber+1)
            elif self.nextLineNumber == self.na:
                self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)

        else:
            #### For OAR
            if self.nextLineNumber < self.na:
                self.targetFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber+1)
            elif self.nextLineNumber == self.na:
                self.targetFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber)
                

    def writesinos(self):
        
        if self.filt == "No":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "Median":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "trimmed_mean":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.prc), self.filt,self.procunit)
        elif self.filt == "sigma":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.thres), self.filt,self.procunit)                
        
        h5f = h5py.File(fn, "w")
        h5f.create_dataset('data', data=self.data)
        h5f.create_dataset('twotheta', data=self.tth)
        h5f.create_dataset('q', data=self.q)
        h5f.create_dataset('d', data=self.d)        
        h5f.create_dataset('slow_axis_steps', data=self.na)
        h5f.create_dataset('fast_axis_steps', data=self.nt)
        h5f.create_dataset('diode', data=self.dio)
        h5f.create_dataset('exptime', data=self.etime)
        h5f.create_dataset('omega', data=self.omega)
        h5f.create_dataset('translations', data=self.trans)      
        h5f.create_dataset('scantype', data=self.scantype)       
        h5f.close()
        
        os.chdir(self.savepath)
        perm = 'chmod 777 %s' %fn
        os.system(perm)
        
    def removedata(self):

        if self.procunit == "MultiGPU":
            try:
                fn = "%s/%s*.azim.h5" % (self.savepath, self.dataset)
                cmd = 'rm %s' %fn
                os.system(cmd)        
        
                fn = "%s/%s*.json" % (self.savepath, self.dataset)
                cmd = 'rm %s' %fn
                os.system(cmd)
            except:
                print 'No .azim.h5 files present in the folder'
        
        else:
            try:
                fn = "%s/%s*linescan*.hdf5" % (self.savepath, self.dataset)
                cmd = 'rm %s' %fn
                os.system(cmd)        
        
                fn = "%s/%s*.json" % (self.savepath, self.dataset)
                cmd = 'rm %s' %fn
                os.system(cmd)
            except:
                print 'No *linescan*.hdf5 files present in the folder'        
                
class Fast_XRDCT_LiveSqueeze(QtCore.QThread): 
    progress = QtCore.pyqtSignal(int)
    
    def __init__(self,prefix,dataset,xrdctpath,maskname,poniname,na,nt,npt_rad,filt,procunit,units,prc,thres,datatype,savepath,scantype,energy,jsonname,omega,trans,dio,etime):
        QtCore.QThread.__init__(self)
        self.prefix = prefix; self.dataset=dataset; self.xrdctpath = xrdctpath; self.maskname = maskname; self.poniname = poniname
        self.filt=filt; self.procunit=procunit;self.units =units;self.E = energy
        self.prc=prc;self.thres = thres;self.datatype = datatype;self.savepath = savepath
        self.na = int(na); self.nt = int(nt); self.npt_rad = int(npt_rad); self.scantype = scantype; self.jsonname = jsonname
        self.omega = omega; self.trans = trans;self.dio = dio; self.etime = etime
        self.gpuxrdctpath = '/gz%s' %self.xrdctpath
        
        self.data = np.zeros((1,1,1))
                
        dpath = os.path.join(self.xrdctpath, self.dataset)
        try:
            os.chdir(dpath)		

            self.mask = fabio.open(self.maskname)
            self.mask = np.array(self.mask.data)
                        
        except:
            print "Cannot open mask file or the xrd-ct dataset directory is wrong"

    def run(self):
        self.previousLine = 0
        self.nextLine = 1        
        self.nextimageFile  =  "%s/%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.na)     
        print self.nextimageFile    
        if self.procunit == "MultiGPU":
            self.periodEventraw = Periodic(1, self.checkForImageFile)   
        else:
            self.periodEventraw = Periodic(1, self.checkForImageFileOar)   

    def checkForImageFileOar(self):
            
        if os.path.exists(self.nextimageFile) & (self.nextLine == 1):
            print('%s exists' % self.nextimageFile)
            self.periodEventraw.stop()
            self.integrateLineDataOar()
            self.setnextimageFileOar()       
        elif os.path.exists(self.nextimageFile) & (self.nextLine < self.nt+1) & os.path.exists(self.hdf5_fn):
            print('%s exists' % self.nextimageFile)
            self.periodEventraw.stop()
            self.integrateLineDataOar()
            self.setnextimageFileOar()     
        else: # keep looking
            print('%s' %self.nextimageFile ,'does not exist')
            print('or %s does not exist' % self.previousdatFile)
#            self.periodEventraw.stop()

    def setnextimageFileOar(self):

        self.previousLine += 1
        self.nextLine += 1
        if self.nextLine < self.nt:

            self.nextimageFile =  "%s/%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.na)
            print self.nextimageFile
            self.periodEventraw.start()

        elif self.nextLine == self.nt: #might need a sleep to give time to write the last image
            
            self.nextimageFile =  "%s/%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.na-1)
            print self.nextimageFile
            self.periodEventraw.start()            
        else:
            print('All done')
            
    def integrateLineDataOar(self):
        
        try:

            self.Int = np.zeros((self.na,self.npt_rad-10));
         
            ai = pyFAI.load(self.poniname)       
            kk = 0
            for ii in range(int(self.na)*self.previousLine,int(self.na)*self.nextLine):             
            
                start=time.time()
                if self.datatype == 'cbf':
                    s = '%s/%s_%.4d.cbf' % (self.dataset,self.prefix,ii)
                else:
                    s = '%s/%s_%.4d.edf' % (self.dataset,self.prefix,ii)
                pat = os.path.join(self.xrdctpath, s)
                f = fabio.open(pat)
                d = np.array(f.data)
                if self.filt == "No":
                    if self.procunit == "CPU":
                        r, I = ai.integrate1d(data=d, npt=self.npt_rad, mask=self.mask, unit=self.units, method="cython",correctSolidAngle=False, polarization_factor=0.95)
                    elif self.procunit == "GPU":
                        r, I = ai.integrate1d(data=d, npt=self.npt_rad, mask=self.mask, unit=self.units, method="ocl_CSR_gpu",correctSolidAngle=False, polarization_factor=0.95)
                elif self.filt == "Median":
                    if self.procunit == "CPU":
                        r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=50, unit=self.units, mask=self.mask, method="splitpixel", correctSolidAngle=False, polarization_factor=0.95);
                    elif self.procunit == "GPU":
                        r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=50, unit=self.units, mask=self.mask, method="ocl_CSR_gpu", correctSolidAngle=False, polarization_factor=0.95);
                elif self.filt == "trimmed_mean":
                    if self.procunit == "CPU":
                        r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=(np.round(float(self.prc)/2),100-np.round(float(self.prc)/2)), unit=self.units, mask=self.mask, method="cython", correctSolidAngle=False, polarization_factor=0.95); #"splitpixel"
                    elif self.procunit == "GPU":
                        r, I = ai.medfilt1d(data=d, npt_rad=int(self.npt_rad), percentile=(np.round(float(self.prc)/2),100-np.round(float(self.prc)/2)), unit=self.units, mask=self.mask, method="ocl_CSR_gpu", correctSolidAngle=False, polarization_factor=0.95); 
                elif self.filt == "sigma":
                    if self.procunit == "CPU":
                        r, I, std = ai.sigma_clip(data=d, npt_rad=int(self.npt_rad), thres=float(self.thres), max_iter=5, unit=self.units, mask=self.mask, method="cython", correctSolidAngle=False, polarization_factor=0.95);
                    elif self.procunit == "GPU":
                        r, I, std = ai.sigma_clip(data=d, npt_rad=int(self.npt_rad), thres=float(self.thres), max_iter=5, unit=self.units, mask=self.mask, method="ocl_CSR_gpu", correctSolidAngle=False, polarization_factor=0.95);
                self.Int[kk,:] = I[0:self.npt_rad-10]
                r = r[0:self.npt_rad-10]
                
                v = (np.round((100.*(ii+1))/(int(self.na)*int(self.nt))))
                self.progress.emit(v)
                kk += 1;
                print s, time.time()-start
                    
    
            print "Integration done, now saving the data"
            self.r = r
            if self.units=='q_A^-1':
                self.q = self.r
            elif self.units=='2th_deg':
                self.tth = self.r
                self.tth2q()
            self.writehdf5()

        except:
            print "Something is wrong with the actual integration..." 
            exit
        
    def tth2q(self):
        self.h = 6.620700406E-34;self.c = 3E8
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1E3)
        self.d = self.wavel/(2*np.sin(np.deg2rad(0.5*self.tth)))
        self.q = np.pi*2/self.d
            
    def writehdf5(self):
        self.hdf5_fn = "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLine)
        h5f = h5py.File(self.hdf5_fn, "w")
        h5f.create_dataset('I', data=self.Int)
        h5f.create_dataset('q', data=self.q)
        if self.units=='2th_deg':
            h5f.create_dataset('twotheta', data=self.tth)
            h5f.create_dataset('d', data=self.d)
        h5f.close()
    	
        os.chdir(self.savepath)
        os.system('chmod 777 *.hdf5')
                 
    def checkForImageFile(self):
            
        if os.path.exists(self.nextimageFile) & (self.nextLine == 1):
            print('%s exists' % self.nextimageFile)
            self.periodEventraw.stop()
            self.writeLineData()
            self.setnextimageFile()       
        elif os.path.exists(self.nextimageFile) & (self.nextLine < self.nt+1) & os.path.exists(self.previoush5File):
            print('%s exists' % self.nextimageFile)
            self.periodEventraw.stop()
            self.writeLineData()
            self.setnextimageFile()     
        else: # keep looking
            print('%s' %self.nextimageFile ,'does not exist')
            print('or %s does not exist' % self.previoush5File)

    def writeLineData(self):

        self.json = "%s%s_%.4d.json" % (self.savepath, self.dataset, self.nextLine)
        print(self.json)
	
        self.creategpujson() # write the .json files
        self.gpuproc() # integrate the 2D diffraction patterns
            
    def setnextimageFile(self):
        self.nextLine += 1
        self.previoush5File =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLine-1)
        if self.nextLine < self.nt:
            self.nextimageFile =  "%s%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.na)
            print self.nextimageFile
            self.periodEventraw.start()
        elif self.nextLine == self.nt: #might need a sleep to give time to write the last image
            self.nextimageFile =  "%s%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.na-1)
            print self.nextimageFile
            self.periodEventraw.start()            
        else:
            print('All done')
            
        v = np.round((100*self.nextLine/self.nt))
        self.progress.emit(v) 
            
    def creategpujson(self):
        
        start=time.time()
        
        try:
            perm = 'chmod 777 %s.h5' %self.of
            os.system(perm)
        except:
            pass
        
        with open(self.jsonname, 'r') as f:
            data = json.load(f)
            data['poni_file'] = data['poni']
            data['do_SA'] = 0
            data['method'] = "ocl_csr_nosplit_gpu"
            
            data['npt'] = int(self.npt_rad)
#            data['npt_azim'] =  256
#            data['dummy'] = -10.0,
#            data['delta_dummy'] = 9.5            
			
            ####### Filters #########
            if self.filt == "No":
                data['plugin_name'] = 'id15.IntegrateManyFrames'
            elif self.filt == "Median":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "medfilt1d"
                data['percentile'] = 50
            elif self.filt == "trimmed_mean":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "medfilt1d"
                data['percentile'] = (np.round(float(self.prc)/2),100-np.round(float(self.prc)/2))         
            elif self.filt == "sigma":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "sigma_clip"                
                data['sigma_clip_thresold'] = float(self.thres)
                data['sigma_clip_max_iter'] = 5
            
    			# Number of xrd-ct dataset in the 3D scan
            for d in range(0, 1):  #range(self.ntomos, self.ntomos+1): 
                    #	da = '%s%.2d' %(self.dataset,d)
                da = '%s' %(self.dataset)
                cbfs = []
        
                filei = (self.na)*(self.nextLine-1)				
                filef = (self.na)*self.nextLine
                for ii in range(filei, filef):
                    if self.datatype == "cbf":
                        f = '%s_%.4d.cbf' % (da,ii)
                    elif self.datatype == "edf":
                        f = '%s_%.4d.edf' % (da,ii)
        
                    cbf = os.path.join(self.xrdctpath, da, f)
                    cbfs.append(cbf)
        
            data['input_files'] = cbfs
            of = '%s/%s_%.4d.azim' % (self.savepath, da, self.nextLine)
            data['output_file'] = of
            self.of = of
            print data['output_file']
        
        self.jsonfile = '%s_%.4d.json' % (da, self.nextLine)
        job = os.path.join( self.savepath, self.jsonfile)
        with open(job, 'w') as outfile:  
            json.dump(data, outfile)

        os.chdir(self.savepath)
        
        perm = 'chmod 777 %s' %self.jsonfile
        os.system(perm) 
        
        print time.time()-start
        
    def gpuproc(self):
        dahu = 'dahu-reprocess %s &' %self.jsonfile #### try with &
        print 'processing dataset %s' %self.jsonfile	
        os.system(dahu)

class Fast_XRDCT_LiveRead(QtCore.QThread): 
    updatedata = QtCore.pyqtSignal()
    exploredata = QtCore.pyqtSignal()
    
    def __init__(self,prefix,dataset,xrdctpath,maskname,poniname,na,nt,npt_rad,filt,procunit,units,prc,thres,datatype,savepath,scantype,energy,jsonname,omega,trans,dio,etime):
        QtCore.QThread.__init__(self)
        self.prefix = prefix; self.dataset=dataset; self.xrdctpath = xrdctpath; self.maskname = maskname; self.poniname = poniname
        self.filt=filt; self.procunit=procunit;self.units =units;self.E = energy
        self.prc=prc;self.thres = thres;self.datatype = datatype;self.savepath = savepath
        self.na = int(na); self.nt = int(nt); self.npt_rad = int(npt_rad); self.scantype = scantype; self.jsonname = jsonname
        self.omega = omega; self.trans = trans;self.dio = dio; self.etime = etime
        self.gpuxrdctpath = '/gz%s' %self.xrdctpath
        
        self.data = np.zeros((1,1,1))
                
        dpath = os.path.join( self.xrdctpath, self.dataset)
        try:
            os.chdir(dpath)		

            self.mask = fabio.open(self.maskname)
            self.mask = np.array(self.mask.data)
            
            if self.filt == "No":
	            self.sinos = np.zeros((self.nt*self.na,2000));
            else:
                self.sinos = np.zeros((self.nt*self.na,self.npt_rad-10));
            print self.sinos.shape
            
        except:
            print "Cannot open mask file or the xrd-ct dataset directory is wrong"

    def run(self):
        self.nextLineNumber = 1
        
        if self.procunit == "MultiGPU":
        #### For GPU machine
            self.nextFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)
            self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber+1)
            self.periodEvent = Periodic(1, self.checkForFile)
        else:
        #### For OAR
            self.nextFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber)
            self.targetFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber+1)
            self.periodEvent = Periodic(5, self.checkForFile)        

    def checkForFile(self):
        if os.path.exists(self.nextFile) & os.path.exists(self.targetFile):
            print('%s exists' % self.nextFile)
            self.periodEvent.stop()
            self.hdfLineScanRead()
            self.setNextFile()
            self.setTargetFile()
            ### I should be sending a signal here, this should be caught by the GUI and update the figures
            self.updatedata.emit()
        else: # keep looking
            print('%s or %s does not exist' %(self.nextFile,self.targetFile))

    def hdfLineScanRead(self):
        with h5py.File(self.nextFile,'r') as f:
            if self.procunit == "MultiGPU":
            #### For GPU  
                if self.filt == "No":
                    data = f['/entry_0000/PyFAI/process_integrate1d/I'][:] 
                    self.tth = f['/entry_0000/PyFAI/process_integrate1d/2th'][:]
                elif self.filt == "Median":
                    data = f['/entry_0000/PyFAI/process_medfilt1d/I'][:,0:self.npt_rad-10]
                    self.tth = f['/entry_0000/PyFAI/process_medfilt1d/2th'][0:self.npt_rad-10]             
                elif self.filt == "trimmed_mean":
                    data = f['/entry_0000/PyFAI/process_medfilt1d/I'][:,0:self.npt_rad-10]
                    self.tth = f['/entry_0000/PyFAI/process_medfilt1d/2th'][0:self.npt_rad-10]
                elif self.filt == "sigma":
                    data = f['/entry_0000/PyFAI/process_sigma_clip/I'][:,0:self.npt_rad-10]
                    self.tth = f['/entry_0000/PyFAI/process_sigma_clip/2th'][0:self.npt_rad-10]      
            else:
            #### For OAR
                data = f['/I'][:]
                self.tth = f['/twotheta'][:]
            f.close
            self.tth2q()
            
        if self.nextLineNumber == 1: # this is a fix as we dont know the number of channels or the x scale
            self.data = np.zeros((self.nt, self.na, data.shape[1] ))
            print self.data.shape
        
        v1 = len(np.arange(0,self.data.shape[0],2))
        v2 = len(np.arange(1,self.data.shape[0],2))
        ll = v1 + v2;
               
        if self.scantype == "fast":
            
            ind = int(np.ceil(float(self.nextLineNumber-1)/2))
            
            if np.mod(self.nextLineNumber-1,2) == 0:
                self.data[ind,:,:] = data
            else:
                self.data[ll-ind,:] = data #data[::-1]
                
        if self.nextLineNumber == 1:
            self.exploredata.emit()

    def tth2q(self):
        self.h = 6.620700406E-34;self.c = 3E8
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1E3)
        self.d = self.wavel/(2*np.sin(np.deg2rad(0.5*self.tth)))
        self.q = np.pi*2/self.d
        
    def setNextFile(self):
        self.nextLineNumber += 1
        
        if self.procunit == "MultiGPU":
            #### For GPU
            if self.nextLineNumber < self.nt+1:
                self.nextFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)

                print self.nextFile
                self.periodEvent.start()
            else:
                print('All done')
                self.isLive = False     
                self.tth2q()
                self.writesinos()            
                print "Saving data done"  
                self.removedata()            
                print "Deleted integrated linescans"  
                
        else: 
            #### For OAR
            if self.nextLineNumber < self.nt+1:
                self.nextFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber)

                print self.nextFile
                self.periodEvent.start()
            else:
                print('All done')
                self.isLive = False     
                self.tth2q()
                self.writesinos()            
                print "Saving data done"  
                self.removedata()            
                print "Deleted integrated linescans"               

    def setTargetFile(self):
        
        if self.procunit == "MultiGPU":
            #### For GPU        
            if self.nextLineNumber < self.nt:
                self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber+1)
            elif self.nextLineNumber == self.nt:
                self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)

        else:
            #### For OAR
            if self.nextLineNumber < self.nt:
                self.targetFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber+1)
            elif self.nextLineNumber == self.nt:
                self.targetFile =  "%s/%s_linescan_%.3d.hdf5" % (self.savepath, self.dataset, self.nextLineNumber)
        
    def writesinos(self):
        
        if self.filt == "No":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "Median":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "trimmed_mean":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.prc), self.filt,self.procunit)
        elif self.filt == "sigma":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.thres), self.filt,self.procunit)                

        h5f = h5py.File(fn, "w")
        h5f.create_dataset('data', data=self.data)
        h5f.create_dataset('twotheta', data=self.tth)
        h5f.create_dataset('q', data=self.q)
        h5f.create_dataset('d', data=self.d)        
        h5f.create_dataset('slow_axis_steps', data=self.na)
        h5f.create_dataset('fast_axis_steps', data=self.nt)
        h5f.create_dataset('diode', data=self.dio)
        h5f.create_dataset('exptime', data=self.etime)
        h5f.create_dataset('omega', data=self.omega)
        h5f.create_dataset('translations', data=self.trans)      
        h5f.create_dataset('scantype', data=self.scantype)     
        h5f.close()
    	
        os.chdir(self.savepath)
        perm = 'chmod 777 %s' %fn
        os.system(perm)
        
    def removedata(self):

        if self.procunit == "MultiGPU":
            try:
                fn = "%s/%s*.azim.h5" % (self.savepath, self.dataset)
                cmd = 'rm %s' %fn
                os.system(cmd)        
        
                fn = "%s/%s*.json" % (self.savepath, self.dataset)
                cmd = 'rm %s' %fn
                os.system(cmd)
            except:
                print 'No .azim.h5 files present in the folder'
        
        else:
            try:
                fn = "%s/%s*linescan*.hdf5" % (self.savepath, self.dataset)
                cmd = 'rm %s' %fn
                os.system(cmd)        
        
                fn = "%s/%s*.json" % (self.savepath, self.dataset)
                cmd = 'rm %s' %fn
                os.system(cmd)
            except:
                print 'No *linescan*.hdf5 files present in the folder' 
                


        
class CalibrationThread(QtCore.QThread):
    def __init__(self,calibrant,energy):
        QtCore.QThread.__init__(self)
        self.calibrant = calibrant
        self.E = energy
        
    def run(self):
        cmd = 'pyFAI-calib -e %.f -c CeO2 -D Pilatus2M_CdTe %s' %(self.E,self.calibrant)
        os.system(cmd)
    
       
        
class CreatMaskThread(QtCore.QThread):
    def __init__(self,calibrant):
        QtCore.QThread.__init__(self)
        self.calibrant = calibrant
        
        mask_file = '/data/id15/inhouse2/inhouse03/gmask24.10.17.edf'
        fm=fabio.open(mask_file)
        fd=fabio.open(self.calibrant)
        dmask=binary_dilation(fm.data,structure=np.ones((1,1))).astype(int)
        outdata=np.where(np.array(dmask)==1,-1,np.array(fd.data))
        outbuffer=fabio.edfimage.edfimage(data=outdata,header=fd.header)
        self.newcalibrant=self.calibrant[0:-4]+'_masked.edf'
        outbuffer.write(self.newcalibrant)

        perm = 'chmod 777 %s' %self.newcalibrant
        os.system(perm) 
        
    def run(self):
        cmd = 'pyFAI-drawmask %s' %(self.newcalibrant)
        os.system(cmd)
        
        
class CreatPDFMaskThread(QtCore.QThread):
    ''' The methods of this class were written by Gavin Vaughan'''
    
    def __init__(self,ponifile,mask):
        QtCore.QThread.__init__(self)
        self.poniname = ponifile     
        self.mask = mask     
                
    def angle_mask(self,arr,tang,centre=None):
        nx,ny = arr.shape
        if centre is None:
        # use centre of the array
            x0=float(nx-1)/2.0
            y0=float(ny-1)/2.0
        else:
            x0,y0=centre
        if y0 > ny/2:
            y,x = np.ogrid[-x0:nx-x0,-y0:ny-y0]
        else:
            y,x = np.ogrid[-x0:nx-x0,y0-ny:y0]
            mask1 = y/x <= tang
            mask2 = x/y <= tang
            mask=np.logical_or(mask1,mask2)
        if y0 < ny/2:
            mask=np.fliplr(mask)
        
        self.amask = mask
    
    def radial_mask(self,arr,r,centre=None):
        # sum all the voxels with in a given radius
        # for now the centre is determined to be the centre of the array unless specified
        # print qarr.shape
        nx,ny = arr.shape
        if centre is None:
        # use centre of the array
            x0=float(nx-1)/2.0
            y0=float(ny-1)/2.0
        else:
            x0,y0=centre
            y,x = np.ogrid[-x0:nx-x0,-y0:ny-y0]
            mask = x*x + y*y >= r*r
        self.rmask = mask
        
    def block_mask(self,arr,tang,centre=None):
        nx,ny = arr.shape
        if centre is None:
        # use centre of the array
            x0=float(nx-1)/2.0
            y0=float(ny-1)/2.0
        else:
            x0,y0=centre
        if y0 > ny/2:
            y,x = np.ogrid[-x0:nx-x0,-y0:ny-y0]
            mask1 = x >= 0
            mask2 = y >= 0
        else:
            y,x = np.ogrid[-x0:nx-x0,-y0:ny-y0]
            mask1 = x <= 0
            mask2 = y >= 0
            mask = np.logical_or(mask1,mask2)
        self.bmask = mask
    
    def run(self):
        
        m=fabio.open(self.mask).data
        ai=pyFAI.load(self.poniname)
        alpha=float(75) #In degrees
        
        x0=ai.poni1/ai.pixel1
        y0=ai.poni2/ai.pixel2
        dy=1479-x0
        dx=1675-y0
        
        tang=np.tan((90.0-alpha)/2.0*np.pi/180.0)
        
        y=1479-dy-x0*tang
        y=dy+x0*tang
        x=1675-dx-y0*tang
        
        # so the points on the edge are
        # (x0,0), (0,y0)
        # which is shorter?
        
        d1x=(x0-x)
        d1y=(y0)
        d2y=(y0-y)
        d2x=(x0)
        
        l1=np.sqrt(d1x*d1x+d1y*d1y)
        l2=np.sqrt(d2x*d2x+d2y*d2y)
        
        # print x0,y0,x,y,tang
        
        # take off 5 pixels near the edge
        self.radial_mask(m,min(l1,l2)-5,(x0,y0))
        self.angle_mask(m,tang,(x0,y0))
        self.block_mask(m,tang,(x0,y0))
        	
        mask=np.logical_or.reduce((self.rmask,self.amask,self.bmask,m))
        #mask=np.logical_or.reduce((rmask,amask,m))
        outfile='PDFmask_%.1fdeg.cbf'%(alpha)
        outbuffer=fabio.cbfimage.cbfimage(data=mask)
        outbuffer.write(outfile)
        print 'Wrote mask to', outfile    
    
        perm = 'chmod 777 %s' %self.outfile
        os.system(perm) 
        
class CreatAzimintThread(QtCore.QThread):
    
    def __init__(self,ponifile,mask,npt_rad):
        QtCore.QThread.__init__(self)
        self.poniname = ponifile     
        self.mask = mask   
        self.npt_rad = npt_rad
        self.jsonname = '.azimint.json'

#        try:
#            self.calibrantpath = self.calibrant.split(".cbf")
#            self.calibrantpath = self.calibrantpath[0]
#            print self.calibrantpath
#        except:
#            self.calibrantpath = self.calibrant.split(".edf")
#            self.calibrantpath = self.calibrantpath[0]
#            print self.calibrantpath     
        
    def run(self):

        data = {}

        data["poni"] = self.poniname
        data['poni_file'] = self.poniname
        data['do_SA'] = 0
        data['method'] = "ocl_csr_nosplit_gpu"
        data['npt'] = self.npt_rad 
        data["chi_discontinuity_at_0"] = False
        data["do_mask"] = True 
        data["do_dark"] = False 
        data["do_azimuthal_range"] = False 
        data["do_flat"] = False 
        data["do_2D"] = False
        data["splineFile"] = "" 
        data["do_OpenCL"] = False
        data["pixel1"] = 0.000172 
        data["pixel2"] = 0.000172 
        data["polarization_factor"] = 0.97 
        data["do_solid_angle"] = False 
        data["do_radial_range"] = False 
        data["do_poisson"] = False 
        data["flat_field"] = "" 
        data["nbpt_rad"] = self.npt_rad 
        data["dark_current"] = "" 
        data["do_polarization"] = True 
        data["detector"] = "detector"
        data["unit"] = "2th_deg"
        data["do_dummy"] = False
        data["mask_file"] = self.mask        
        
        with open(self.poniname, 'r') as poni:
            for line in poni:
                if 'Wavelength' in line:
                    args=line.split()
                    Wavelength = float(args[1])
                if 'Distance' in line:
                    args=line.split()
                    Distance = float(args[1])
                if 'Poni1' in line:
                    args=line.split()
                    Poni1 = float(args[1])
                if 'Poni2' in line:
                    args=line.split()
                    Poni2 = float(args[1])
                if 'Rot1' in line:
                    args=line.split()
                    Rot1 = float(args[1])
                if 'Rot2' in line:
                    args=line.split()
                    Rot2 = float(args[1])
                if 'Rot3' in line:
                    args=line.split()
                    Rot3 = float(args[1])
                    
            print Wavelength
            print Distance
            print Poni1
            print Poni2
            print Rot1
            print Rot2
            print Rot3
            
        data["wavelength"] = Wavelength
        data["rot1"] = Rot1
        data["rot2"] = Rot2
        data["rot3"] = Rot3
        data["poni1"] = Poni1
        data["poni2"] = Poni2
        data["dist"] = Distance
            
        
        with open(self.jsonname, 'w') as f:
            json.dump(data, f)
        
        perm = 'chmod 777 %s' %self.jsonname
        os.system(perm)     
    
    
    
    
    
    