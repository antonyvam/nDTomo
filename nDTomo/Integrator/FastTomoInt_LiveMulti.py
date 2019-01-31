# -*- coding: utf-8 -*-
"""
Class to integrate XRD-CT data collected with the continuous rotation-translation method.
The integration is performed with the multi-GPU ID15A PC.
The type of integration process is intended to be couple with live visualization.

@author: A. Vamvakeros
"""

from PyQt5.QtCore import pyqtSignal, QThread
import os, fabio, h5py, time, json
from numpy import zeros, array, arange, mod, ceil, deg2rad, sin, pi
from PThread import Periodic

class Fast_XRDCT_ID15ASqueeze(QThread): 
    
    progress = pyqtSignal(int)

    def __init__(self,prefix,dataset,xrdctpath,maskname,poniname,na,nt,npt_rad,filt,procunit,units,prc,thres,datatype,savepath,scantype,energy,jsonname,omega,trans,dio,etime):
        QThread.__init__(self)
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
            self.mask = array(self.mask.data)
            
#            self.sinos = np.zeros((self.nt*self.na,2000));

#            if self.filt == "No":
##	            self.sinos = np.zeros((self.nt*self.na,2000));
#                self.sinos = zeros((self.nt*self.na,self.npt_rad));
#            else:
#                self.sinos = zeros((self.nt*self.na,self.npt_rad-10));
            self.sinos = zeros((self.nt*self.na,self.npt_rad-10));
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
        v = round((100*self.nextLine/self.nt))
        self.progress.emit(v)
            
    def tth2q(self):
        self.h = 6.620700406E-34;self.c = 3E8
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1E3)
        self.d = self.wavel/(2*sin(deg2rad(0.5*self.tth)))
        self.q = pi*2/self.d
            
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
                data = f['/entry_0000/PyFAI/process_integrate1d/I'][:,0:self.npt_rad-10]
                self.tth = f['/entry_0000/PyFAI/process_integrate1d/2th'][:,0:self.npt_rad-10]
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
                data['percentile'] = (round(float(self.prc)/2),100-round(float(self.prc)/2))         
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