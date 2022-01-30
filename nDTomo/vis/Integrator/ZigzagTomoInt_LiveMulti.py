# -*- coding: utf-8 -*-
"""
Class to integrate XRD-CT data collected with the zigzag method.
The integration is performed with the multi-GPU ID15A PC.
This type of integration process is intended to be couple with live visualization.

@author: A. Vamvakeros
"""

from PyQt5.QtCore import pyqtSignal, QThread
import os, fabio, h5py, time, json
from numpy import zeros, array, arange, mod, ceil, deg2rad, sin, pi
from nDTomo.vis.Integrator.PThread import Periodic

class XRDCT_ID15ASqueeze(QThread): 
    
    """
    
    Integrate continuous rotation-translation XRD-CT data
	
        :prefix: prefix used for the filenames of the experimental data
    
	:dataset: foldername where the experimental data are stored
    
	:xrdctpath: full path where the experimental data are stored
    
	:maskname: detector mask file ('mask.edf')
    
	:poniname: .poni file ('filename.poni')
    
	:na: number of angular steps during the tomographic scan
    
	:nt: number of translation steps during the tomographic scan
    
	:npt_rad: number of bins used for the diffraction pattern (e.g. 2048)
    
	:procunit: processing unit, options are: 'CPU', 'GPU' and 'MultiGPU'
    
	:units:  x axis units for the integrated diffraction patterns, options are: 'q_A^-1' or '2th_deg'
    
	:prc: percentage to be used for the trimmed mean filter
    
	:thres: number of standard deviation values to be used for the adaptive standard deviation filter
    
	:datatype: type of image files, options are: 'edf' or 'cbf'
    
	:savepath: full path where results will be saved
    
	:scantype: scantype, options are: 'Zigzag', 'ContRot' and 'Interlaced'
    
	:energy: X-ray energy in keV		
    
	:jsonname: full path for the .azimint.json file
    
	:omega:	rotation axis motor positions during the tomographic scan (1d array)
    
	:trans:	translation axis motor positions during the tomographic scan (1d array)
    
	:dio: diode values per point during the tomographic scan (1d array)
    
	:etime:	values for exposure time per point during the tomographic scan (1d array)
    
    """     
    
    progress = pyqtSignal(int)

    def __init__(self,prefix,dataset,xrdctpath,maskname,poniname,na,nt,npt_rad,filt,procunit,units,prc,thres,asym,datatype,savepath,scantype,energy,jsonname,omega,trans,dio,etime):
        QThread.__init__(self)
        self.prefix = prefix; self.dataset=dataset; self.xrdctpath = xrdctpath; self.maskname = maskname; self.poniname = poniname
        self.filt=filt; self.procunit=procunit;self.units =units;self.E = energy
        self.prc=prc;self.thres = thres;self.asym=asym;self.datatype = datatype;self.savepath = savepath
        self.na = int(na); self.nt = int(nt); self.npt_rad = int(npt_rad); self.scantype = scantype; self.jsonname = jsonname
        self.omega = omega; self.trans = trans;self.dio = dio; self.etime = etime
        self.gpuxrdctpath = '/gz%s' %self.xrdctpath
        
                
        dpath = os.path.join( self.xrdctpath, self.dataset)
        try:
            os.chdir(dpath)		

            self.mask = fabio.open(self.maskname)
            self.mask = array(self.mask.data)
            
#            if self.filt == "No":
##	            self.sinos = np.zeros((self.nt*self.na,2000));
#                self.sinos = np.zeros((self.nt*self.na,self.npt_rad));
#            else:
#                self.sinos = np.zeros((self.nt*self.na,self.npt_rad-10));
                
            self.sinos = zeros((self.nt*self.na,self.npt_rad-10));
            print(self.sinos.shape)
            
        except:
            print("Cannot open mask file or the xrd-ct dataset directory is wrong")
            

    def run(self):
        
        """
		Initiate the XRD-CT data integration process
		"""
        
        self.previousLine = 0
        self.nextLine = 1        
        self.nextimageFile  =  "%s/%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nt)     
        print(self.nextimageFile)
          
        self.periodEventraw = Periodic(1, self.checkForImageFile)

    def checkForImageFile(self):
        
        """
		Look if data have been generated
		"""           
        
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

        """
		Perform the diffraction data integration using the MultiGPU method
		""" 
        
        self.json = "%s%s_%.4d.json" % (self.savepath, self.dataset, self.nextLine)
        print(self.json)
	
        self.creategpujson() # write the .json files
        self.gpuproc() # integrate the 2D diffraction patterns
            
    def setnextimageFile(self):
        
        """
		Set the next target image file
		"""  	
        
        self.nextLine += 1
        self.previoush5File =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLine-1)
        if self.nextLine < self.na:
            self.nextimageFile =  "%s%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.nt)
            print(self.nextimageFile)
            self.periodEventraw.start()
        elif self.nextLine == self.na: #might need a sleep to give time to write the last image
            self.nextimageFile =  "%s%s/%s_%.4d.cbf" % (self.xrdctpath, self.dataset, self.dataset, self.nextLine*self.nt-1)
            print(self.nextimageFile)
            self.periodEventraw.start()            
        else:
            print('Integration done')

        v = round((100*self.nextLine/self.na))
        self.progress.emit(v)
            
    def tth2q(self):
        
        """
		Convert 2theta to d and q spacing
		"""  	
        
        self.h = 6.620700406E-34;self.c = 3E8
        self.wavel = 1E10*6.242E18*self.h*self.c/(self.E*1E3)
        self.d = self.wavel/(2*sin(deg2rad(0.5*self.tth)))
        self.q = pi*2/self.d
            
    def liveRead(self):
        
        """
		Initiate the live read of the integrated diffraction data
		"""  
        
        self.nextLineNumber = 1
        
        self.nextFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)
        self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber+1)
        self.periodEvent = Periodic(1, self.checkForFile)
        
    def checkForFile(self):
        
        """
		Look for integrated data file 
		"""  
        
        if os.path.exists(self.nextFile) & os.path.exists(self.targetFile):
            print('%s exists' % self.nextFile)
            self.periodEvent.stop()
            self.hdfLineScanRead()
            self.setNextFile()
            self.setTargetFile()
        else: # keep looking
            print('%s or %s does not exist' %(self.nextFile,self.targetFile))

    def hdfLineScanRead(self):
        
        """
		Read integrated data 
		""" 
        
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
            elif self.filt == "assymetric":
                data = f['/entry_0000/PyFAI/process_medfilt1d/I'][:,0:self.npt_rad-10]     
                self.tth = f['/entry_0000/PyFAI/process_medfilt1d/2th'][0:self.npt_rad-10]
        f.close
        
        filei = self.nt*(self.nextLineNumber-1)				
        filef = self.nt*self.nextLineNumber
        self.sinos[range(filei, filef),:] = data
        
    def setNextFile(self):
        
        """
		Look for the target h5 file. If all done, save the integrated data as a sinongram volume.
		"""          
        
        self.nextLineNumber += 1
        
        if self.nextLineNumber < self.na+1:
            self.nextFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)

            print(self.nextFile)
            self.periodEvent.start()
        else:
            print('All done')
            self.tth2q()
            self.writesinos()            
            print("Saving data done")  
            self.removedata()            
            print("Deleted integrated linescans")     

    def setTargetFile(self):

        """
		Set the next target h5 file
		""" 
        
        if self.nextLineNumber < self.na:
            self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber+1)
        elif self.nextLineNumber == self.na:
            self.targetFile =  "%s/%s_%.4d.azim.h5" % (self.savepath, self.dataset, self.nextLineNumber)

    def writesinos(self):

        """
		Export the sinogram data volume as a single .hdf5 file
		"""   
        
        if self.filt == "No":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "Median":
            fn = "%s/%s_integrated_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, self.filt,self.procunit)
        elif self.filt == "trimmed_mean":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.prc), self.filt,self.procunit)
        elif self.filt == "sigma":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.thres), self.filt,self.procunit)                
        elif self.filt == "assymetric":
            fn = "%s/%s_integrated_%.1f_%s_Filter_%s.hdf5" % (self.savepath, self.dataset, float(self.asym), self.filt,self.procunit)
        
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
        
        """
		Remove integrated linescan data
		"""
        
        fn = "%s/%s*.h5" % (self.savepath, self.dataset)
        cmd = 'rm %s' %fn
        os.system(cmd)        

        fn = "%s/%s*.json" % (self.savepath, self.dataset)
        cmd = 'rm %s' %fn
        os.system(cmd)   
        
    def creategpujson(self):
        
        """
		Create the .json file for the diffraction data integration using the MultiGPU method
		""" 
        
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
            elif self.filt == "assymetric":
                data['plugin_name'] = 'id15v2.IntegrateManyFrames'
                data['integration_method'] = "medfilt1d"
                data['percentile'] = (0,100-round(float(self.asym)))         
            
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
            print(data['output_file'])
        
        self.jsonfile = '%s_%.4d.json' % (da, self.nextLine)
        job = os.path.join( self.savepath, self.jsonfile)
        with open(job, 'w') as outfile:  
            json.dump(data, outfile)

        os.chdir(self.savepath)
        
        perm = 'chmod 777 %s' %self.jsonfile
        os.system(perm) 
        
        print(time.time()-start)
        
    def gpuproc(self):
        
        """
		Use the dahu-reprocess method to perform the diffraction data integration using the MultiGPU method
		"""  
        
        dahu = 'dahu-reprocess %s &' %self.jsonfile #### try with &
        print('processing dataset %s' %self.jsonfile)
        os.system(dahu)