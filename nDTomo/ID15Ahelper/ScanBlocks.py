# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:46:21 2017

@author: simon
"""


from baseBlocks import Block, EasternBlock
from PyQt5.QtGui import QColor, QPen, QFont, QImage
import settings
from PyQt5.QtCore import Qt, QRect, QSize, QPoint
#from PyQt5.QtGui import QPen
import numpy as np
##############################################################################
class ScanBlock(Block):
    def __init__(self):
        Block.__init__(self) #####
#####        super(ScanBlock, self).__init__()
#        super().__init__()
        self.Type = 'ScanBlock'
        self.name = 'generic scan block'
        self.scanText = ''
        self.addNode('N')
        self.addNode('E')
        self.addNode('S')
        self.regularColor = QColor(50, 150, 50)
        #        self.x = 100
        self.easternBlock = EasternBlock()
        self.easternBlock.y = self.y+(self.height-self.easternBlock.height)/2 # danger if this is zero
        #self.initiateParameters()
        self.percents = '%s'
        self.percentd = '%.4d'
         
    def preDrawActions(self, qp):
        self.easternBlock.color = self.color # a bit rubbish that this line needs to be here
        self.easternBlock.height = self.height*1
        self.easternBlock.x = self.x + 250
        self.easternBlock.y = self.y+(self.height-self.easternBlock.height)/2 # danger if this is zero
        self.easternBlock.alpha = self.alpha
#        self.easternBlock.drawBlock(qp)
        
    def postDrawActions(self, qp):
        pass
       # pen = QPen(QColor(100,0,0), 1,2)
       # qp.setPen(pen)
       # qp.drawLine(self.nodes[1].boundingBox.center(), self.easternBlock.nodes[0].boundingBox.center())
#

    def item(self, name, value='', values=None, **kwargs):
        #Add an item to a parameter tree.
        if 'type' not in kwargs:
            if values:
                kwargs['type'] = 'list'
            elif isinstance(value, bool):
                kwargs['type'] = 'bool'
            elif isinstance(value, int):
                kwargs['type'] = 'int'
            elif isinstance(value, float):
                kwargs['type'] = 'float'
            else:
                kwargs['type'] = 'str'
        return dict(name=name, value=value, values=values, **kwargs) 

    def initiateParameters(self):
        self.parameters.setName(self.name)
                
    def genSprintfFromLoopCounters(self, prefix, loopCounters):
        if len(loopCounters) == 0 :
            s = 'fileName = \"%s\"\n' % prefix
            return s
        loopCounters = loopCounters[::-1]
        s1 = 'fileName = sprintf(\"%s' % prefix
        s2 = '\"'
        for i in range(0, len(loopCounters)) :
            s1 = s1 + '_'+ self.percentd
            s2 = s2 + ', ' + loopCounters[i]
        s2 = s2 + ')\n'
        s = s1 + s2
        return s
             
##############################################################################        
class Interlaced_XRD_CT_Block(ScanBlock):
    def __init__(self):
        ScanBlock.__init__(self) #####
#####        super(Interlaced_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'interlaced XRD-CT'
        self.initiateParameters()
        
    def initiateParameters(self):
        pathToData = 'C:\\Users\\simon\\Dropbox (Finden)\\Projects\\ID15_XRDCT_development\\scripts\\simon\\virtualDataArea\\'
        self.parameters.setName(self.name)
        self.parameters.addChild(self.item('scanFileName', 'dummy'))
        self.parameters.addChild(self.item('pathToData', pathToData))
        self.parameters.addChild(self.item('nSubTomos', 4))
        self.parameters.addChild(self.item('slowAxis', 2, type='list', values = settings.userMotorDict))
        self.parameters.addChild(self.item('slowAxisStart', 0.00))
        self.parameters.addChild(self.item('slowAxisStep', 2.00))
        self.parameters.addChild(self.item('slowAxisN', 180))
        self.parameters.addChild(self.item('fastAxis', 2, type='list', values = settings.userMotorDict))      
        self.parameters.addChild(self.item('fastAxisStart', -5.00))
        self.parameters.addChild(self.item('fastAxisStep', 0.10))
        self.parameters.addChild(self.item('fastAxisN', 100))
        self.parameters.addChild(self.item('zigzag', 0, type='bool'))
        self.parameters.addChild(self.item('acquisition', 0.02))
    
    def setMethodText(self):
        self.methodText = "pass"

##############################################################################            
class OneShotBlock(ScanBlock):
    def __init__(self):
        ScanBlock.__init__(self) #####
#####        super(OneShotBlock, self).__init__()
#        super().__init__()
        self.name = 'Point XRD'
#        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        self.image = QImage('.//images//singlePoint.png')
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFilePrefix', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='countTime (sec)', value = 0.250, type = 'float'))
        self.parameters.addChild(dict(name='aerofreerun', value = 0, type = 'bool'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFilePrefix']
        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        if self.parameters['aerofreerun'] :
            self.methodText += "cmd = sprintf(\"xrd_point %s %d\", fileName)\n" % (self.percents, 1E3*self.parameters['countTime (sec)'])
        else : 
            self.methodText += "cmd = sprintf(\"xrd_point_norot %s %d\", fileName)\n" % (self.percents, 1E3*self.parameters['countTime (sec)'])
        self.methodText += "eval(cmd)\n"
        self.time = self.parameters['countTime (sec)']
        self.sideText = self.humanTime()
        
##############################################################################
class AEROYSCAN_Block(ScanBlock):
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'hor XRD linescan'
#        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        self.image = QImage('.//images//horizontalLineScan.png')
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 2.00, type = 'float'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.020, type = 'float'))
        self.parameters.addChild(dict(name='resolution (microns)', value = 50, type = 'int'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName'] + '_aeroyScan'
        scan_range = self.parameters['scanRange (mm)']
        exp_time = float(self.parameters['exposureTime (sec)'])
        y_resolution = float(self.parameters['resolution (microns)'])/1000.0

        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
#        self.methodText += "xrdct3d fileName %f %f %f %f\n" % (sample_size, scan_range, exp_time, y_resolution)
        self.methodText += "cmd = sprintf(\"xrdmap %s %f %f %f %f\", fileName)\n" % (self.percents, 0, scan_range, y_resolution, exp_time)
        self.methodText += "eval(cmd)\n"
        self.time = (scan_range/y_resolution) * exp_time
        self.sideText = self.humanTime()
        
##############################################################################
class ZSCAN_Block(ScanBlock):
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'ver XRD linescan'
#        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        self.image = QImage('.//images//verticalLineScan.png')
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.020, type = 'float'))
        self.parameters.addChild(dict(name='Z start (mm)', value = 0.0, type = 'float'))
        self.parameters.addChild(dict(name='Z end (mm)', value = 1.0, type = 'float'))
        self.parameters.addChild(dict(name='Z step (mm)', value = 0.05, type = 'float'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName'] + '_zscan'
        exp_time = float(self.parameters['exposureTime (sec)'])*1E3
        z_min = self.parameters['Z start (mm)']
        z_max = self.parameters['Z end (mm)']
        z_step = self.parameters['Z step (mm)']

        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        self.methodText += "cmd = sprintf(\"xrd_zscan %s %f %f %f %f\", fileName)\n" % (self.percents, exp_time, z_min, z_max, z_step)
        self.methodText += "eval(cmd)\n"
        self.time = ((z_max-z_min)/z_step+1) * exp_time
        self.sideText = self.humanTime()
        
##############################################################################
class XRDMAP_Block(ScanBlock):
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'XRD-MAP'
#        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        self.image = QImage('.//images//grid.png')
        

    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 2.00, type = 'float'))
        self.parameters.addChild(dict(name='resolution (microns)', value = 50, type = 'int'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.020, type = 'float'))
        self.parameters.addChild(dict(name='Z start (mm)', value = 0.0, type = 'float'))
        self.parameters.addChild(dict(name='Z end (mm)', value = 1.0, type = 'float'))
        self.parameters.addChild(dict(name='Z step (mm)', value = 0.05, type = 'float'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName'] + '_map'
        scan_range = self.parameters['scanRange (mm)']
        exp_time = float(self.parameters['exposureTime (sec)'])
        y_resolution = float(self.parameters['resolution (microns)'])/1000.0
        
        z_min = self.parameters['Z start (mm)']
        z_max = self.parameters['Z end (mm)']
#        if self.parameters['Z end (mm)'] < self.parameters['Z start (mm)'] and self.parameters['Z step (mm)'] >= 0:
#            self.parameters['Z step (mm)'] = -self.parameters['Z step (mm)']
        z_step = self.parameters['Z step (mm)']
        
        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        self.methodText += "cmd = sprintf(\"xrdmap %s %f %f %f %f %f %f\", fileName)\n" % (self.percents, scan_range, y_resolution, exp_time, z_min, z_max, z_step)
        self.methodText += "eval(cmd)\n"
        self.time = 0#(np.mod(z_max-z_min)/z_step+1)*(scan_range/y_resolution) * exp_time
        self.sideText = self.humanTime()
        
##############################################################################
class XRDCT_Block(ScanBlock):
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'XRD-CT'
#        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        self.image = QImage('.//images//singleSlice.png')
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='sampleSize (mm)', value = 5.00, type = 'float'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 2.00, type = 'float'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.020, type = 'float'))
        self.parameters.addChild(dict(name='resolution (microns)', value = 50, type = 'int'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName']  + '_xrdct'
        sample_size = self.parameters['sampleSize (mm)']
        scan_range = self.parameters['scanRange (mm)']
        exp_time = float(self.parameters['exposureTime (sec)'])
        y_resolution = float(self.parameters['resolution (microns)'])/1000.0
        loopCounters = self.getParentLoopCounters([])
        
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        self.methodText += "cmd = sprintf(\"xrdct3d %s %f %f %f %f\", fileName)\n" % (self.percents, sample_size, scan_range, y_resolution, exp_time)
        self.methodText += "eval(cmd)\n"
        self.time = (sample_size/y_resolution)*(scan_range/y_resolution) * exp_time
        self.sideText = self.humanTime()
        

##############################################################################
class XRDCT3D_Block(ScanBlock):
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = '3D XRD-CT'
#        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        self.image = QImage('.//images//stack.png')
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='sampleSize (mm)', value = 5.00, type = 'float'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 2.00, type = 'float'))
        self.parameters.addChild(dict(name='resolution (microns)', value = 50, type = 'int'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.020, type = 'float'))
        self.parameters.addChild(dict(name='Z start (mm)', value = 0.0, type = 'float'))
        self.parameters.addChild(dict(name='Z end (mm)', value = 1.0, type = 'float'))
        self.parameters.addChild(dict(name='Z step (mm)', value = 0.05, type = 'float'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName'] + '_3Dxrdct'
        sample_size = self.parameters['sampleSize (mm)']
        scan_range = self.parameters['scanRange (mm)']
        exp_time = float(self.parameters['exposureTime (sec)'])
        y_resolution = float(self.parameters['resolution (microns)'])/1000.0

        z_min = self.parameters['Z start (mm)']
        z_max = self.parameters['Z end (mm)']
#        if self.parameters['Z end (mm)'] < self.parameters['Z start (mm)'] and self.parameters['Z step (mm)'] >= 0:
#            self.parameters['Z step (mm)'] = -self.parameters['Z step (mm)']
        z_step = self.parameters['Z step (mm)']

        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        # should really give a warning if Z start, step and end are not well defined
        self.methodText += "cmd = sprintf(\"xrdct3d %s %f %f %f %f %f %f %f\", fileName)\n" % (self.percents, sample_size, scan_range, y_resolution, exp_time,  z_min, z_max, z_step)
        self.methodText += "eval(cmd)\n"
        Nz = (z_max-z_min)/z_step + 1
        self.time = 0#(sample_size/y_resolution)*(scan_range/y_resolution) * exp_time * Nz
        self.sideText = "%d slices\n" % (Nz)
        self.sideText += self.humanTime()


##############################################################################
class FASTXRDCT_Block(ScanBlock):
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'FAST XRD-CT'
#        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        self.image = QImage('.//images//singleSlice.png') ###### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! To sort
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='sampleSize (mm)', value = 5.00, type = 'float'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 2.00, type = 'float'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.020, type = 'float'))
        self.parameters.addChild(dict(name='resolution (microns)', value = 50, type = 'int'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName']  + '_fastxrdct'
        sample_size = self.parameters['sampleSize (mm)']
        scan_range = self.parameters['scanRange (mm)']
        exp_time = float(self.parameters['exposureTime (sec)'])
        y_resolution = float(self.parameters['resolution (microns)'])/1000.0
        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        self.methodText += "cmd = sprintf(\"continuous_rot_xrdct3d %s %f %f %f %f\", fileName)\n" % (self.percents, sample_size, scan_range, y_resolution, exp_time)
        self.methodText += "eval(cmd)\n"
        self.time = (sample_size/y_resolution)*(scan_range/y_resolution) * exp_time
        self.sideText = self.humanTime()
        

##############################################################################
class FASTXRDCT3D_Block(ScanBlock):
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'Fast 3D XRD-CT'
#        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        self.image = QImage('.//images//stack.png') ###### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! To sort
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='sampleSize (mm)', value = 5.00, type = 'float'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 2.00, type = 'float'))
        self.parameters.addChild(dict(name='resolution (microns)', value = 50, type = 'int'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.020, type = 'float'))
        self.parameters.addChild(dict(name='Z start (mm)', value = 0.0, type = 'float'))
        self.parameters.addChild(dict(name='Z end (mm)', value = 1.0, type = 'float'))
        self.parameters.addChild(dict(name='Z step (mm)', value = 0.05, type = 'float'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName'] + '_fast3Dxrdct'
        sample_size = self.parameters['sampleSize (mm)']
        scan_range = self.parameters['scanRange (mm)']
        exp_time = float(self.parameters['exposureTime (sec)'])
        y_resolution = float(self.parameters['resolution (microns)'])/1000.0

        z_min = self.parameters['Z start (mm)']
        z_max = self.parameters['Z end (mm)']
#        if self.parameters['Z end (mm)'] < self.parameters['Z start (mm)'] and self.parameters['Z step (mm)'] >= 0:
#            self.parameters['Z step (mm)'] = -self.parameters['Z step (mm)']
        z_step = self.parameters['Z step (mm)']

        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        # should really give a warning if Z start, step and end are not well defined
        self.methodText += "cmd = sprintf(\"continuous_rot_xrdct3d %s %f %f %f %f %f %f %f\", fileName)\n" % (self.percents, sample_size, scan_range, y_resolution, exp_time, z_min, z_max, z_step)
        self.methodText += "eval(cmd)\n"
        Nz = (z_max-z_min)/z_step + 1
        self.time = 0#(sample_size/y_resolution)*(scan_range/y_resolution) * exp_time * Nz
        self.sideText = "%d slices\n" % (Nz)
        self.sideText += self.humanTime()


##############################################################################
class ABSCT_Block(ScanBlock):
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'ABS-CT'
#        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        self.image = QImage('.//images//stack.png')
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='sampleSize (mm)', value = 5.00, type = 'float'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.100, type = 'float'))
        self.parameters.addChild(dict(name='rotation centre pos', value = -100, type = 'float'))   
        self.parameters.addChild(dict(name='flat pos', value = -115, type = 'float'))    
        self.parameters.addChild(dict(name='rot_axis_at_sensor_edge', value = 0.0, type = 'float'))    
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName'] + '_absct'
        pxsz = 0.0058
        sample_size = round(self.parameters['sampleSize (mm)']/pxsz)
        exp_time = float(self.parameters['exposureTime (sec)'])
        rotcen = float(self.parameters['rotation centre pos'])
        flatpos = float(self.parameters['flat pos'])
        rot_axis_at_sensor_edge = float(self.parameters['rot_axis_at_sensor_edge'])
        
        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        # should really give a warning if Z start, step and end are not well defined
        self.methodText += "cmd = sprintf(\"absct %s %f %f %f %f %f\", fileName)\n" % (self.percents, exp_time, sample_size, rotcen, flatpos, rot_axis_at_sensor_edge)
        self.methodText += "eval(cmd)\n"
        
        self.time = sample_size * exp_time
        self.sideText += self.humanTime()
        
##############################################################################
