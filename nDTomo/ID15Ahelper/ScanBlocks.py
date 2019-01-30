# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:46:21 2017

@author: simon
"""


from baseBlocks import Block, EasternBlock
from PyQt5.QtGui import QColor, QPen, QFont
import settings
from PyQt5.QtCore import Qt, QRect, QSize, QPoint
#from PyQt5.QtGui import QPen
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
#        self.x = 100
        self.easternBlock = EasternBlock()
        self.easternBlock.y = self.y+(self.height-self.easternBlock.height)/2 # danger if this is zero
        #self.initiateParameters()
         
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
        percentd = '%.4d'
        s1 = 'fileName = sprintf(\"%s' % prefix
        s2 = '\"'
        for i in range(0, len(loopCounters)) :
            s1 = s1 + '_'+ percentd
            s2 = s2 + ', ' + loopCounters[i]
        s2 = s2 + ')\n'
        s = s1 + s2
        return s
             
##############################################################################       
class Simple_XRD_CT_Block(ScanBlock):
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'Simple XRD-CT'
        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        
    def initiateParameters(self):
        pathToData = 'C:\\virtualDataArea\\'
        self.parameters.setName(self.name)
        self.parameters.addChild(self.item('scanFileName', 'dummy'))
        self.parameters.addChild(self.item('pathToData', pathToData))
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
        self.scanText = self.parameter['scanFileName']
        
    def setMethodText(self):
        self.methodText = ""
        self.methodText += "# " + self.name + "\n"     
        self.methodText += "xrdct("
        c = self.parameters.children()
        for i in range(0, len(c)): # should replace this with enumerate
            if c[i].type() == 'str':
                self.methodText += "\"%s\"" % c[i].value()
            if c[i].type() == 'float':
                self.methodText += "%f" % c[i].value()
            if i<len(c)-1:
                self.methodText += ", "
#            elif c[i].isType('int'):
#                self.methodText += "umv %s %d \n" % (c[i].name(), c[i].value())
        self.methodText += ")"
        self.sideText = self.parameter['scanFileName']

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
        self.name = 'Single shot XRD'
        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFilePrefix', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='countTime (msecs)', value = 250, type = 'int'))
        self.parameters.addChild(dict(name='aerofreerun', value = 0, type = 'bool'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFilePrefix']
        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        if self.parameters['aerofreerun'] :
            self.methodText += "oneshot_norot fileName %d\n" % (self.parameters['countTime (msecs)'])
        else : 
            self.methodText += "oneshot fileName %d\n" % (self.parameters['countTime (msecs)'])
        self.time = self.parameters['countTime (msecs)']/1000.0
        self.sideText = self.humanTime()
##############################################################################
class XRDCT3D_Block(ScanBlock):
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = '3D XRD-CT'
        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='sampleSize (mm)', value = 5.00, type = 'float'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 2.00, type = 'float'))
        self.parameters.addChild(dict(name='exposureTime (msecs)', value = 20, type = 'int'))
        self.parameters.addChild(dict(name='resolution (microns)', value = 50, type = 'int'))
        self.parameters.addChild(dict(name='Z start (mm)', value = 0.0, type = 'float'))
        self.parameters.addChild(dict(name='Z end (mm)', value = 1.0, type = 'float'))
        self.parameters.addChild(dict(name='Z step (mm)', value = 0.05, type = 'float'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName']
        sample_size = self.parameters['sampleSize (mm)']
        scan_range = self.parameters['scanRange (mm)']
        exp_time = float(self.parameters['exposureTime (msecs)'])/1000.0
        y_resolution = float(self.parameters['resolution (microns)'])/1000.0
        z_min = self.parameters['Z start (mm)']
        z_max = self.parameters['Z end (mm)']
        z_step = self.parameters['Z step (mm)']
        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        # should really give a warning if Z start, step and end are not well defined
        self.methodText += "xrdct3d fileName %f %f %f %f %f %f %f\n" % (sample_size, scan_range, exp_time, y_resolution, z_min, z_max, z_step)
        Nz = (z_max-z_min)/z_step + 1
        self.time = (sample_size/y_resolution)*(scan_range/y_resolution) * exp_time * Nz
        self.sideText = "%d slices\n" % (Nz)
        self.sideText += self.humanTime()
##############################################################################
class XRDCT_Block(ScanBlock):
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'XRD-CT'
        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='sampleSize (mm)', value = 5.00, type = 'float'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 2.00, type = 'float'))
        self.parameters.addChild(dict(name='exposureTime (msecs)', value = 20, type = 'int'))
        self.parameters.addChild(dict(name='resolution (microns)', value = 50, type = 'int'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName']
        sample_size = self.parameters['sampleSize (mm)']
        scan_range = self.parameters['scanRange (mm)']
        exp_time = float(self.parameters['exposureTime (msecs)'])/1000.0
        y_resolution = float(self.parameters['resolution (microns)'])/1000.0
        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        self.methodText += "xrdct3d fileName %f %f %f %f\n" % (sample_size, scan_range, exp_time, y_resolution)
        self.time = (sample_size/y_resolution)*(scan_range/y_resolution) * exp_time
        self.sideText = self.humanTime()
##############################################################################