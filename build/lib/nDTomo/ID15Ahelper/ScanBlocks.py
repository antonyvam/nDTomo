# -*- coding: utf-8 -*-
"""

Scan blocks

@author: S.D.M. Jacques and A. Vamvakeros

"""


import settings
from baseBlocks import Block, EasternBlock
from PyQt5.QtGui import QColor, QPen, QFont, QImage
from PyQt5.QtCore import Qt, QRect, QSize, QPoint
#from PyQt5.QtGui import QPen

##############################################################################
class ScanBlock(Block):
    
    """
    
    Gerenic scan block class
    
    """
    
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
class OneShotBlock(ScanBlock):
    
    """
    
    XRD point scan block
    
    """
    
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
class HORSCAN_Block(ScanBlock):
    
    """
    
    XRD horizontal profile scan block
    
    """
    
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
class VERSCAN_Block(ScanBlock):
    
    """
    
    XRD vertical profile scan block
    
    """
    
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

    """
    
    XRD map scan block
    
    """
    
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
class InterlacedXRDCT_Block(ScanBlock):
    
    """
    
    Interlaced XRD-CT scan block
    
    """
    
    def __init__(self):
        ScanBlock.__init__(self) #####
#####        super(Interlaced_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'Interlaced XRD-CT'
        self.initiateParameters()
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='sampleSize (mm)', value = 5.00, type = 'float'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 4.00, type = 'float'))
        self.parameters.addChild(dict(name='resolution (mm)', value = 0.025, type = 'float'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.020, type = 'float'))

        
    def setMethodText(self):
        prefix = self.parameters['scanFileName']  + '_Intxrdct'
        sample_size = self.parameters['sampleSize (mm)']
        scan_range = self.parameters['scanRange (mm)']
        y_resolution = float(self.parameters['resolution (mm)'])
        exp_time = float(self.parameters['exposureTime (sec)'])
        loopCounters = self.getParentLoopCounters([])
        
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        self.methodText += "cmd = sprintf(\"xrdct3d %s %f %f %f %f\", fileName)\n" % (self.percents, sample_size, scan_range, y_resolution, exp_time)
        self.methodText += "eval(cmd)\n"
        self.time = (sample_size/y_resolution)*(scan_range/y_resolution) * exp_time
        self.sideText = self.humanTime()
        
##############################################################################

class XRDCT_Block(ScanBlock):
    
    """
    
    Zigzag 2D-XRD-CT scan block
    
    """     
    
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'Zigzag 2D-XRD-CT'
#        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        self.image = QImage('.//images//singleSlice.png')
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='sampleSize (mm)', value = 5.00, type = 'float'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 4.00, type = 'float'))
        self.parameters.addChild(dict(name='resolution (mm)', value = 0.025, type = 'float'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.020, type = 'float'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName']  + '_Intxrdct'
        sample_size = self.parameters['sampleSize (mm)']
        scan_range = self.parameters['scanRange (mm)']
        y_resolution = float(self.parameters['resolution (mm)'])
        exp_time = float(self.parameters['exposureTime (sec)'])
        loopCounters = self.getParentLoopCounters([])
        
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        self.methodText += "cmd = sprintf(\"xrdct3d %s %f %f %f %f\", fileName)\n" % (self.percents, sample_size, scan_range, y_resolution, exp_time)
        self.methodText += "eval(cmd)\n"
        self.time = (sample_size/y_resolution)*(scan_range/y_resolution) * exp_time
        self.sideText = self.humanTime()
        

##############################################################################
class XRDCT3D_Block(ScanBlock):
    
    """
    
    Zigzag 3D-XRD-CT scan block
    
    """     
    
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'Zigzag 3D-XRD-CT'
#        self.regularColor = QColor(0, 0, 250)
        self.initiateParameters()
        self.image = QImage('.//images//stack.png')
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='sampleSize (mm)', value = 5.00, type = 'float'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 4.00, type = 'float'))
        self.parameters.addChild(dict(name='resolution (mm)', value = 0.025, type = 'float'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.020, type = 'float'))
        self.parameters.addChild(dict(name='Z start (mm)', value = 0.0, type = 'float'))
        self.parameters.addChild(dict(name='Z end (mm)', value = 1.0, type = 'float'))
        self.parameters.addChild(dict(name='Z step (mm)', value = 0.05, type = 'float'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName'] + '_zigzag3Dxrdct'
        sample_size = self.parameters['sampleSize (mm)']
        scan_range = self.parameters['scanRange (mm)']
        exp_time = float(self.parameters['exposureTime (sec)'])
        y_resolution = float(self.parameters['resolution (microns)'])

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
    
    """
    
   Continuous rotation 2D-XRD-CT scan block
    
    """ 
    
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'ContRot XRD-CT'
        self.initiateParameters()
        self.image = QImage('.//images//singleSlice.png') ###### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! To sort
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='sampleSize (mm)', value = 5.00, type = 'float'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 4.00, type = 'float'))
        self.parameters.addChild(dict(name='resolution (mm)', value = 0.025, type = 'float'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.020, type = 'float'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName']  + '_ContRotxrdct'
        sample_size = self.parameters['sampleSize (mm)']
        scan_range = self.parameters['scanRange (mm)']
        exp_time = float(self.parameters['exposureTime (sec)'])
        y_resolution = float(self.parameters['resolution (microns)'])
        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        self.methodText += "cmd = sprintf(\"continuous_rot_xrdct3d %s %f %f %f %f\", fileName)\n" % (self.percents, sample_size, scan_range, y_resolution, exp_time)
        self.methodText += "eval(cmd)\n"
        self.time = (sample_size/y_resolution)*(scan_range/y_resolution) * exp_time
        self.sideText = self.humanTime()
        

##############################################################################
class FASTXRDCT3D_Block(ScanBlock):
    
    """
    
   Continuous rotation 3D-XRD-CT scan block
    
    """    
    
    def __init__(self): 
        ScanBlock.__init__(self) #####
#####        super(Simple_XRD_CT_Block, self).__init__()
#        super().__init__()
        self.name = 'ContRot 3D-XRD-CT'
        self.initiateParameters()
        self.image = QImage('.//images//stack.png') ###### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! To sort
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='sampleSize (mm)', value = 5.00, type = 'float'))
        self.parameters.addChild(dict(name='scanRange (mm)', value = 4.00, type = 'float'))
        self.parameters.addChild(dict(name='resolution (mm)', value = 0.025, type = 'float'))
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.020, type = 'float'))
        self.parameters.addChild(dict(name='Z start (mm)', value = 0.0, type = 'float'))
        self.parameters.addChild(dict(name='Z end (mm)', value = 1.0, type = 'float'))
        self.parameters.addChild(dict(name='Z step (mm)', value = 0.05, type = 'float'))
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName'] + '_ContRot3Dxrdct'
        sample_size = self.parameters['sampleSize (mm)']
        scan_range = self.parameters['scanRange (mm)']
        exp_time = float(self.parameters['exposureTime (sec)'])
        y_resolution = float(self.parameters['resolution (microns)'])

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
    
    """
    
    Absorption-contrast CT scan block
    
    """
    
    def __init__(self): 
        ScanBlock.__init__(self) #####
        self.name = 'ABS-CT'
        self.initiateParameters()
        self.image = QImage('.//images//stack.png')
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='number projections', value = 1500, type = 'float'))        
        self.parameters.addChild(dict(name='exposureTime (sec)', value = 0.100, type = 'float'))
        self.parameters.addChild(dict(name='rotation centre pos', value = -100, type = 'float'))   
        self.parameters.addChild(dict(name='flat pos', value = -115, type = 'float'))    
        self.parameters.addChild(dict(name='0-360 deg tomo', value = 0.0, type = 'float'))    
        
    def setMethodText(self):
        prefix = self.parameters['scanFileName'] + '_absct'
        nproj = round(self.parameters['number of projections'])
        exp_time = float(self.parameters['exposureTime (sec)'])
        rotcen = float(self.parameters['rotation centre pos'])
        flatpos = float(self.parameters['flat pos'])
        rot_axis_at_sensor_edge = float(self.parameters['0-360 deg tomo'])
        
        loopCounters = self.getParentLoopCounters([])
        self.methodText = self.genSprintfFromLoopCounters(prefix, loopCounters)
        # should really give a warning if Z start, step and end are not well defined
        self.methodText += "cmd = sprintf(\"absct %s %f %f %f %f %f\", fileName)\n" % (self.percents, exp_time, nproj, rotcen, flatpos, rot_axis_at_sensor_edge)
        self.methodText += "eval(cmd)\n"
        
#        self.time = sample_size * exp_time
        self.time = nproj * exp_time
        self.sideText += self.humanTime()
        
##############################################################################
