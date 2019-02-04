# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:46:21 2017

@author: simon
"""


from baseBlocks import Block, BlockGroup
from MidLevelBlocks import ScanBlock, SimpleFunctionCallBlock
from PyQt5.QtGui import QColor
from pyqtgraph.parametertree import Parameter
import copy
import settings

class SingleEurothermBlock(Block):
    def __init__(self):
        Block.__init__(self) #####
#####        super(SingleEurothermBlock, self).__init__()
#        super().__init__()
        self.addNode('N')
        self.addNode('E')
        self.addNode('S')
        self.regularColor = QColor(50, 0, 0)
        self.color = self.regularColor
        self.name = 'Single Eurotherm'
        self.initiateParameters()
        
    def setMethodText(self):
        # this bit is wrong : list(self.motorNames.keys())[self.parameters['name']]
        motor = settings.euroMotorList[self.parameters['name']]
        self.methodText = "euro2400parameters unit=%d RampRate=%d ; umv %s %d\n" % (self.parameters['unit'], self.parameters['rampRate'], motor, self.parameters['targetTemp'] )

    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='unit', value=3, type = 'int'))
        self.parameters.addChild(dict(name='name', value=1, type = 'list', values = settings.euroMotorDict))
        self.parameters.addChild(dict(name='rampRate', value=20, type = 'int'))
        self.parameters.addChild(dict(name='targetTemp', value=25, type = 'int'))

class DualEurothermBlock(Block):
    pass
             
class RelMotorMoveBlock(Block):
    def __init__(self):
        Block.__init__(self) #####
#####        super(RelMotorMoveBlock, self).__init__()
#        super().__init__()
        self.addNode('N')
        self.addNode('E')
        self.addNode('S')
        value = 0
        self.regularColor = QColor(210, 150, 0)
        self.color = self.regularColor
        self.parameters.setName('rel motor move')
        self.name = ''
        self.value = value
        self.initiateParameters(0, value)
        
    def setMethodText(self):
        motor = settings.userMotorList[self.parameters['motor']]
        self.methodText = "umvr %s %f" % (motor, self.parameters['incrementValue'])
           
    def preDrawActions(self, qp):
        motor = settings.userMotorList[self.parameters['motor']]
        self.name = "umvr %s %2.4f" % (motor, self.parameters['incrementValue'])
        
    def initiateParameters(self, initialName, initialValue):
            # WARNING WOULD NEED TO GET THE CURRENT VALUE FOR ABS MOVE AS THE DEFAULT
#        self.parameters.addChild(dict(name='motor', value=initialName, type = 'str'))
        self.parameters.addChild(dict(name = 'motor', value = initialName, type = 'list', values = settings.userMotorDict))   
        self.parameters.addChild(dict(name = 'incrementValue', value = initialValue, type = 'float'))
        
class AbsMotorMoveBlock(Block):        
    def __init__(self):
        Block.__init__(self) #####
#####        super(AbsMotorMoveBlock, self).__init__()
#        super().__init__()
        self.addNode('N')
        self.addNode('E')
        self.addNode('S')
        value = 0
        self.regularColor = QColor(0, 150, 210)
        self.color = self.regularColor
        self.parameters.setName('rel motor move')
        self.name = ''
        self.value = value
        self.initiateParameters(0, value)       
        
    def setMethodText(self):
        motor = settings.userMotorList[self.parameters['motor']]
        self.methodText = "umv %s %f" % (motor, self.parameters['absoluteValue'])
           
    def preDrawActions(self, qp):
        motor = settings.userMotorList[self.parameters['motor']]
        self.name = "umv %s %2.4f" % (motor, self.parameters['absoluteValue'])
        
    def initiateParameters(self, initialName, initialValue):
            # WARNING WOULD NEED TO GET THE CURRENT VALUE FOR ABS MOVE AS THE DEFAULT
#        self.parameters.addChild(dict(name='motor', value=initialName, type = 'str'))
        self.parameters.addChild(dict(name = 'motor', value = initialName, type = 'list', values = settings.userMotorDict))   
        self.parameters.addChild(dict(name = 'absoluteValue', value = initialValue, type = 'float'))
        
class XRD_demo_Block(Block):
    def __init__(self): #####
        Block.__init__(self)
#####        super(XRD_Mode_Block, self).__init__()
#        super().__init__()
        self.name = 'XRD Mode'
        self.regularColor = QColor(250, 0, 0)
        self.color = self.regularColor
        self.initiateParameters()

    def initiateParameters(self):
        pathToData = 'C:\\virtualDataArea\\xrd\\'
        self.parameters.setName(self.name)
        self.parameters.addChild(self.item('detector', 1, type = 'list', values = {"Pilatus": 1, "PCO": 2, "Medipix": 3}) )
        self.parameters.addChild(self.item('dx', 1000.0))
        self.parameters.addChild(self.item('hsfg', 0.35))
        self.parameters.addChild(self.item('vsfg', 0.50))
        self.parameters.addChild(self.item('someMotor', 2.35))
        self.parameters.addChild(self.item('activeFastShutter', 0, type='bool'))

    def setMethodText(self):
        c = self.parameters.children()
        self.methodText = ""
        self.methodText += "# " + self.name + "\n"
        for i in range(0, len(c)):
            if c[i].type() == 'float':
#                self.methodText += "\t"*self.indentN +"umv %s %f \n" % (c[i].name(), c[i].value())
                self.methodText += "umv %s %f\n" % (c[i].name(), c[i].value())
            elif c[i].type() == 'int':
                self.methodText += "umv %s %d\n" % (c[i].name(), c[i].value())

class AbsModeBlock(SimpleFunctionCallBlock):
    def __init__(self):
        SimpleFunctionCallBlock.__init__(self) #####
#####       super(AbsModeBlock, self).__init__()
#        super().__init__()
        self.name = 'absconf'
        self.initValue = self.name
        self.initiateParameters()       
        
class XrdModeBlock(SimpleFunctionCallBlock):
    def __init__(self):
        SimpleFunctionCallBlock.__init__(self) #####
#####        super(XrdModeBlock, self).__init__()
#        super().__init__()
        self.name = 'xrdconf'
        self.initValue = self.name
        self.initiateParameters()       
       
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

class WhileLessThanSomeTime(BlockGroup):
    def __init__(self):
        BlockGroup.__init__(self)
#####        super(WhileLessThanSomeTime, self).__init__()
#        super().__init__()
        Type = 'BlockGroup'
        self.addNode('N')
        self.addNode('S')
        self.expanded = True
        self.visible = True
        self.name = 'while < time'
        self.regularColor = QColor(125, 125, 125)
        self.initiateParameters()
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='time', value= 60, type = 'int'))
        self.parameters.addChild(dict(name = 'units', value = 1, type='list', values = {"minutes" : 1, "seconds" : 2}))      
        self.parameters.addChild(dict(name = 'loopCounter', value = 1, type='list', values = settings.loopCounterDict))      

    def setMethodText(self):
        loopCounter = settings.loopCounterList[self.parameters['loopCounter']]
        targetTime = self.parameters['time']
        if self.parameters['units'] == 1:
            targetTime = self.parameters['time'] * 60
        else:
            targetTime = self.parameters['time']
        self.methodText = ''
        self.methodText += 'local %s = 0 \n' % (loopCounter)
        self.methodText += 't0 = time() ; t = time() \n'
        self.methodText += "while ( (t-t0) < %d ) {" % (targetTime)
        self.closingText = ''
        self.closingText += '\n\t# update while loop parameters'
        self.closingText += '\n\t%s++ ; t = time()\n' % (loopCounter)
        self.closingText += '}'
        
        
class WhileLessThanSomeTemp(BlockGroup):
    def __init__(self):
        BlockGroup.__init__(self) #####
#####        super(WhileLessThanSomeTemp, self).__init__()
#        super().__init__()
        Type = 'BlockGroup'
        self.addNode('N')
        self.addNode('S')
        self.expanded = True
        self.visible = True
        self.name = 'while < T degC'
        self.regularColor = QColor(125, 125, 125)
        self.initiateParameters()
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='targetTemp', value= 25, type = 'int'))
        self.parameters.addChild(dict(name='rampRate', value= 10, type = 'int'))
        self.parameters.addChild(dict(name='sleepPeriod (sec)', value= 10, type = 'int'))
        self.parameters.addChild(dict(name='unit', value=3, type = 'int'))
        self.parameters.addChild(dict(name='name', value=1, type = 'list', values = settings.euroMotorDict))
        self.parameters.addChild(dict(name='loopCounter', value = 0, type = 'list', values = settings.loopCounterDict))

    def setMethodText(self):
        motor = settings.euroMotorList[self.parameters['name']]
        loopCounter = settings.loopCounterList[self.parameters['loopCounter']]
        self.methodText = ''
        self.methodText = "euro2400parameters unit=%d RampRate=%d \n" % (self.parameters['unit'], self.parameters['rampRate'])
        if self.parameters['loopCounter'] > 0 :
            self.methodText += 'local %s = 0 \n' % (loopCounter)
        self.methodText += 'umv %s %d ; ct 1 \n' % (motor, self.parameters['targetTemp']) 
        self.methodText += "while( S[euro0pv] < (%d-1) ) {\n" % (self.parameters['rampRate'])
        self.closingText = ''
        self.closingText += '\n\t# update while loop parameters'
        self.closingText += '\n\tct 1 ; sleep(%d)\n' % (self.parameters['sleepPeriod (sec)'])
        if self.parameters['loopCounter'] > 0 :
            self.closingText += '\n\t%s++\n' % (loopCounter)
        self.closingText += '}'
            
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
        self.parameters.addChild(dict(name='scanFileName', value='prefix', type ='str'))
        self.parameters.addChild(dict(name='countTime (msecs)', value = 250, type = 'int'))
        self.parameters.addChild(dict(name='aerofreerun', value = 0, type = 'bool'))
        
    def setMethodText(self):
        if self.parameters['aerofreerun'] :
            self.methodText = "oneshot_norot %s %d\n" % (self.parameters['scanFileName'], self.parameters['countTime (msecs)'])
        else : 
            self.methodText = "oneshot %s %d\n" % (self.parameters['scanFileName'], self.parameters['countTime (msecs)'])
