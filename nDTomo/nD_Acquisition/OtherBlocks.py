# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:04:54 2017

@author: simon
"""

from baseBlocks import Block
from PyQt5.QtGui import QColor
import settings
        
##############################################################################
class SleepBlock(Block):
    def __init__(self):
        Block.__init__(self) #####
#####        super(SleepBlock, self).__init__()
#        super().__init__()
        self.addNode('N')
        self.addNode('E')
        self.addNode('S')
        self.regularColor = QColor(0, 150, 150)
        self.color = self.regularColor
        self.initiateParameters()
          
    def initiateParameters(self):
        self.parameters.addChild(dict(name='message', value='Sleeping ...', type = 'str'))    
        self.parameters.addChild(dict(name='sleepTime(secs)', value=10, type = 'int')) 
        
    def setMethodText(self):
        self.methodText = ''
        self.methodText += "p \"%s\"\n" % (self.parameters['message'])
        self.methodText += "Sleep(%d)\n" % (self.parameters['sleepTime(secs)'])
        
    def preDrawActions(self, qp):
        self.name = "sleep(%d)" % (self.parameters['sleepTime(secs)'])
        
##############################################################################   
class WaitForUserBlock(Block):
    def __init__(self):
        Block.__init__(self) #####
#####        super(WaitForUserBlock, self).__init__()
#        super().__init__()
        self.addNode('N')
        self.addNode('E')
        self.addNode('S')
        self.regularColor = QColor(0, 150, 150)
        self.color = self.regularColor
        self.initiateParameters()    
        
    def initiateParameters(self):
        self.parameters.addChild(dict(name='message', value='Waiting to proceed ...', type = 'str'))
        self.parameters.addChild(dict(name='preAction', value='shclose', type = 'str')) 
        self.parameters.addChild(dict(name='postAction', value='shopen', type = 'str')) 
        
        
    def setMethodText(self):
        self.methodText = ''
        self.methodText += '%s\n' % self.parameters['preAction']
        self.methodText += 'yesno("%s","1")\n' % self.parameters['message']
        self.methodText += '%s\n' % self.parameters['preAction']
        
    def preDrawActions(self, qp):
        self.name = "%s" % (self.parameters['message'])
    
##############################################################################
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
        
##############################################################################
class DualEurothermBlock(Block):
    pass
##############################################################################