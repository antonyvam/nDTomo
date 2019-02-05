# -*- coding: utf-8 -*-
"""

Other blocks

@author: S.D.M. Jacques

"""

from baseBlocks import Block, BlockGroup
from PyQt5.QtGui import QColor, QImage
import settings
        
##############################################################################
class SleepBlock(Block):
    
    """

    Sleep block

    """
    
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
        self.image = QImage('.//images//sleep.png')
          
    def initiateParameters(self):
        self.parameters.addChild(dict(name='message', value='Sleeping ...', type = 'str'))    
        self.parameters.addChild(dict(name='sleepTime(secs)', value=10, type = 'int')) 
        
    def setMethodText(self):
        self.methodText = ''
        self.methodText += "p \"%s\"\n" % (self.parameters['message'])
        self.methodText += "sleep(%d)\n" % (self.parameters['sleepTime(secs)'])
        
    def preDrawActions(self, qp):
        self.name = "sleep(%d)" % (self.parameters['sleepTime(secs)'])
        
##############################################################################   
class WaitForUserBlock(Block):
    
    """

    Wait-for-user action block

    """    
    
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
        self.image = QImage('.//images//wait.png')
        
    def initiateParameters(self):
        self.parameters.addChild(dict(name='message', value='Waiting to proceed ...', type = 'str'))
        self.parameters.addChild(dict(name='preAction', value='shclose', type = 'str')) 
        self.parameters.addChild(dict(name='postAction', value='shopen', type = 'str')) 
        
        
    def setMethodText(self):
        self.methodText = ''
        self.methodText += '%s\n' % self.parameters['preAction']
        self.methodText += 'yesno("%s","1")\n' % self.parameters['message']
        self.methodText += '%s\n' % self.parameters['postAction']
        
    def preDrawActions(self, qp):
        self.name = "%s" % (self.parameters['message'])
    
##############################################################################
class HeatingSystemBlock(Block):
    
    """

    Heating system block

    """    
    
    def __init__(self):
        Block.__init__(self) #####
#####        super(SingleEurothermBlock, self).__init__()
#        super().__init__()
        self.addNode('N')
        self.addNode('E')
        self.addNode('S')
        self.regularColor = QColor(200, 200, 200)
        self.color = self.regularColor
        self.name = 'Single Eurotherm'
        self.initiateParameters()
        self.image = QImage('.//images//heat&cool.png')
        
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