# -*- coding: utf-8 -*-
"""

Motor blocks

@author: S.D.M. Jacques

"""
from baseBlocks import Block
from PyQt5.QtGui import QColor, QImage
import settings

##############################################################################             
class RelMotorMoveBlock(Block):
    
    """
    
    Relative motor movement block
    
    """
    
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
        self.image = QImage('.//images//cog2.png')
        
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

##############################################################################        
class AbsMotorMoveBlock(Block):        
    
    """
    
    Absolute motor movement block
    
    """
    
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
        self.image = QImage('.//images//cog3.png')
        
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