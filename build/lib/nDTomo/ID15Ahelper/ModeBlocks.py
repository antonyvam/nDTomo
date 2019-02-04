# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:15:33 2017

@author: simon
"""

from baseBlocks import Block
from PyQt5.QtGui import QColor


##############################################################################
class SimpleFunctionCallBlock(Block):
    def __init__(self):
        Block.__init__(self) #####
#####        super(SimpleFunctionCallBlock, self).__init__()
#        super().__init__()
        self.name = '#'
        self.regularColor = QColor(0, 0, 250)
        self.color = self.regularColor
#        self.parameters.setName(self.name)
        self.initValue = self.name

    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='MacroName', value=self.initValue, type = 'str'))

    def setMethodText(self):
        self.name = self.parameters['MacroName']
        self.methodText = self.name
        
##############################################################################
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
                
##############################################################################
class AbsModeBlock(SimpleFunctionCallBlock):
    def __init__(self):
        SimpleFunctionCallBlock.__init__(self) #####
#####       super(AbsModeBlock, self).__init__()
#        super().__init__()
        self.name = 'absconf'
        self.initValue = self.name
        self.initiateParameters()       
        
##############################################################################        
class XrdModeBlock(SimpleFunctionCallBlock):
    def __init__(self):
        SimpleFunctionCallBlock.__init__(self) #####
#####        super(XrdModeBlock, self).__init__()
#        super().__init__()
        self.name = 'xrdconf'
        self.initValue = self.name
        self.initiateParameters()  

##############################################################################        
class PdfModeBlock(SimpleFunctionCallBlock):
    def __init__(self):
        SimpleFunctionCallBlock.__init__(self) #####
#####        super(XrdModeBlock, self).__init__()
#        super().__init__()
        self.name = 'pdfconf'
        self.initValue = self.name
        self.initiateParameters()  
		
##############################################################################