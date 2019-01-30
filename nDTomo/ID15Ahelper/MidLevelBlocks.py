# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:04:54 2017

@author: simon
"""

from baseBlocks import Block, EasternBlock, BlockGroup
from PyQt5.QtGui import QColor
from pyqtgraph.parametertree import Parameter
from PyQt5.QtGui import QPen
import copy
import settings
##############################################################################


class SimpleFunctionCallBlock(Block):
    def __init__(self):
#        Block.__init__(self)
        super(SimpleFunctionCallBlock, self).__init__()
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
        self.methodText = 'pass'
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

    
class SimpleForBlock(BlockGroup):
    def __init__(self):
        BlockGroup.__init__(self) #####
#####       super(SimpleForBlock, self).__init__()
#        super().__init__()
        Type = 'BlockGroup'
        self.addNode('N')
        self.addNode('S')
        self.expanded = True
        self.visible = True
        self.name = 'for block'
        self.regularColor = QColor(125, 125, 125)
        self.initiateParameters()
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='loopCounter', value = 1, type = 'list', values = settings.loopCounterDict))
        self.parameters.addChild(dict(name='start', value=0, type = 'str'))
        self.parameters.addChild(dict(name='num', value=1, type = 'str'))
        self.parameters.addChild(dict(name='inc', value='++', type = 'str'))
        self.parameters.addChild(dict(name='prefix', value='dummy', type = 'str'))        
    
    ## possibly need something here if you move loops around but user may have changed
    ## the value so perhaps we check if it iii, jjj or kkk
        #self.parameters.child('loopCounters').setValue(possibleCounters[self.indentN-1]) 
    
    def setMethodText(self):
        loopCounter = settings.loopCounterList[self.parameters['loopCounter']]
        prefixVarName = 'prefix'
        prefixLoopCounterFormat = '%.4d'
        self.methodText = "for (local %s=%s; %s<%s; %s%s) {" % (loopCounter, self.parameters['start'], loopCounter, self.parameters['num'], loopCounter, self.parameters['inc'])
        self.methodText += '\n\t%s = sprintf(\"%s_%s\", %s)' % (prefixVarName, self.parameters['prefix'], prefixLoopCounterFormat, loopCounter)

class SimpleWhileLoop(BlockGroup):
    def __init__(self):
        BlockGroup.__init__(self) #####
#####        super(SimpleWhileLoop, self).__init__()
#        super().__init__()
        Type = 'BlockGroup'
        self.addNode('N')
        self.addNode('S')
        self.expanded = True
        self.visible = True
        self.name = 'while'
        self.regularColor = QColor(125, 125, 125)
        self.initiateParameters()
        
    def initiateParameters(self):
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='expression', value='', type = 'str'))
    
    ## possibly need something here if you move loops around but user may have changed
    ## the value so perhaps we check if it iii, jjj or kkk
        #self.parameters.child('loopCounters').setValue(possibleCounters[self.indentN-1]) 
    
    def setMethodText(self):
        self.methodText = "while (%s) {" % (self.parameters['expression'])

    
class ScanBlock(Block):
    def __init__(self):
        Block.__init__(self) #####
#####        super(ScanBlock, self).__init__()
#        super().__init__()
        Type = 'ScanBlock'
        name = 'generic scan block'
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
        self.easternBlock.drawBlock(qp)
        
    def postDrawActions(self, qp):
        pen = QPen(QColor(100,0,0), 1,2)
        qp.setPen(pen)
        qp.drawLine(self.nodes[1].boundingBox.center(), self.easternBlock.nodes[0].boundingBox.center())             

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
    # need checks like slow axis cannot equal fast axis
    # need checks one instance cant have same name as another instance

