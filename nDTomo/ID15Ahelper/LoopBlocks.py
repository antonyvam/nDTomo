# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:34:24 2017

@author: simon
"""

from baseBlocks import BlockGroup
from PyQt5.QtGui import QColor
import settings

from LoopCounterManager import LoopCounterManager
loopCounterManager = LoopCounterManager()     
##############################################################################

class SimpleForBlock(BlockGroup):
    def __init__(self):
        BlockGroup.__init__(self) #####
#####       super(SimpleForBlock, self).__init__()
#        super().__init__()
        self.Type = 'BlockGroup'
        self.addNode('N')
        self.addNode('S')
        self.expanded = True
        self.visible = True
        self.name = 'for block'
        self.regularColor = QColor(125, 125, 125)
        self.initiateParameters()
        
    def initiateParameters(self):
        key = next(iter(loopCounterManager.availableCounters))
        value = loopCounterManager.availableCounters[key]
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='loopCounter', value = value, type = 'list', values = loopCounterManager.availableCounters))
        self.parameters.addChild(dict(name='start', value=0, type = 'str'))
        self.parameters.addChild(dict(name='num', value=1, type = 'str'))
        self.parameters.addChild(dict(name='inc', value='++', type = 'str'))
        self.parameters.addChild(dict(name='prefix', value='dummy', type = 'str')) 
#        loopCounterManager.useCounter(key)
    
    def setMethodText(self):
        loopCounter = loopCounterManager.getCounterName(self.parameters['loopCounter'])
        prefixVarName = 'prefix'
        prefixLoopCounterFormat = '%.4d'
        self.methodText = "for (local %s=%s; %s<%s; %s%s) {" % (loopCounter, self.parameters['start'], loopCounter, self.parameters['num'], loopCounter, self.parameters['inc'])
        self.methodText += '\n\t%s = sprintf(\"%s_%s\", %s)' % (prefixVarName, self.parameters['prefix'], prefixLoopCounterFormat, loopCounter)

class SimpleWhileLoop(BlockGroup):
    def __init__(self):
        BlockGroup.__init__(self) #####
#####        super(SimpleWhileLoop, self).__init__()
#        super().__init__()
        self.Type = 'BlockGroup'
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
    
    def setMethodText(self):
        self.methodText = "while (%s) {" % (self.parameters['expression'])

class WhileLessThanSomeTime(BlockGroup):
    def __init__(self):
        BlockGroup.__init__(self)
#####        super(WhileLessThanSomeTime, self).__init__()
#        super().__init__()
        self.Type = 'BlockGroup'
        self.addNode('N')
        self.addNode('S')
        self.expanded = True
        self.visible = True
        self.name = 'while < time'
        self.regularColor = QColor(125, 125, 125)
        self.initiateParameters()
        
    def initiateParameters(self):
        key = next(iter(loopCounterManager.availableCounters))
        value = loopCounterManager.availableCounters[key]
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='time', value= 60, type = 'int'))
        self.parameters.addChild(dict(name = 'units', value = 1, type='list', values = {"minutes" : 1, "seconds" : 2}))      
        self.parameters.addChild(dict(name = 'loopCounter', value = value, type='list', values = loopCounterManager.availableCounters))      

    def setMethodText(self):
        loopCounter = loopCounterManager.getCounterName(self.parameters['loopCounter'])
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
        self.Type = 'BlockGroup'
        self.addNode('N')
        self.addNode('S')
        self.expanded = True
        self.visible = True
        self.name = 'while < T degC'
        self.regularColor = QColor(125, 125, 125)
        self.initiateParameters()
        
    def initiateParameters(self):
        key = next(iter(loopCounterManager.availableCounters))
        value = loopCounterManager.availableCounters[key]
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='targetTemp', value= 25, type = 'int'))
        self.parameters.addChild(dict(name='rampRate', value= 10, type = 'int'))
        self.parameters.addChild(dict(name='sleepPeriod (sec)', value= 10, type = 'int'))
        self.parameters.addChild(dict(name='unit', value=3, type = 'int'))
        self.parameters.addChild(dict(name='name', value=1, type = 'list', values = settings.euroMotorDict))
        self.parameters.addChild(dict(name='loopCounter', value = value, type = 'list', values = loopCounterManager.availableCounters))

    def setMethodText(self):
        motor = settings.euroMotorList[self.parameters['name']]
        loopCounter = loopCounterManager.getCounterName(self.parameters['loopCounter'])
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
