# -*- coding: utf-8 -*-
"""

Loop blocks

@author: S.D.M. Jacques

"""

from baseBlocks import BlockGroup
from PyQt5.QtGui import QColor, QImage
import settings

from LoopCounterManager import LoopCounterManager
loopCounterManager = LoopCounterManager()     

##############################################################################

class SimpleForBlock(BlockGroup):
    
    """
    
    For loop block
    
    """
    
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
        #self.image = QImage('.//images//loop.png')
        
    def initiateParameters(self):
        key = next(iter(loopCounterManager.availableCounters))
        value = loopCounterManager.availableCounters[key]
        d = {key : value}
        d.update(loopCounterManager.availableCounters)
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='loopCounter', value = value, type = 'list', values = d))
        self.parameters.addChild(dict(name='start', value=0, type = 'int'))
        self.parameters.addChild(dict(name='num', value=1, type = 'int'))
        self.parameters.addChild(dict(name='inc', value='++', type = 'str'))
#        self.parameters.addChild(dict(name='prefix', value='dummy', type = 'str')) 
        self.loopCounter = loopCounterManager.getCounterName(self.parameters['loopCounter'])
        loopCounterManager.useCounter(key)
    
    def setMethodText(self):
        self.loopCounter = loopCounterManager.getCounterName(self.parameters['loopCounter'])
        prefixVarName = 'prefix'
        prefixLoopCounterFormat = '%.4d'
        self.methodText = "local %s\nfor (%s=%s; %s<%s; %s%s) {" % (self.loopCounter, self.loopCounter, self.parameters['start'], self.loopCounter, self.parameters['num'], self.loopCounter, self.parameters['inc'])
#        self.methodText += '\n\t%s = sprintf(\"%s_%s\", %s)' % (prefixVarName, self.parameters['prefix'], prefixLoopCounterFormat, self.loopCounter)

class SimpleWhileLoop(BlockGroup):
    
    """
    
    While loop block
    
    """
    
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
    
    """
    
    While temporal block
    
    """
    
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
        d = {key : value}
        d.update(loopCounterManager.availableCounters)
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='time', value= 60, type = 'int'))
        self.parameters.addChild(dict(name = 'units', value = 1, type='list', values = {"minutes" : 1, "seconds" : 2}))      
        self.parameters.addChild(dict(name = 'loopCounter', value = value, type='list', values = d))      
        self.loopCounter = loopCounterManager.getCounterName(self.parameters['loopCounter'])
        loopCounterManager.useCounter(key)
        
    def setMethodText(self):
        self.loopCounter = loopCounterManager.getCounterName(self.parameters['loopCounter'])
        targetTime = self.parameters['time']
        if self.parameters['units'] == 1:
            targetTime = self.parameters['time'] * 60
        else:
            targetTime = self.parameters['time']
        self.methodText = ''
        self.methodText += 'local %s \n' % (self.loopCounter)
        self.methodText += '%s = 0 \n' % (self.loopCounter)
        self.methodText += 't0 = time() ; t = time() \n'
        self.methodText += "while ( (t-t0) < %d ) {" % (targetTime)
        self.closingText = ''
        self.closingText += '\n\t# update while loop parameters'
        self.closingText += '\n\t%s++ ; t = time()\n' % (self.loopCounter)
        self.closingText += '}'
              
class WhileTempRamp(BlockGroup):
    
    """
    
    While temperature block
    
    """
    
    def __init__(self):
        BlockGroup.__init__(self) #####
#####        super(WhileLessThanSomeTemp, self).__init__()
#        super().__init__()
        self.Type = 'BlockGroup'
        self.addNode('N')
        self.addNode('S')
        self.expanded = True
        self.visible = True
        self.name = 'while ramp'
        self.regularColor = QColor(232,65,60)
        self.initiateParameters()
        self.image = QImage('.//images//heat.png')
        
    def initiateParameters(self):
        key = next(iter(loopCounterManager.availableCounters))
        value = loopCounterManager.availableCounters[key]
        d = {key : value}
        d.update(loopCounterManager.availableCounters)
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='up/down', value= 1, type = 'list',values = {'Down' : 0, 'Up' : 1}))
        self.parameters.addChild(dict(name='targetTemp', value= 25, type = 'int'))
        self.parameters.addChild(dict(name='rampRate', value= 10, type = 'int'))
        self.parameters.addChild(dict(name='targetMargin', value = 1, type = 'int'))
        self.parameters.addChild(dict(name='sleepPeriod (sec)', value= 10, type = 'int'))
        self.parameters.addChild(dict(name='currentTemp', value=25, type = 'int'))        
        self.parameters.addChild(dict(name='unit', value=3, type = 'int'))
        self.parameters.addChild(dict(name='name', value=0, type = 'list', values = settings.euroMotorDict))
        self.parameters.addChild(dict(name='loopCounter', value = value, type = 'list', values = d))
        self.parameters.addChild(dict(name='euroScaleFactor', value= 0.1, type = 'float'))
        
        self.loopCounter = loopCounterManager.getCounterName(self.parameters['loopCounter'])
        loopCounterManager.useCounter(key)
        
    def setMethodText(self):
        motor = settings.euroMotorList[self.parameters['name']]
        if self.parameters['up/down'] == 0 :
            conditionSymbol = '>'
            mathsSymbol = '+'
            self.name = 'while ramp DOWN'
            self.regularColor = QColor(21, 87, 166)
            self.image = QImage('.//images//cool.png')
        else : 
            conditionSymbol = '<'
            mathsSymbol = '-'
            self.name = 'while ramp UP'
            self.regularColor = QColor(232,65,60)
            self.image = QImage('.//images//heat.png')
        self.loopCounter = loopCounterManager.getCounterName(self.parameters['loopCounter'])
        self.methodText = ''
        self.methodText = "euro2400parameters unit=%d RampRate=%d \n" % (self.parameters['unit'], self.parameters['rampRate'])
        if self.parameters['loopCounter'] > -1 : # this if block   
            self.methodText += 'local %s \n' % (self.loopCounter)
            self.methodText += '%s = 0 \n' % (self.loopCounter)
            
        self.methodText += 'umv %s %f ; ct 1 \n' % (motor, self.parameters['targetTemp']*self.parameters['euroScaleFactor']) 
        self.methodText += "while( S[euro0pv] %s (%d%s%d) ) {\n" % (conditionSymbol, self.parameters['targetTemp'], mathsSymbol, abs(self.parameters['targetMargin']))
        self.closingText = ''
        self.closingText += '\n\t# update while loop parameters'
        self.closingText += '\n\tct 1 ; sleep(%d)\n' % (self.parameters['sleepPeriod (sec)'])
        if self.parameters['loopCounter'] > 0 :
            self.closingText += '\n\t%s++\n' % (self.loopCounter)
        self.closingText += '}'
        # need to look for previos set point and if not assume it 25 degC
        self.time = abs(float(self.parameters['targetTemp']-self.parameters['currentTemp']))/float(self.parameters['rampRate'])*60
        self.sideText = 'Ramp to %d degC @ %d deg/min\n' % (self.parameters['targetTemp'], self.parameters['rampRate'])
        self.sideText += self.humanTime()


class PositionalSeriesBlock(BlockGroup):
    
    """
    
    Positional series block
    
    """
    
    def __init__(self):
        BlockGroup.__init__(self) #####
#####       super(SimpleForBlock, self).__init__()
#        super().__init__()
        self.Type = 'BlockGroup'
        self.addNode('N')
        self.addNode('S')
        self.expanded = True
        self.visible = True
        self.name = 'Positional Series'
        self.regularColor = QColor(125, 125, 125)
        self.initiateParameters(2,0)
        self.positionsList = [] #np.empty(0)
        #self.image = QImage('.//images//loop.png')
    
    def initiateParameters(self, initialName, initialValue):
        key = next(iter(loopCounterManager.availableCounters))
        value = loopCounterManager.availableCounters[key]
        d = {key : value}
        d.update(loopCounterManager.availableCounters)
        self.parameters.setName(self.name)
        self.parameters.addChild(dict(name='loopCounter', value = value, type = 'list', values = d))
        self.parameters.addChild(dict(name = 'motor', value = initialName, type = 'list', values = settings.userMotorDict))  
        self.parameters.addChild(dict(name='num', value=2, type = 'int'))
        self.parameters.addChild(dict(name = 'listName', value = 'vector1', type = 'str'))  
#        self.parameters.addChild(dict(name='scanType', value=1, type = 'int'))
#        self.parameters.addChild(dict(name='prefix', value='dummy', type = 'str')) 
        self.loopCounter = loopCounterManager.getCounterName(self.parameters['loopCounter'])
        loopCounterManager.useCounter(key)
        
        self.positionsList = [0, -2,-10]
    
    def setMethodText(self):
        self.loopCounter = loopCounterManager.getCounterName(self.parameters['loopCounter'])
        motor = settings.userMotorList[self.parameters['motor']]
        prefixVarName = 'prefix'
        prefixLoopCounterFormat = '%.4d'
        self.methodText = "local %s" % (self.parameters['listName'])
        #for i in range(0, len(self.positionsList)):
        #    self.methodText+= "%s[%d] = %f\n" % (self.parameters['listName'], i, self.positionsList[i])
        self.methodText = "for (local %s=%s; %s<%s; %s%s) {" % (self.loopCounter, 0, self.loopCounter, self.parameters['num'], self.loopCounter, '++')
#        self.methodText += '\n\t%s = sprintf(\"umv %s %s\", %s)' % (prefixVarName, self.parameters['prefix'], prefixLoopCounterFormat, self.loopCounter)
        self.methodText += "\n\tumv %s %f" % (motor, 999.99)
