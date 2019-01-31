# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:41:06 2017

@author: simon
"""

import numpy as np
from PyQt5.QtCore import QObject, QRect, Qt, QSize, QPoint
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from pyqtgraph.parametertree import Parameter

#from PyQt5.QtGui import QPen

class Node():
    def __init__(self, Location):
        self.Location = Location
        self.regularColor = QColor(255, 255, 255)
        self.color = self.regularColor
        self.highlightColor = QColor(255, 255, 0)
        self.x = 0
        self.y = 0
        self.nodeSize = 6
        self.boundingBox = QRect(self.x, self.y, self.nodeSize, self.nodeSize)

    def drawNode(self, qp, parent):
        if (self.Location == 'N'):
            self.x = parent.boundingBox.left() + parent.boundingBox.width()/2-self.nodeSize/2
            self.y = parent.boundingBox.top()  #- self.nodeSize/2
            self.boundingBox = QRect(self.x, self.y, self.nodeSize, self.nodeSize)
        if (self.Location == 'S'):
            self.x = parent.boundingBox.left() + parent.boundingBox.width()/2-self.nodeSize/2
            self.y = self.y = parent.boundingBox.bottom()  - self.nodeSize + 1
            self.boundingBox = QRect(self.x, self.y, self.nodeSize, self.nodeSize)
        if (self.Location == 'W'):
            self.x = parent.boundingBox.left()
            self.y = parent.boundingBox.top() + parent.boundingBox.height()/2-self.nodeSize/2
            self.boundingBox = QRect(self.x, self.y, self.nodeSize, self.nodeSize)
        if (self.Location == 'E'):
            self.x = parent.boundingBox.right()-self.nodeSize
            self.y = parent.boundingBox.top() + parent.boundingBox.height()/2-self.nodeSize/2
            self.boundingBox = QRect(self.x, self.y, self.nodeSize, self.nodeSize)
        qp.setBrush(self.color)
        qp.drawRect(self.boundingBox)
        
    def overObject(self, x, y):
        if self.boundingBox.contains(x, y):
            self.color = self.highlightColor
            return True
        else:
            self.color = self.regularColor
            return False


class NodeAddress():
    def __init__(self, blockNumber0, nodeNumber0, blockNumber1, nodeNumber1):
        self.fromBlock = blockNumber0
        self.fromNode = nodeNumber0
        self.toBlock = blockNumber1
        self.toNode = nodeNumber1


class Block():
    def __init__(self):
        from pyqtgraph.parametertree import Parameter
#        object.__init__(self)
#        super.__init__()
        self.className = self.__class__.__name__
        self.moduleName = self.__class__.__module__
        self.Type = 'Block'
        self.parentBlock = 0
        self.visible = True
        self.name = ''
        self.regularColor = QColor(200, 200, 200)
        self.color = self.regularColor
        self.highlightColor = QColor(255, 0, 255)
        self.regularAlpha = .8
        self.alpha = self.regularAlpha
        self.fontColor = QColor(10,10,10)
        self.fontSize = 12
        self.x = 100 # perhaps duff now
        self.y = 50
        self.width = 180
        self.height = 50
        self.blockSize = 10
        self.boundingBox = QRect(self.x, self.y, self.width, self.height)     
        self.nodes = np.empty(0,dtype=object)
        self.indentSize = 75
        self.indentN = 0  
        self.parameters = Parameter.create(name=self.name, type='group', children=[])
        self.methodText = 'pass'
        self.sideText = ''
        self.time = 0
        self.selected = False
        # these properties recently added
        self.parentId = 0
        self.id = 0
        self.childrenId = []
        self.image = QImage('.//images//null.png')

#    def clone(self):
#        newBlock = type(self.__class__.__name__, (self.__class__.__base__,), self.__dict__)
#        newBlock.parameters = Parameter.create(name=self.parameters.name(), type='group', children = self.parameters.children())
#        newBlock.blocks = np.empty(0,dtype=object)
#        for attr, value in self.__dict__.items():
#            print(attr, value)
#        return newBlock

    def getCode(self, i, j) :
        self.setMethodText()
        s = "\n# Block %d\n" % (i) + self.methodText 
        indent =  "\t"*j
        s = indent.join(s.splitlines(True))
        return s, i

    def copy(self):
        print('coying ..' + self.__class__.__name__)
        copyInstance = self.__class__()
        copyInstance.__dict__= self.__dict__.copy()
        copyInstance.parameters = Parameter.create(name=self.parameters.name(), type='group', children =[])#children = self.parameters.children()
        parametersCopy = self.parameters.saveState()
        copyInstance.parameters.restoreState(parametersCopy)
        return copyInstance 

    def serialise(self):
        blockDictCopy = self.__dict__.copy()
        blockParametersCopy = self.parameters.saveState()
        del blockDictCopy['parentBlock']
        del blockDictCopy['nodes']
        del blockDictCopy['parameters']
        del blockDictCopy['boundingBox']
        del blockDictCopy['color']
        del blockDictCopy['fontColor']
        del blockDictCopy['highlightColor']
        del blockDictCopy['regularColor']
        del blockDictCopy['image']
        if 'easternBlock' in blockDictCopy.keys():
            del blockDictCopy['easternBlock']
#        serialList = [blockDictCopy, blockParametersCopy]
        return [blockDictCopy, blockParametersCopy]
    
    def getParentLoopCounters(self, loopCounters):    
        if self.parentBlock == 0:
            return loopCounters
        if not(self.parentBlock.loopCounter == ''):
            loopCounters.append(self.parentBlock.loopCounter)
        loopCounters = self.parentBlock.getParentLoopCounters(loopCounters)
        return loopCounters

    def initiateParameters(self):
        pass
  
    def setParentBlock(self, parentBlock):
        self.parentBlock = parentBlock
    
    def setIndent(self):
            self.indentN = self.parentBlock.indentN+1
        
    def setX(self, x):
        self.x = x
             
    def setY(self, y):
        self.y = y             

    def relX(self, x):
        self.x += x
             
    def relY(self, y):
        self.y += y
        
    def setMethodText(self): # this may be defunct as it is now in the DemoMainWindow class
        pass
         
    def addNode(self, Location):
        node = Node(Location)
        self.nodes = np.append(self.nodes, node)
        
    def drawBlock(self, qp):
        if not self.visible :
            return
        self.preDrawActions(qp)
        qp.setPen(self.color)
        qp.setBrush(self.color)
        qp.setOpacity(self.alpha)
        self.boundingBox = QRect(self.x, self.y, self.width, self.height)
        qp.drawRect(self.boundingBox)
        for i in range(0, self.nodes.size):
            self.nodes[i].drawNode(qp, self)
        qp.setPen(self.fontColor)
        qp.setFont(QFont("Segoe UI", self.fontSize, italic=False))
        qp.drawText(self.boundingBox, Qt.AlignCenter, self.name)
### moved from postDrawActions in scanblock
        rect = QRect(self.boundingBox.topRight()+QPoint(10,0),QSize(500,100))
        qp.setFont(QFont("Segoe UI", self.fontSize, italic=False))
        qp.setPen(QColor(0,0,0))
        qp.drawText(rect, Qt.AlignLeft, self.sideText)       
###
        rect2 = QRect(self.boundingBox.topLeft()-QPoint(self.height+10,0),QSize(self.height,self.height))
        qp.drawImage(rect2, self.image)
        self.postDrawActions(qp)
        
    def overObject(self, x, y):
        if self.boundingBox.contains(x, y):
            self.color = self.highlightColor
            return True
        else:
            self.color = self.regularColor
            return False
        
    def selectedObject(self):
        self.selected = not(self.selected)
        if self.selected:
            self.color = self.highlightColor
        else:
            self.color = self.regularColor

    def preDrawActions(self, qp):
        pass
    
    def postDrawActions(self, qp):
        pass
    
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
    
    def humanTime(self):
        self
        m, s = divmod(self.time, 60)
        h, m = divmod(m, 60)
        if h > 0 :
            strTime = "%d hr %d min %d sec" % (h,m,s)
        if h == 0 and m > 0:
            strTime = "%d min %d sec" % (m,s)
        if h == 0 and m == 0:
            strTime = "%d sec" % (s)
        return strTime
        
class EasternBlock(Block):
    def __init__(self):
        Block.__init__(self) #####
####        super(EasternBlock, self).__init__()
#        super().__init__()
        self.addNode('W') 
        self.regularColor = QColor(240, 240, 240)
        self.Color = self.regularColor 
        self.x = 350
        self.width = 90
        self.height = 90     


##############################################################################        

class BlockGroup(Block):
    def __init__(self):
        Block.__init__(self)
#####        super(BlockGroup, self).__init__()
#        super().__init__()
        self.Type = 'BlockGroup'
        #self.__dict__ = self
        self.blocks = np.empty(0,dtype=object)
        self.expanded = True
        self.visible = False
        self.openingText = ''
        self.methodText = '{'
        self.closingText = '\n}'
        self.loopCounter = ''

    def getCode(self, i, j): # i is the block level and j is the block number
        self.setMethodText()
        s = ''
        # only want to do this if its practical
        t = "\n# Block %d\n" % (i) + self.openingText + self.methodText    
        indent =  "\t"*j
        t = indent.join(t.splitlines(True))
        closingText = indent.join(self.closingText.splitlines(True))
        for k in range(0,self.blocks.size):
            i += 1
            sOfChild, i = self.blocks[k].getCode(i, j+1)
            s += sOfChild
        s = t + s + closingText
        return s, i

    def copy(self):
        copyInstance = self.__class__()
        copyInstance.__dict__= self.__dict__.copy()
        copyInstance.parameters = Parameter.create(name=self.parameters.name(), type='group', children =[])#children = self.parameters.children()
        parametersCopy = self.parameters.saveState()
        copyInstance.parameters.restoreState(parametersCopy)
        copyInstance.blocks = np.empty(0,dtype=object)
        for i in range(0, self.blocks.size):
            childCopy = self.blocks[i].copy()
            childCopy.setParentBlock(copyInstance)
            copyInstance.blocks = np.append(copyInstance.blocks, childCopy)
        return copyInstance 
    
    def serialise(self):
        serialList = []
        blockDictCopy = self.__dict__.copy()
        blockParametersCopy = self.parameters.saveState()
        del blockDictCopy['parentBlock']
        del blockDictCopy['blocks']
        del blockDictCopy['nodes']
        del blockDictCopy['parameters']
        del blockDictCopy['boundingBox']
        del blockDictCopy['color']
        del blockDictCopy['fontColor']
        del blockDictCopy['highlightColor']
        del blockDictCopy['regularColor']
        del blockDictCopy['image']
        blockDictList = []
        blockParametersList = []
        for i in range(0, self.blocks.size):
            blockChilsDictCopy, blockChildParametersCopy = self.blocks[i].serialise()
            blockDictList.append(blockChilsDictCopy)
            blockParametersList.append(blockChildParametersCopy)          
        blockDictList = [blockDictCopy, blockDictList]
        blockParametersList = [blockParametersCopy, blockParametersList]
        return blockDictList, blockParametersList   

    def setParentBlock(self, parentBlock):
        self.parentBlock = parentBlock
        for i in range(0, self.blocks.size):
            self.blocks[i].setParentBlock(self)     
        
    def drawBlock(self, qp):
        # if expanded (otherwise we should do what is in the drawBlock method in the Block class)
        self.drawSelf(qp)
        for i in range(0, self.blocks.size):
            #print(self.blocks[i].name)
            self.blocks[i].drawBlock(qp)
        
    def drawSelf(self, qp):
        if not self.visible :
            return
        self.preDrawActions(qp)
        qp.setPen(self.color)
        qp.setBrush(self.color)
        qp.setOpacity(self.alpha)
        self.boundingBox = QRect(self.x, self.y, self.width, self.height)
        qp.drawRect(self.boundingBox)
        for i in range(0, self.nodes.size):
            self.nodes[i].drawNode(qp, self)
        qp.setPen(self.fontColor)
        qp.setFont(QFont("Arial", self.fontSize, italic=False))
        qp.drawText(self.boundingBox, Qt.AlignCenter, self.name)
        rect = QRect(self.boundingBox.topRight()+QPoint(10,0),QSize(500,100))
        qp.setFont(QFont("Segoe UI", self.fontSize, italic=False))
        qp.setPen(QColor(0,0,0))
        qp.drawText(rect, Qt.AlignLeft, self.sideText)
        rect2 = QRect(self.boundingBox.topLeft()-QPoint(self.height+10,0),QSize(self.height,self.height))
        qp.drawImage(rect2, self.image)
        self.postDrawActions(qp)
        
    def setIndent(self):
        if not(self.parentBlock == 0) :
            self.indentN = self.parentBlock.indentN+1
        for i in range(0, self.blocks.size):
            self.blocks[i].setIndent()     

    def addBlock(self, newBlock):
        self.blocks = np.append(self.blocks, newBlock)   
        i = self.blocks.size
        self.blocks[i-1].setParentBlock(self)
    
    def insertBlock(self, i, block):
        self.blocks = np.insert(self.blocks, i, block)   
        self.blocks[i].setParentBlock(self)
        
    def deleteBlock(self, i):
        self.blocks = np.delete(self.blocks, i)   
        
    def getIndex(self, block):
        for i in range(0, self.blocks.size):
            if self.blocks[i] == block:
                return i
        return -1
  
    def toggleGroupVisibility(self, state):
        self.expanded = state
        for i in range(0, self.blocks.size):
            self.blocks[i].visible = state #!self.blocks[i].isVisible
            if self.blocks[i].Type == 'BlockGroup' :
                self.blocks[i].toggleGroupVisibility(state)
                
    def getAllBlocks(self, allBlocks): 
        allBlocks = np.append(allBlocks, self)  
        for i in range(0, self.blocks.size):
            if self.blocks[i].Type == 'BlockGroup':
                allBlocks = self.blocks[i].getAllBlocks(allBlocks)
            else:
                allBlocks = np.append(allBlocks, self.blocks[i])
        return allBlocks

    def getAllBlockGroups(self, allBlocks): 
        allBlocks = np.append(allBlocks, self)  
        for i in range(0, self.blocks.size):
            if self.blocks[i].Type == 'BlockGroup' and self.expanded :
                allBlocks = self.blocks[i].getAllBlockGroups(allBlocks)
        return allBlocks
       
    def getAllNonGroupBlocks(self, allBlocks):
        for i in range(0, self.blocks.size):
            if self.blocks[i].Type == 'BlockGroup' and self.expanded :
                allBlocks = self.blocks[i].getAllNonGroupBlocks(allBlocks)
            else:
                allBlocks = np.append(allBlocks, self.blocks[i])
        return allBlocks
    
    def getAllVisibleBlocks(self, allBlocks):
        for i in range(0, self.blocks.size):
            if self.blocks[i].visible:
                allBlocks = np.append(allBlocks, self.blocks[i])
            if self.blocks[i].Type == 'BlockGroup' :
                allBlocks = self.blocks[i].getAllVisibleBlocks(allBlocks)
        return allBlocks
    
    def getAllMinusGodBlock(self, allBlocks): 
        for i in range(0, self.blocks.size):
            allBlocks = np.append(allBlocks, self.blocks[i])
            if self.blocks[i].Type == 'BlockGroup':
                allBlocks = self.blocks[i].getAllMinusGodBlock(allBlocks)
        return allBlocks
       
    def relX(self, x):
        self.x += x
        for i in range(0, self.blocks.size):
            self.blocks[i].relX(x)
             
    def relY(self, y):
        self.y += y
        for i in range(0, self.blocks.size):
            self.blocks[i].relY(y)
        
    def printBlock(self):
        for i in range(0, self.blocks.size):
            print(self.blocks[i].visible, '\t', self.blocks[i].Type,'\t',self.blocks[i].name,'\t', self.blocks[i].parentBlock.name)
            if self.blocks[i].Type == 'BlockGroup' :
                self.blocks[i].printBlock()
    


