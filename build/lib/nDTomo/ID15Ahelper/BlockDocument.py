# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:36:33 2017

@author: simon
"""
from sys import platform

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QAction, QMenu
from PyQt5.QtGui import QPainter, QBrush, QImage
import numpy as np
import json
import ast

from BlockStyles import BlockStyleStandard, BlockStyleBaby
from baseBlocks import BlockGroup #, NodeAddress, Block,
import baseBlocks

# need this import statement otherwise treeDictRead will not work properly
import ModeBlocks, ScanBlocks, MotorBlocks, LoopBlocks, OtherBlocks
  
class BlockDocument(QWidget): 
    overObjectSignal = pyqtSignal() # object cannot be inside constructor 
    trigger = pyqtSignal()
    def __init__(self):
        QWidget.__init__(self) #####
#####        super(BlockDocument, self).__init__()

        self.setMinimumWidth(1400)
        self.setMinimumHeight(1000)

        self.setMouseTracking(True)
        self.overObjectSignal.connect(self.handle_trigger)
        self.LeftButton = False  # fix for the moment to stop checkBlocks from doing too much
        self.suspend = False  

        self.selectedBlock = -1
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showMenu)
        
        # create the top block group
        self.blockTree = BlockGroup()
        self.blockTree.name = 'godBlock'
        self.blockTree.methodText = 'def id15_user_mac \'{'
        self.blockTree.parentBlock = 0
#        self.blockTree.parentId = -1
        
        self.blocks = np.empty(0,dtype=object)
        self.groups = np.empty(0,dtype=object)
        self.visibleBlocks = np.empty(0,dtype=object)
        self.clipBoardBlocks = [] #= np.empty(0,dtype=object) # could be an array later
        
        self.currentBlockStyle = 0
        self.setUpStyles() 
        self.updateBlocks()
        
        self.setAcceptDrops(True)
        
    def handle_trigger(self):
        pass

    def initUI(self):      
        self.setAutoFillBackground(True)
        sImage = QImage(".//images//backgroundImage.png")
        palette = self.palette()
        palette.setBrush(10, QBrush(sImage)) 
        self.setPalette(palette)
        self.show()
    
    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.blockTree.drawBlock(qp)
 ##       self.drawLinks(qp)
        qp.end()
        self.checkBlocks()

    def checkBlocks(self):

# inelegant bit to resize the widget        
        maxWidth = 0
        maxHeight = 0
        for i in range(0, self.visibleBlocks.size) :
#            if self.visibleBlocks[i].x > maxWidth:
#                maxWidth = self.visibleBlocks[i].x 
            if self.visibleBlocks[i].y > maxHeight:
                maxHeight = self.visibleBlocks[i].y
 #       self.setFixedWidth(maxWidth*1.1 + 250 )
        newHeight = (maxHeight+self.blockStyles[self.currentBlockStyle].height)+250
        if newHeight < 1000:
            newHeight=1000
        self.setFixedHeight(newHeight)
        
        if self.visibleBlocks.size <= 1 :
            return
# inelegant bit to resize the widget        
        

#        if self.visibleBlocks.size == 2:
#            if self.visibleBlocks[1].boundingBox.intersects(self.visibleBlocks[0].boundingBox):
#                if self.selectedBlock == 1:
#                    self.visibleBlocks[0].y -= 1
#                if self.selectedBlock == 0:
#                    self.visibleBlocks[1].y += 1
 
        self.adjustY(self.selectedBlock)        
        if self.visibleBlocks.size > 2:
            for i in range(0, self.visibleBlocks.size) :
                if i == self.selectedBlock:
                    continue
                self.adjustY(i)
             
        for i in range(0, self.visibleBlocks.size):
            defaultx = self.visibleBlocks[i].indentN * self.visibleBlocks[i].indentSize # this indentSize should be in BlockStyles
            if not self.LeftButton: 
                if self.visibleBlocks[i].x > defaultx:
                    self.visibleBlocks[i].x -= 1
                if self.visibleBlocks[i].x < defaultx:
                    self.visibleBlocks[i].x += 1            
        self.update()
        
    def adjustY(self, i):         
        if i == -1:
            return
        # there is an index out of bounds here
        if  i > 0 and i+1 >= self.visibleBlocks.size - 1:
            a = self.visibleBlocks[i-1].boundingBox;
            b = self.visibleBlocks[i].boundingBox;
            if b.intersects(a): 
                self.visibleBlocks[i-1].y -= 1
              
        if  i <= 0 and i < self.visibleBlocks.size :
            b = self.visibleBlocks[i].boundingBox;
            c = self.visibleBlocks[i+1].boundingBox;
            if b.intersects(c):
                self.visibleBlocks[i+1].y += 1
                
        if i > 0 and i < self.visibleBlocks.size - 1 :
            a = self.visibleBlocks[i-1].boundingBox;
            b = self.visibleBlocks[i].boundingBox;
            c = self.visibleBlocks[i+1].boundingBox;
            if b.intersects(a) : 
                self.visibleBlocks[i-1].y -= 1
            if b.intersects(c) :#and not(b.intersects(a)):
                self.visibleBlocks[i+1].y += 1   
                
    def setUpStyles(self):
        self.blockStyles = np.empty(0,dtype=object)
        self.blockStyles = np.append(self.blockStyles, BlockStyleStandard())
        self.blockStyles = np.append(self.blockStyles, BlockStyleBaby())
               
    def applyStyle(self, n):
        if n > self.visibleBlocks.size:
            print('error')     
        if n == 0:
            self.visibleBlocks[n].y = self.blockStyles[self.currentBlockStyle].topMargin
        else:
            self.visibleBlocks[n].y = self.visibleBlocks[n-1].y + self.blockStyles[self.currentBlockStyle].blockSpacing
        self.visibleBlocks[n].height = self.blockStyles[self.currentBlockStyle].height
        self.visibleBlocks[n].width = self.blockStyles[self.currentBlockStyle].width
        self.visibleBlocks[n].fontSize = self.blockStyles[self.currentBlockStyle].fontSize
        
    def restyle(self, newStyle):
        self.currentBlockStyle = newStyle
        for i in range(0, self.visibleBlocks.size):
            self.applyStyle(i)
        self.update()
    
    def addBlock(self, newBlock):
        self.activeGroup.addBlock(newBlock)
        self.updateBlocks()
        self.applyStyle(self.visibleBlocks.size-1)
 #       if self.visibleBlocks.size > 0 :
 #           pass
        self.update()
        self.trigger.emit()
        
    def updateBlocks(self):
        self.blockTree.setIndent()
        # sequential (non-tree) lists
        self.blocks = np.empty(0,dtype=object)
        self.blocks = self.blockTree.getAllBlocks(self.blocks)
        self.groups = np.empty(0,dtype=object)
        self.groups = self.blockTree.getAllBlockGroups(self.groups)
        self.visibleBlocks = np.empty(0,dtype=object)
        self.visibleBlocks = self.blockTree.getAllVisibleBlocks(self.visibleBlocks)
        self.activeGroup = self.groups[-1]
        for i in range(0, len(self.groups)):
            if self.groups[i-1].expanded :
                print(i)
                self.activeGroup = self.groups[i-1]
                break

        print('Visible\tType\tName\tParentName')
        print('-----------------------------------')
        self.blockTree.printBlock()
        
    def mouseMoveEvent(self, event):
        #if self.suspend :
        #    return
        
        self.selectedBlock = -1
        for i in range(0, self.visibleBlocks.size):
            if self.visibleBlocks[i].overObject(event.x(),event.y()):
                self.selectedBlock = i
              # self.overBlockSignal.emit(self.selectedBlock)
               #self.overBlockSignal.emit()                  
        if event.buttons() == Qt.NoButton:
            self.LeftButton = False
            for i in range(0, self.visibleBlocks.size):
                activeBlockFlag = self.visibleBlocks[i].overObject(event.x(),event.y())
                if activeBlockFlag:
                    self.selectedBlock = i
                    self.overObjectSignal.emit()
                    #print('emit signal')
                for j in range(0, self.visibleBlocks[i].nodes.size):
                    activeNodeFlag = self.visibleBlocks[i].nodes[j].overObject(event.x(),event.y())
                    if activeNodeFlag:
                        break
                    if activeBlockFlag:
                        break
            self.update()        
        
        elif event.buttons() == Qt.LeftButton:
            self.LeftButton = True
            for i in range(0, self.visibleBlocks.size):
                if self.visibleBlocks[i].overObject(event.x(),event.y()):
                    self.selectedBlock = i
                   # p = self.visibleBlocks[i].parentBlock
                   # j = np.nonzero(p.blocks == self.visibleBlocks[i]) # why is this bad
                   # j = int(j[0]) # why do i need this fix                    
                   # p.blocks[j].alpha = 0.5
                   # dx = event.x() - p.blocks[j].x - (p.blocks[j].width/2)
                   # dy = event.y() - p.blocks[j].y - (p.blocks[j].height/2)
                   # p.blocks[j].relX(dx)
                   # p.blocks[j].relY(dy)
                    dx = event.x() - self.visibleBlocks[i].x - (self.visibleBlocks[i].width/2)
                    dy = event.y() - self.visibleBlocks[i].y - (self.visibleBlocks[i].height/2)
                    self.visibleBlocks[i].relX(dx)
                    self.visibleBlocks[i].relY(dy)
                    if i > 0 and self.visibleBlocks[i].y < self.visibleBlocks[i-1].y:
                        self.suspend = True
                        self.swapVisibleBlocks(i, i-1)
                    if i < self.visibleBlocks.size-1 and self.visibleBlocks[i].y > self.visibleBlocks[i+1].y:
                         self.suspend = True
                         self.swapVisibleBlocks(i, i+1)
            self.suspend = False
            self.update()        
            
        elif event.buttons() == Qt.RightButton:
            pass
  
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            #print("Press!")
            pass
        
    def mouseDoubleClickEvent(self, event):
        if self.selectedBlock == -1:
            return
        if not(self.visibleBlocks[self.selectedBlock].overObject(event.x(),event.y())):
            self.selectedBlock = -1
        if self.visibleBlocks[self.selectedBlock].Type == 'BlockGroup':
            self.visibleBlocks[self.selectedBlock].toggleGroupVisibility(not(self.visibleBlocks[self.selectedBlock].expanded))
            self.updateBlocks()
            self.restyle(self.currentBlockStyle)
    
        
    def mouseReleaseEvent(self, event):
        self.LeftButton = False
        if  not(self.selectedBlock == []) and not(self.selectedBlock == -1):
            self.visibleBlocks[self.selectedBlock].alpha = self.visibleBlocks[self.selectedBlock].regularAlpha       
        self.update()

    def swapVisibleBlocks(self, i, ii):
        p = self.visibleBlocks[i].parentBlock
        q = self.visibleBlocks[ii].parentBlock
        j = p.getIndex(self.visibleBlocks[i])
        k = q.getIndex(self.visibleBlocks[ii])

        if p == q:
            p.blocks[j], p.blocks[k] = p.blocks[k], p.blocks[j]
            self.updateBlocks()
 
        if not(p == q): # or else !
            print(j,k,p.name,q.name)
            q.insertBlock(k, p.blocks[j])
            p.deleteBlock(j)
        
        self.updateBlocks()

    def showMenu(self, pos):      
        menu = QMenu(self)
        actionCopyHere = QAction("Copy here", menu)
        actionCopy = QAction("Copy", menu)
        actionPaste = QAction("Paste", menu)
        actionDelete = QAction("Delete", menu)
        menu.addAction(actionCopyHere)
        menu.addAction(actionCopy)
        menu.addAction(actionPaste)
        menu.addAction(actionDelete )
        menu.popup(self.mapToGlobal(pos))
        actionCopyHere.triggered.connect(lambda: self.copyHere(self.selectedBlock))
        actionCopy.triggered.connect(lambda: self.copyToClipboard(self.selectedBlock))
        actionPaste.triggered.connect(lambda: self.pasteBlock(self.selectedBlock))
        actionDelete.triggered.connect(lambda: self.deleteVisibleBlock(self.selectedBlock))
        
    def copyHere(self, i):
        if i < 0:
            return
        self.copyToClipboard(i)
        self.pasteBlock(i)

    def copyToClipboard(self, i):
        if i < 0:
            return             
        p = self.visibleBlocks[i].parentBlock
        j = p.getIndex(self.visibleBlocks[i])
        self.clipBoardBlocks = p.blocks[j].copy()#self.visibleBlocks[i].copy()
       
    def pasteBlock(self, i):
        if (i < 0) or (self.clipBoardBlocks == []):
            return
        p = self.visibleBlocks[i].parentBlock
        j = p.getIndex(self.visibleBlocks[i])
#        print(p.name,j)
        #self.clipBoardBlocks.setParentBlock(p)
        p.insertBlock(j+1, self.clipBoardBlocks.copy()) # need to paste a copy of what is on the clipboard otherwise its not reuseable
        self.updateBlocks()
        self.restyle(self.currentBlockStyle)
        self.update()

    def deleteVisibleBlock(self, i): 
        if i < 0:
            return        
        p = self.visibleBlocks[i].parentBlock
        j = p.getIndex(self.visibleBlocks[i])
        p.deleteBlock(j)
        self.updateBlocks()

######### to be deleted #########
    def readBlocksTest(self):
        with open('block.txt', 'r') as f:
            s = f.read()
            d = ast.literal_eval(s)
        block = eval(d['Type']+'()')
        block.__dict__.update(d)
        self.addBlock(block)
#################################

    def importBlocks(self, fileName):
        print(fileName)
        with open(fileName, 'r') as json_data:
            d = json.load(json_data)
            print(d)
        blockDict = d[0]  
        paramDict = d[1]
        self.updateBlocks()
        self.treeDictRead(blockDict, paramDict, 0)
        self.update()
 #       self.localParents = np.empty(0,dtype=object)

    def treeDictRead(self, d, p, i):
        currentActiveBlock = self.activeGroup
        for j in range(0,len(d)) :
            if (type(d[j]) == list) :
                self.treeDictRead(d[j], p[j], i+1)
            else :
                message = 'Level %d block %d\t' % (i, j)
                message = message + d[j]['Type'] + '\t' + d[j]['name'] +  '\t' + d[j]['className'] + '\t' + p[j]['name']
                print(message) 
                # if there are no blocks treat this as loading from scratch and ignore the godBlock
                if self.blockTree.blocks.size == 0 and i == 0 and d[j]['name'] == 'godBlock' :
                    #print('ham')
                    continue
                 # if there are some blocks just create a new group but dont make this a godblock
                elif self.blockTree.blocks.size > 0 and i == 0 and d[j]['name'] == 'godBlock' :
#                    print(d[j]['className'])
#                    print(d[j]['moduleName'])                    
#                    block = eval(d[j]['moduleName'] + '.' + d[j]['className'] + '()')
                    block = BlockGroup()
                    block.name = 'group'
                    block.expanded = True
                    block.visible = True
                    self.addBlock(block)
                    continue
                # for any other block type
                else : 
#                    print(d[j]['moduleName'])
#                    print(d[j]['className'])
#                    if d[j]['className'] == 'BlockGroup':
#                        block = BlockGroup
#                        block.name = 'group'
#                        block.expanded = True
#                        block.visible = True
#                        print('cheese')
#                        self.addBlock(block)
#                        continue
                    block = eval(d[j]['moduleName'] + '.' + d[j]['className'] + '()')
                    block.parameters.restoreState(p[j])
##                    self.activeGroup = self.groups[-1]
                    currentActiveBlock.addBlock(block)
                    self.updateBlocks()
                    self.applyStyle(self.visibleBlocks.size-1)                # if it were the god block do not create it but copy the dictionaries
 #               message = 'Level %d block %d\t' % (i, j)
 #               message = message + d[j]['Type'] + '\t' + d[j]['name'] +  '\t' + d[j]['className'] + '\t' + p[j]['name']
 #               print(message)            

    def exportBlocks(self, fileName):        
        stuffToWrite = []
#        for i in range(0, self.visibleBlocks.size) :
#            block = self.visibleBlocks[i]
        stuffToWrite = self.blockTree.serialise()
        print(stuffToWrite)
        with open(fileName, 'w') as file:
            file.write(json.dumps(stuffToWrite))
            
            
    def dropEvent(self, event) :
        s = event.mimeData().text()
        if 'file:' in s and '.blox' in s :
            if platform == 'win32' or platform == 'win64' :
                s = s.replace('file:///', '')
            else : # linux
                s = s.replace('file://', '')
            self.importBlocks(s)
        #self.addItem(e.mimeData().text())
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.accept()
        else:
            event.ignore()

    def clear(self, event):
        pass

'''
import HighLevelBlocks, MidLevelBlocks
w=BlockDocument()
w.addBlock(HighLevelBlocks.XRD_Mode_Block())
w.addBlock(HighLevelBlocks.Simple_XRD_CT_Block())
w.show()
w.restyle(1)

w.addBlock(MidLevelBlocks.ForBlock())
w.addBlock(HighLevelBlocks.RelMotorMoveBlock())
w.addBlock(HighLevelBlocks.RelMotorMoveBlock())
#w.addBlock(MidLevelBlocks.ForBlock())
#w.addBlock(HighLevelBlocks.RelMotorMoveBlock('but',10.0))
#w.addBlock(HighLevelBlocks.RelMotorMoveBlock('saz',0.0))
w.update()
A=w.visibleBlocks[0]
B=w.visibleBlocks[2]
C=w.visibleBlocks[3]
'''
