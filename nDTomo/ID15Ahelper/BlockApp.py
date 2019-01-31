# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:08:04 2017

@author: simon
"""
#from PyQt5 import QtWidgets, QtGui, QtCore
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QTextEdit 
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QDockWidget, QTabWidget, QScrollArea
from PyQt5.QtGui import QFont, QFontMetrics#, QTextCursor
from PyQt5.QtCore import Qt


import specSyntax

import HighLevelBlocks
import MidLevelBlocks
from BlockDocument import BlockDocument
from pyqtgraph.parametertree import ParameterTree


class BlockMainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self) #####
#####        super(BlockMainWindow, self).__init__()
        
#        layout = QHBoxLayout()        
        self.setWindowTitle('ID15 SpecBlox')
        
#        self.setWindowModality(Qt.ApplicationModal)
        self.setGeometry(200, 200, 800, 700)

        self.layout = QHBoxLayout(self)
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        # Add tabs
        self.tabs.addTab(self.tab1,"Block Diagram")
        self.tabs.addTab(self.tab2,"Generated Code") 
        self.tab1.layout = QVBoxLayout(self)       
        self.tab2.layout = QVBoxLayout(self)

         # Create first tab
        self.setCentralWidget(self.tabs)
        self.expt = BlockDocument()
        self.scrollArea = QScrollArea()
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidget(self.expt)
        self.scrollArea.setWidgetResizable(True)
        self.tab1.layout.addWidget(self.scrollArea)#.scrollArea)
        self.tab1.setLayout(self.tab1.layout)

        self.setupEditor()
        self.tab2.layout.addWidget(self.editor)
        self.tab2.setLayout(self.tab2.layout)
    

        self.setupParameterTreeViewer()
        self.setupDirTreeViewer()
        self.setUpMenuBar()
        self.expt.initUI()
        self.show()
        
# Some demo blocks
  #      self.expt.addBlock(HighLevelBlocks.WhileLessThanSomeTime())
#        self.expt.addBlock(HighLevelBlocks.XRD_Mode_Block())
#        self.expt.addBlock(HighLevelBlocks.Simple_XRD_CT_Block())
#        self.expt.restyle(1)

#        self.expt.addBlock(MidLevelBlocks.ForBlock())
#        self.expt.addBlock(HighLevelBlocks.RelMotorMoveBlock())
#        self.expt.addBlock(HighLevelBlocks.RelMotorMoveBlock())
#        self.expt.addBlock(MidLevelBlocks.ForBlock())
#        self.expt.addBlock(HighLevelBlocks.RelMotorMoveBlock())
#        self.expt.addBlock(HighLevelBlocks.RelMotorMoveBlock())


    def setupParameterTreeViewer(self):
        font = QFont()
        font.setFamily('Segoe UI')
        font.setFixedPitch(True)
        font.setPointSize(10)
        self.parameterTree = ParameterTree()
#        self.parameterTree.changeEvent(self.writeCode())
        self.parameterTree.setFont(font)
        self.dock = QDockWidget("Property Editor", self)
        self.dock.setWidget(self.parameterTree)
        self.dock.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.setLayout(self.layout)
        self.expt.overObjectSignal.connect(self.updatePropertyEditor)
        
    def setupEditor(self):
        font = QFont()
        font.setFamily('Courier')
        font.setFixedPitch(True)
        font.setPointSize(11)
        self.editor = QTextEdit()
        self.editor.setFont(font)
        tabStop = 4;
        metrics = QFontMetrics(font)
        self.editor.setTabStopWidth(tabStop * metrics.width(' '))
        self.highlighter = specSyntax.Highlighter(self.editor.document())
        
    def setupDirTreeViewer(self):
        from PyQt5.QtGui import QTreeView, QFileSystemModel
 #       from PyQt5.QtCore import QStringList
        self.dirTreeView = QTreeView()
        fileSystemModel = QFileSystemModel(self.dirTreeView)
        fileSystemModel.setReadOnly(False)
        self.dirTreeView.setModel(fileSystemModel)
        folder = "."
        filters = ['*.blox', '*.mac']
        fileSystemModel.setNameFilters(filters)
        self.dirTreeView.setRootIndex(fileSystemModel.setRootPath(folder))

        self.dock2 = QDockWidget("Directory browser", self)
        self.dock2.setWidget(self.dirTreeView)
        self.dock2.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock2)
        self.dirTreeView.setDragEnabled(True)
        self.dirTreeView.show()
    
    def updatePropertyEditor(self):
        pass

        if self.expt.visibleBlocks.size == 0:
            return
        i = self.expt.selectedBlock
        p = self.expt.visibleBlocks[i].parameters

         # this is not working and probably should not go here
        p.sigTreeStateChanged.connect(self.updateEverywhere)

        self.parameterTree.setParameters(p)
        self.writeCode()
        
    def updateEverywhere(self):
        self.expt.update()
        self.writeCode()
        
    def writeCode(self):
        if (self.expt.blocks.size == 0):
            return
        self.editor.clear()
        code, null = self.expt.blockTree.getCode(0,0)
        self.editor.append(code)
        self.editor.insertPlainText('\'\n')
#        self.editor.append('\'\n')

    def setUpMenuBar(self):
        bar = self.menuBar()
        file = bar.addMenu("File")
        addModes = bar.addMenu("Modes")
        addScans = bar.addMenu("Scans")
        addMotors = bar.addMenu("Motors")
        addLoops = bar.addMenu("Loops")
        addOthers = bar.addMenu("Others")
        viewOptions = bar.addMenu("View")
        
        action_New = file.addAction("New")
        action_Save = file.addAction("Save")
        action_Import = file.addAction("Import")
        action_Export = file.addAction("Export")
        action_Quit = file.addAction("Quit")

        action_ABS_Mode = addModes.addAction("ABS Mode")
        action_XRD_Mode =  addModes.addAction("XRD Mode")        
        action_PDF_Mode = addModes.addAction("PDF Mode")
        action_Single_XRD = addScans.addAction("single XRD")
        action_INTER = addScans.addAction("interlaced XRD-CT")
        action_XRDCT = addScans.addAction("simple XRD-CT")
        action_rel = addMotors.addAction("rel move")
        action_abs = addMotors.addAction("abs move")
        action_for = addLoops.addAction("for")
        action_while = addLoops.addAction("while < time")
        action_while2 = addLoops.addAction("while < T degC")
        action_addTimer = addOthers.addAction("add timer")
        action_addSingleEuro = addOthers.addAction("single euro")
        action_addDualEuro = addOthers.addAction("dual euro")
        action_sleep = addOthers.addAction("sleep")
        
        action_StandardView =  viewOptions.addAction(self.expt.blockStyles[0].name)        
        action_BabyView =  viewOptions.addAction(self.expt.blockStyles[1].name)        

        action_New.triggered.connect(lambda: self.newBlockDocument())
        action_Save.triggered.connect(lambda: self.saveStuff())
        action_Import.triggered.connect(lambda: self.importStuff())
        action_Export.triggered.connect(lambda: self.exportStuff())
        action_Quit.triggered.connect(lambda: sys.exit(0))
        action_ABS_Mode.triggered.connect(lambda: self.expt.addBlock(HighLevelBlocks.AbsModeBlock()))
        action_XRD_Mode.triggered.connect(lambda: self.expt.addBlock(HighLevelBlocks.XrdModeBlock()))
        action_Single_XRD.triggered.connect(lambda: self.expt.addBlock(HighLevelBlocks.OneShotBlock()))
        action_INTER.triggered.connect(lambda: self.expt.addBlock(HighLevelBlocks.Interlaced_XRD_CT_Block()))
        action_XRDCT.triggered.connect(lambda: self.expt.addBlock(HighLevelBlocks.Simple_XRD_CT_Block()))
        action_rel.triggered.connect(lambda: self.expt.addBlock(HighLevelBlocks.RelMotorMoveBlock()))
        action_abs.triggered.connect(lambda: self.expt.addBlock(HighLevelBlocks.AbsMotorMoveBlock()))
        action_for.triggered.connect(lambda: self.expt.addBlock(MidLevelBlocks.SimpleForBlock()))
        action_while.triggered.connect(lambda: self.expt.addBlock(HighLevelBlocks.WhileLessThanSomeTime()))
        action_while2.triggered.connect(lambda: self.expt.addBlock(HighLevelBlocks.WhileLessThanSomeTemp()))
        action_addSingleEuro.triggered.connect(lambda: self.expt.addBlock(HighLevelBlocks.SingleEurothermBlock()))        
        action_addDualEuro.triggered.connect(lambda: self.expt.addBlock(HighLevelBlocks.DualEurothermBlock()))
        action_sleep.triggered.connect(lambda: self.expt.addBlock(MidLevelBlocks.SleepBlock()))
        action_StandardView.triggered.connect(lambda: self.expt.restyle(0))
        action_BabyView.triggered.connect(lambda: self.expt.restyle(1))

    def newBlockDocument(self):
        # this is really bad way I think better to call self.expt.clear() and
        # do the work there
        del self.expt
        self.expt = BlockDocument()
        self.scrollArea.setWidget(self.expt)
        
    def importStuff(self):
        dlg = QFileDialog()
        fileName = dlg.getOpenFileName(self, 'Import blocks', filter = '*.blox')
        if len(fileName[0]) ==  0:
            return
        self.expt.importBlocks(fileName[0])
       
    def exportStuff(self):
        dlg = QFileDialog()
        fileName = dlg.getSaveFileName(self, 'Export blocks', filter = '*.blox')
        if len(fileName[0]) ==  0:
            return
        self.expt.exportBlocks(fileName[0])

    def saveStuff(self):
        dlg = QFileDialog()
        fileName = dlg.getSaveFileName(self, 'Save File', 'id15_user_mac.mac', filter = '*.mac')
        print(fileName)
        file = open(fileName[0],'w')
        print(file)
        self.writeCode()
        text = self.editor.toPlainText()
        file.write(text)
        file.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = BlockMainWindow()
    w.show()
    sys.exit(app.exec_())   

#w = BlockMainWindow()
#w.show()