# -*- coding: utf-8 -*-
"""

The main ID15A helper GUI code

@author: S.D.M. Jacques

"""
#from PyQt5 import QtWidgets, QtGui, QtCore
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QTextEdit, QAction 
from PyQt5.QtWidgets import QVBoxLayout, QDockWidget, QTabWidget, QScrollArea, QMessageBox
from PyQt5.QtGui import QFont, QFontMetrics, QPixmap, QIcon#, QTextCursor
from PyQt5.QtCore import Qt, qVersion, PYQT_VERSION_STR, QSize


import specSyntax
import ModeBlocks, ScanBlocks, MotorBlocks, LoopBlocks, OtherBlocks


from BlockDocument import BlockDocument
from pyqtgraph.parametertree import ParameterTree


class BlockMainWindow(QMainWindow):
    
    """
    
    The ID15A helper GUI
    
    """
    
    def __init__(self):
        QMainWindow.__init__(self) #####
#####        super(BlockMainWindow, self).__init__()
        
#        layout = QHBoxLayout()        
        self.setWindowTitle('ID15A helper')
        
#        self.setWindowModality(Qt.ApplicationModal)
        self.setGeometry(200, 200, 800, 700)

 #       self.layout = QHBoxLayout(self)
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
        self.expt.trigger.connect(self.updateEverywhere)
        self.show()

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
#        self.setLayout(self.layout)
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
        helpMe = bar.addMenu("Help")
        
        action_New = file.addAction("New")
        action_Save = QAction(QIcon('.\\nDTomo\\ID15Ahelper\\images\\save-button-icon.png'), "Save", self)
#        action_SaveAs = file.addAction("Save As")
        file.addAction(action_Save)
        
        action_Import = QAction(QIcon('.\\nDTomo\\ID15Ahelper\\images\\import.png'), "Import", self)
        action_Export = QAction(QIcon('.\\nDTomo\\ID15Ahelper\\images\\export.png'), "Export", self)
        file.addAction(action_Import)
        file.addAction(action_Export)
        action_Quit = file.addAction("Quit")

        action_ABS_Mode = addModes.addAction("ABS Mode")
        action_XRD_Mode =  addModes.addAction("XRD Mode")        
        action_PDF_Mode = addModes.addAction("PDF Mode")
		
        action_Single_XRD = addScans.addAction("point XRD")
        action_HORSCAN = addScans.addAction("hor XRD linescan")
        action_VERSCAN = addScans.addAction("ver XRD linescan")
        action_XRDMAP = addScans.addAction("XRD Map")
        action_XRDCT = addScans.addAction("single XRD-CT")
        action_XRDCT3D = addScans.addAction("3D-XRD-CT")
        action_FASTXRDCT = addScans.addAction("Fast XRD-CT")
        action_FASTXRDCT3D = addScans.addAction("Fast 3D-XRD-CT")
        action_INTER = addScans.addAction("Interlaced XRD-CT")
        action_ABSCT = addScans.addAction("ABS-CT")
        
        action_rel = addMotors.addAction("rel move")
        action_abs = addMotors.addAction("abs move")
        action_for = addLoops.addAction("for")
        action_while = addLoops.addAction("while < time")
        action_while2 = addLoops.addAction("while Temp ramp")
        action_positionalSeries = addLoops.addAction("positionalSeries") 
        action_addheatsys = addOthers.addAction("heating system")        
#        action_addSingleEuro = addOthers.addAction("single suro")
#        action_addDualEuro = addOthers.addAction("dual euro")
        action_sleep = addOthers.addAction("sleep")
        action_wait = addOthers.addAction("wait for user")
#        action_group = addOthers.addAction("group")       

        action_StandardView =  viewOptions.addAction(self.expt.blockStyles[0].name)        
        action_BabyView =  viewOptions.addAction(self.expt.blockStyles[1].name)        
#        action_AboutView =  viewOptions.addAction('About')        

#        action_Support =  helpMe.addAction('Support')        
        action_About =  helpMe.addAction('About')        

        action_New.triggered.connect(lambda: self.newBlockDocument())
        action_Save.triggered.connect(lambda: self.saveStuff())
        action_Import.triggered.connect(lambda: self.importStuff())
        action_Export.triggered.connect(lambda: self.exportStuff())
        action_Quit.triggered.connect(lambda: sys.exit(0))
        
        action_ABS_Mode.triggered.connect(lambda: self.expt.addBlock(ModeBlocks.AbsModeBlock()))
        action_XRD_Mode.triggered.connect(lambda: self.expt.addBlock(ModeBlocks.XrdModeBlock()))
        action_PDF_Mode.triggered.connect(lambda: self.expt.addBlock(ModeBlocks.PdfModeBlock()))
		
        action_Single_XRD.triggered.connect(lambda: self.expt.addBlock(ScanBlocks.OneShotBlock()))
        action_HORSCAN.triggered.connect(lambda: self.expt.addBlock(ScanBlocks.HORSCAN_Block()))
        action_VERSCAN.triggered.connect(lambda: self.expt.addBlock(ScanBlocks.VERSCAN_Block()))
        action_XRDMAP.triggered.connect(lambda: self.expt.addBlock(ScanBlocks.XRDMAP_Block()))
        action_INTER.triggered.connect(lambda: self.expt.addBlock(ScanBlocks.InterlacedXRDCT_Block()))
        action_XRDCT.triggered.connect(lambda: self.expt.addBlock(ScanBlocks.XRDCT_Block()))
        action_XRDCT3D.triggered.connect(lambda: self.expt.addBlock(ScanBlocks.XRDCT3D_Block()))
        action_FASTXRDCT.triggered.connect(lambda: self.expt.addBlock(ScanBlocks.FASTXRDCT_Block()))
        action_FASTXRDCT3D.triggered.connect(lambda: self.expt.addBlock(ScanBlocks.FASTXRDCT3D_Block()))
        action_ABSCT.triggered.connect(lambda: self.expt.addBlock(ScanBlocks.ABSCT_Block()))
        
        action_rel.triggered.connect(lambda: self.expt.addBlock(MotorBlocks.RelMotorMoveBlock()))
        action_abs.triggered.connect(lambda: self.expt.addBlock(MotorBlocks.AbsMotorMoveBlock()))
        
        action_for.triggered.connect(lambda: self.expt.addBlock(LoopBlocks.SimpleForBlock()))
        action_while.triggered.connect(lambda: self.expt.addBlock(LoopBlocks.WhileLessThanSomeTime()))
        action_while2.triggered.connect(lambda: self.expt.addBlock(LoopBlocks.WhileTempRamp()))
        action_positionalSeries.triggered.connect(lambda: self.expt.addBlock(LoopBlocks.PositionalSeriesBlock()))
        
        action_addheatsys.triggered.connect(lambda: self.expt.addBlock(OtherBlocks.HeatingSystemBlock()))        
#        action_addSingleEuro.triggered.connect(lambda: self.expt.addBlock(OtherBlocks.SingleEurothermBlock()))        
#        action_addDualEuro.triggered.connect(lambda: self.expt.addBlock(OtherBlocks.DualEurothermBlock()))
        action_sleep.triggered.connect(lambda: self.expt.addBlock(OtherBlocks.SleepBlock()))
        action_wait.triggered.connect(lambda: self.expt.addBlock(OtherBlocks.WaitForUserBlock()))
#        action_group.triggered.connect(lambda: self.expt.addBlock(OtherBlocks.GenericGroupBlock()))
        action_StandardView.triggered.connect(lambda: self.expt.restyle(0))
        action_BabyView.triggered.connect(lambda: self.expt.restyle(1))
#        action_AboutView.triggered.connect(lambda: self.about())        
##        action_Support.triggered.connect()        
        action_About.triggered.connect(lambda: self.about())        

        self.toolbar = self.addToolBar('hello')
#        self.toolbar.addAction(action_Import)
        self.toolbar.addAction(action_Export)
        self.toolbar.addAction(action_Save)
        self.toolbar.setIconSize(QSize(32,32))
        
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
        try:
            if len(fileName)>0:
                print(fileName)
                file = open(fileName[0],'w')
                print(file)
                self.writeCode()
                text = self.editor.toPlainText()
                file.write(text)
                file.close()
        except:
            print('Select a directory to save the macro')
    
    def about(self):
        message = '<b>ID15A helper v0.1.0<p>'
        message += '<p><i>Created by <a href=www.finden.co.uk>Finden</a>. Running under license under GPLv3'
        message += '\t '
        sImage = QPixmap(".//images//logoLetters.png")
        d = QMessageBox()
        d.setWindowTitle('About')
        d.setIconPixmap(sImage)
        d.setText(message)
        d.exec_()
        
	
def main():
    qApp = QApplication(sys.argv)
    w = BlockMainWindow()
    w.show()
    sys.exit(qApp.exec_())
    qApp.exec_()
   
if __name__ == "__main__":
    main()
    
    
#aw = BlockMainWindow()    
#aw.show()    
    