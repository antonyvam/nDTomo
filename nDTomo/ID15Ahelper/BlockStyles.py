# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:07:11 2017

@author: simon
"""

    
class BlockStyleStandard():
    def __init__(self):
        self.name = 'standard'
        self.topMargin = 50
        self.blockSpacing = 70
        self.height = 40
        self.width = 150
        self.fontSize = 10


class BlockStyleBaby():
    def __init__(self):
        self.name = 'baby'
        self.topMargin = 50
        self.blockSpacing = 50
        self.height = 30
        self.width = 120
        self.fontSize = 8