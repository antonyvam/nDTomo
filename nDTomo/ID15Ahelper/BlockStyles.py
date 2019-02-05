# -*- coding: utf-8 -*-
"""

Block styles

@author: S.D.M. Jacques

"""

    
class BlockStyleStandard():
    def __init__(self):
        self.name = 'standard'
        self.topMargin = 50
        self.blockSpacing = 60#70
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