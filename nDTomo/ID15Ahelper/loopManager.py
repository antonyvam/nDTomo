# -*- coding: utf-8 -*-
"""

Highlevel blocks

@author: S.D.M. Jacques

"""

# This class will need to be singleton
# see http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
class LoopCounters():
    
    """
    
    A loop counter class
    
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.counterList = ['none', 'ii' , 'jj', 'kk', 'mm', 'nn', 'pp', 'qq', 'rr', 'iii', 'jjj', 'kkk', 'mmm', 'nnn']
        self.counterDict = {k: v for v, k in enumerate(loopCounterList)}
#        self.availableCountersList = self.counterList
        self.availableCountersDict = self.counterDict
#        self.usedCounterList = []
        self.usedCounterDict = {}

    def useCounter(self, key):
        value = self.counterDict[key]
        self.usedCounterDict[key] = value
        #        self.availableCountersList.remove(key)
        del self.availableCountersDict[key]
        
    def releaseCounter(self, key):
        value = self.counterDict[key]
        self.usedCounterDict[key] = value
#        self.availableCountersList.remove(key)
        del self.availableCountersDict[key]

    def moveBetweenDictionaries(key, sourceDict, targetDict):
        value = sourceDict[key]                
        targetDict[key] = value
        del sourceDict[key]
        
