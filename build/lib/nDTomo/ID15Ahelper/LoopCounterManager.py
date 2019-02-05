# -*- coding: utf-8 -*-
"""

Loop counter manager

@author: S.D.M. Jacques

"""

# This class will need to be singleton
# see http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
class LoopCounterManager():
    
    """
    
    Loop counter manager class
    
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.allCountersList = ['ii' , 'jj', 'kk', 'mm', 'nn', 'pp', 'qq', 'rr']#, 'iii', 'jjj', 'kkk', 'mmm', 'nnn']
        self.allCounters = {k: v for v, k in enumerate(self.allCountersList)}
        self.invertedCounterDict = self.inverseMapping(self.allCounters)
        self.availableCounters = self.allCounters 
        self.usedCounters = {}
        
    def getCounterName(self, value):
        return self.invertedCounterDict[value]
            
    def inverseMapping(self, f):
        return f.__class__(map(reversed, f.items()))

    def useCounter(self, key):
        self.moveItemBetweenDictionaries(key, self.availableCounters, self.usedCounters)        
        
    def releaseCounter(self, key):
        self.moveItemBetweenDictionaries(key, self.usedCounters, self.availableCounters)        

    def moveItemBetweenDictionaries(self, key, sourceDict, targetDict):
        if not(key in sourceDict):
            print('key not in list in instance of Class LoopCounter')
            return
        value = sourceDict[key]                
        targetDict[key] = value
        del sourceDict[key]
        
##############################################################################
        
