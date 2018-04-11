# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrien_F)s
"""
# =============================================================================
# Adrien's Holy Code
# =============================================================================
print(__doc__)
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pd.read_csv('titanic_train.csv', sep = ',')
print(data)