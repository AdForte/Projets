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
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
"""
# =============================================================================
# importer les données dans data
# =============================================================================
"""
data = pd.read_csv('titanic_train.csv', sep = ',')
data2 = data
data = pd.DataFrame(data)
testeur = pd.read_csv('titanic_test.csv', sep = ',')
"""
# =============================================================================
# arranger les données
# =============================================================================
"""
#print(data.groupby('Sex').mean())

data['Sex'].replace('female', 1, inplace=True)
data['Sex'].replace('male', 0, inplace=True)
testeur['Sex'].replace('female', 1, inplace=True)
testeur['Sex'].replace('male', 0, inplace=True)
data.drop(['Cabin', 'Ticket', 'Embarked'], axis= 1, inplace= True)
testeur.drop(['Cabin', 'Ticket', 'Embarked'], axis= 1, inplace= True)
data =  data.fillna(data['Fare'].median(),axis = 0)
data =  data.fillna(data['Age'].median(),axis = 0)
data2 = data
testeur['Fare'] = testeur['Fare'].fillna(testeur['Fare'].median(),axis = 0)
testeur['Age'] = testeur['Age'].fillna(testeur['Age'].median(),axis = 0)
#print(data2)
    #print(data.index())
    #print(itm)
#print(data['Sex'])
    
dataX_train = data2[['Age','Sex','Pclass','Fare','SibSp']] 
dataY_train = data2['Survived']
dataX_test = testeur[['Age','Sex','Pclass','Fare','SibSp']]

"""
# =============================================================================
# classifiers# =============================================================================
"""

"""logistic reg"""
data_logistic = linear_model.LogisticRegression()
data_logistic.fit(dataX_train, dataY_train)
data_test_lr = data_logistic.predict(dataX_test)
data_test_lr = pd.DataFrame(data_test_lr)
#print(data_logistic.score(dataX_train, dataY_train))
#print(data_test_lr)
#print(testeur['PassengerId'], data_test_lr)
a = testeur['PassengerId'].tolist()
#print(a)
#df_lr = pd.DataFrame({'PassengerId': a , 'Survived': data_test_lr[0].tolist()})
#df_lr.set_index('PassengerId')
#df_lr.to_csv('gender_submission.csv', index = None)

#print(data_test_lr.score())

"""Bayesien Naïf"""
gnb = GaussianNB()
data_NBayes = gnb.fit(dataX_train, dataY_train).predict(dataX_test)
#print(gnb.score(dataX_train, dataY_train))
"""Arbre machin"""

data_tree = tree.DecisionTreeClassifier(max_leaf_nodes = 35)

data_DTC = data_tree.fit(dataX_train, dataY_train).predict(dataX_test)
print(data_tree.score(dataX_train, dataY_train), data_DTC)
df_lr = pd.DataFrame({'PassengerId': a , 'Survived': data_DTC})
df_lr.set_index('PassengerId')
df_lr.to_csv('gender_submission.csv', index = None)


"""
# =============================================================================
# Cimetière des functions, prints etc...:

#print(data.loc[:,'Sex'])
#a = data.loc[40:50,'Sex']
#print('a:', a)
#data[['Pclass','Survived','Sex']].hist()
plt.scatter( data['Age'], data['Survived'])
plt.show()

print(data.groupby('Sex').mean()) :
    
        PassengerId  Survived    Pclass        Age     SibSp     Parch  \
Sex                                                                      
female   431.028662  0.742038  2.159236  27.915709  0.694268  0.649682   
male     454.147314  0.188908  2.389948  30.726645  0.429809  0.235702   

             Fare  
Sex                
female  44.479818  
male    25.523893  

data2 = data.dropna(axis = 0, how= 'any')  on vire les NaN









# =============================================================================
"""






