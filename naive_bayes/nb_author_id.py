#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
train_seconds = round(time() - t0, 3)
print "Training time = {}s".format(train_seconds)

t0 = time()
pred = clf.predict(features_test)
pred_seconds = round(time() - t0, 3)
print "Predicting time = {}s".format(pred_seconds)

# accuracy = clf.score(features_test, labels_test)
accuracy = accuracy_score(pred, labels_test)
print "Accuracy score: {}".format(accuracy)

#########################################################


