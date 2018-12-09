#!/usr/local/bin/python
# Script:   Analysis.py
# Author:   Florian Spychiger
# Date:     02 December 2018

from helpers import FromDBtoDF
from helpers import FeatureEncoder
from helpers import FuzzyFeatureEncoder
from helpers import PlotFeatureStatistics
from helpers import SetUpNeuralNetwork
from helpers import CalculateAccuracy
from helpers import PrintAccuracy
from helpers import SavePredictions
from helpers import PlotROCs
import argparse
import os
import sys
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from numpy.random import seed
seed(29)
import random as rn
rn.seed(29)
from tensorflow import set_random_seed
set_random_seed(29)

######################################################################################################
#Preparation

#disable annoying tensorflow AVX/FMA warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#define argument parser and evaluate optional arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--calibration",action='store_true',help="enable calibration mode")
ap.add_argument("-p", "--plotting",action='store_true',help="plot feature statistics")
args = ap.parse_args()
if args.calibration:
    print("---------------------> CALIBRATION MODE TURNED ON, USING CROSS-VALIDATION SET")

if args.plotting:
    print("---------------------> PLOTTING STATISTICS ENABLED")

######################################################################################################
#1. Data import

print("[INFO] Importing data..")

#define db path and query to fetch the data
dbPath = "../db/Bugs.db"
query = "SELECT Bugs.*, Assignees.SuccessAssignee, Reporters.SuccessReporter FROM Bugs INNER JOIN Assignees ON Bugs.AssigneeID = Assignees.AssigneeID INNER JOIN Reporters ON Bugs.ReporterID = Reporters.ReporterID;"

#store the data in a dataframe
df = FromDBtoDF(dbPath,query)


######################################################################################################
#2. Feature Construction 

print("[INFO] Constructing features..")

#initialize the labels and the features
N=df.shape[0]
labels = np.zeros((N,1))
features = np.zeros((N,10))

#encode the labels (1,0), where we set fixed=1
outcome_dict = {"FIXED": 1}
labels[:,0] = FeatureEncoder(df,'Outcome',outcome_dict,False)

#the first feature is the time a bug has been open
features[:,0]=df['Closing']-df['Opening']

#the 2nd feature is the number of assignments
features[:,1] = df['Assignments']

#the 3rd feature is the number of CCs
features[:,2]=df['CC']

#the 4th feature is product, here we assign each unique product a value
products = df["Product"].unique()
product_dict = dict(zip(products, range(len(products))))
print(product_dict)
features[:,3] = FeatureEncoder(df,'Product',product_dict,True)

#the 5th feature is the main operating system, we use a fuzzy encoder
os_dict = {"Windows": 1, "Mac": 2, "Linux": 3, "Unix": 4, "Solaris": 5, "All": 6}
features[:,4]=FuzzyFeatureEncoder(df,'OS',os_dict)

#the 6th and 7th features are the successrates
features[:,5] = df['SuccessAssignee']
features[:,6] = df['SuccessReporter']

#the 8th feature is the latest component affected
components = df["Component"].unique()
component_dict = dict(zip(components, range(len(components))))
features[:,7] = FeatureEncoder(df,'Component',component_dict,True)

#the 9th feature is the social relationship between assignee and reporter
features[:,8]=df['Social']

#the 10th feature is an indicator whether the reporter fixes the bug himself
features[:,9]=np.where(df["ReporterID"]==df["AssigneeID"],1,0)


######################################################################################################
#3. Plotting feature statistcs 

if args.plotting:
    print("[INFO] Plotting statistics of features..")

    PlotFeatureStatistics(features,labels)


######################################################################################################
#4. Training the Models 

print("[INFO] Training the models..")

#now, we standardize the features
std_scale = StandardScaler().fit(features)
features_std = std_scale.transform(features)

#split up data into training-crossvalidation-test sets
indices = np.arange(features_std.shape[0])
x_train, x_temp, y_train, y_temp, indices_train, indices_temp = train_test_split(features_std, labels, indices, test_size=0.5, random_state=29)
x_cv, x_test, y_cv, y_test, indices_cv, indices_test = train_test_split(x_temp,y_temp, indices_temp, test_size=0.5, random_state=29)

#flatten the labels
y_train=y_train.flatten()
y_cv = y_cv.flatten()
y_test = y_test.flatten()

models = []

print("\t1. Training Naive Bayes..")
NB = GaussianNB().fit(x_train, y_train)
models.append(["Naive Bayes", NB])

print("\t2. Training Logistic Regression..")
LR = LogisticRegression(random_state=29,solver='lbfgs',C=1).fit(x_train,y_train)
models.append(["Logistic Regression", LR])

print("\t3. Training Random Forest..")
RF = RandomForestClassifier(n_estimators=1000, max_depth=15, random_state=29).fit(x_train, y_train)
models.append(["Random Forest", RF])

print("\t4. Training Boosting Classifier..")
GB = GradientBoostingClassifier(n_estimators= 1000, max_depth= 6, random_state= 29).fit(x_train,y_train)
models.append(["Boosting Classifier", GB])

print("\t5. Training Support Vector Machine..")
SVM = SVC(gamma='auto', probability=True, C=0.8).fit(x_train,y_train)
models.append(["Support Vector Machine", SVM])

print("\t6. Training Neural Network..")
NN = SetUpNeuralNetwork(features.shape[1])
NN.fit(x_train, y_train, epochs=50, batch_size=128, verbose=0)
models.append(["Neural Network", NN])


######################################################################################################
#5. Cross-validate the Models 

if args.calibration:
    print("[INFO]  Crossvalidation accuracy..")

    #calcuate accuracy
    models = CalculateAccuracy(models,x_cv,y_cv)

    #print accuracy
    PrintAccuracy('Crossvalidation Set',models)

    #we finish here
    sys.exit()


######################################################################################################
#6. Evaluate the Models 

print("[INFO]  Test set accuracy..")

#calcuate accuracy
models = CalculateAccuracy(models,x_test,y_test)

#print accuracy
PrintAccuracy('Test Set',models)

#save predictions for each model
SavePredictions(x_test,indices_test,models)

#plot the roc-curves
PlotROCs(models,x_test,y_test)
