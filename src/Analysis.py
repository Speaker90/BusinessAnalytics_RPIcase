#!/usr/local/bin/python
# Script:   Analysis.py
# Author:   Florian Spychiger
# Date:     02 December 2018
from helpers import FromDBtoDF
from helpers import FeatureEncoder
from helpers import FuzzyFeatureEncoder
from helpers import SetUpNeuralNetwork
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential

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

#the 2nd feature is the priority set by the assignee
priority_dict = {"P1": 1,"P2": 2,"P3": 3,"P4": 4,"P5": 5}
features[:,1] = FeatureEncoder(df,'Priority',priority_dict,False)

#the 3rd feature is the number of CCs
features[:,2]=df['CC']

#the 4th feature is product, here we assign each unique product a value
products = df["Product"].unique()
product_dict = dict(zip(products, range(len(products))))
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

#now, we normalize the features
features_norm=normalize(features,axis=0)

#split up data into training-crossvalidation-test sets
x_train, x_test, y_train, y_test = train_test_split(features_norm, labels, test_size=0.5, random_state=29)
x_cv, x_test, y_cv, y_test = train_test_split(x_test,y_test, test_size=0.5, random_state=29)

#flatten the labels
y_train=y_train.flatten()
y_cv = y_cv.flatten()
y_test = y_test.flatten()


######################################################################################################
#3. Training the Models 

print("[INFO] Start training the models..")

models = []

print("[INFO] Train Naive Bayes..")
NB = GaussianNB().fit(x_train, y_train)
models.append(["Naive Bayes", NB])

print("[INFO] Train Logistic Regression..")
LM = LogisticRegression(random_state=29,solver='lbfgs').fit(x_train,y_train)

print("[INFO] Train Random Forest..")
RF = RandomForestClassifier(n_estimators=1000, max_depth=30, random_state=29).fit(x_train, y_train)

print("[INFO] Train Boosting Classifier..")
GB = GradientBoostingClassifier(n_estimators= 1000, max_depth= 5, random_state= 29).fit(x_train,y_train)

print("[INFO] Train Support Vector Machine..")
SVM = SVC(gamma='auto', probability=True).fit(x_train,y_train)

print("[INFO] Train Neural Network..")
model = SetUpNeuralNetwork(features.shape[1])
NN = model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=0)


######################################################################################################
#4. Cross-validate the Models 

print("[INFO]  crossvalidation accuracy..")

models[0].extend(NB.score(x_cv,y_cv))

print("{%s} accuracy on the cross-validation set: {:.4f}%".format(models[0][0],models[0][2]*100))
#print(model.evaluate(x_cv, y_cv, batch_size=128))
#print(NB.score(x_cv,y_cv))
#prob = NB.predict_proba(x_cv)[:,1]

#fpr, tpr, thresholds = roc_curve(y_cv, prob)
#roc_auc = auc(fpr, tpr)
#plt.plot(fpr, tpr, lw=1, alpha=0.3,
#             label='ROC fold (AUC = %0.2f)' % roc_auc)
#prob = LM.predict_proba(x_cv)[:,1]

#fpr, tpr, thresholds = roc_curve(y_cv, prob)
#roc_auc = auc(fpr, tpr)
#plt.plot(fpr, tpr, lw=1, alpha=0.3,
#             label='ROC fold (AUC = %0.2f)' % roc_auc)
