import sys
import sqlite3
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.metrics import roc_curve, auc
from numpy.random import seed
seed(29)
import random as rn
rn.seed(29)
from tensorflow import set_random_seed
set_random_seed(29)

plt.rcParams.update({'font.size': 10})

def FromDBtoDF(dbPath,query):
    """This functions uses the supplied query to retrieve data from a DB and stores it in a pandas dataframe."""

    #initialize connection
    conn = None

    try:
        conn = sqlite3.connect(dbPath)
        df = pd.read_sql_query(query, conn)

    except sqlite3.Error, errorOuput:
        print("Error %s:" % errorOuput.args[0])
        sys.exit(1)

    finally:
        if conn:
            conn.close()

    return df


def FeatureEncoder(df,header,dict,exhaustive):
    """This function encodes the values of a df-header according to a dictionary. If exhaustive is set false, then values not in the dictionary are set to 0."""

    if exhaustive:
        encodedValues = df[header].map(dict)
    else:
        encodedValues = df[header].map(dict).fillna(0)

    return encodedValues


def FuzzyFeatureEncoder(df,header,dict):
    """This function encodes the values of a df-header according to a fuzzy dictionary. Sets undefined values to 0."""

    encodedValues = np.zeros((df.shape[0],1))
    #oop through the rows and match the dict keys
    for i,row in df.iterrows():
        text = row[header]
        #create a regex from the dict
        regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

        #return the dict value for the first match
        if regex.search(text):
            ret = regex.search(text)
            encodedValues[i] = dict[ret.group()]
        else:
            encodedValues[i] = 0

    return encodedValues.flatten()


def PlotFeatureStatistics(features,labels):
    """This function plots the boxplots and the correlations of the features and the labels"""

    #define the columns and create a dataframe for plotting
    columns = ['OpenTime','Assignments', 'CC','Product', 'OS', 'SuccessAssignee','SuccessReporter','Component','Social','Equal']
    index = np.arange(features.shape[0])
    features_df = pd.DataFrame(data = features, index = index, columns = columns)
    features_df['Fixed'] = labels
    columns.append("Fixed")

    #plot the boxplots
    features_df.plot(kind="box", subplots= True, layout=(4,3), sharex=False, sharey=False, figsize=(15,9))
    plt.savefig('../docs/Latex/pictures/boxplots.png', bbox_inches='tight')
    plt.savefig('../results/boxplots.png', bbox_inches='tight')
    plt.show()

    #plot the correlation matrix
    correlations = features_df.corr()
    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations,  vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(columns),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(columns,rotation=45)
    ax.set_yticklabels(columns)

    #add correlation values
    correlations = np.array(correlations)
    correlations = np.round(correlations,2)
    for i in range(correlations.shape[0]):
        for j in range(correlations.shape[1]):
            text = ax.text(j, i, correlations[i, j],
                           ha="center", va="center", color="w")

    rect = ptch.Rectangle((-0.5,9.5), 11, 1,edgecolor='red',facecolor='none',linewidth=3)
    ax.add_patch(rect)
    fig.savefig('../docs/Latex/pictures/correlations.png',bbox_inches='tight')
    fig.savefig('../results/correlations.png',bbox_inches='tight')
    plt.show()


def SetUpNeuralNetwork(featureCount):
    """This function sets up simple neural network for a binary classification problem."""

    model = Sequential()
    model.add(Dense(64, input_dim=featureCount, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def CalculateAccuracy(models,x,y):
    """This function calculates the accuracy for all the models supplied in the models list."""

    for i in range(0,len(models)-1):
        models[i].append(models[i][1].score(x,y))
    models[5].append(models[5][1].evaluate(x, y, batch_size=128,verbose=0)[1])

    return models


def PrintAccuracy(title,models):
    """This functions prints the accuracy of the models in a nice table."""

    dash = '_' * 40
    print("\n"+title)
    print(dash)
    print("{:<30s}{:<10s}".format("MODEL", "ACCURACY"))
    print(dash)
    for model in models:
        print("{:<26s} {:>10.4f}%".format(model[0],model[2]*100))


def SavePredictions(x,indices,models):
    """This function saves the predictions for each bugID of the test set to a file."""

    #add bugIDs
    predictions = np.zeros((len(indices),2))
    predictions[:,0] = indices
    path = '../results/'
    
    #save the outcom of each model in csv file
    for i in range(len(models)-1):
        predictions[:,1] = models[i][1].predict(x).flatten()
        filename=models[i][0]+'Predictions.out'
        np.savetxt((path+filename).replace(" ",""),predictions.astype(int),header = 'ACCURACY: {:.4f}%\nBugID,Prediction'.format(models[i][2]*100),delimiter=',')
    predictions[:,1] = models[5][1].predict(x).flatten()
    filename=models[5][0]+'Predictions.out'
    np.savetxt((path+filename).replace(" ",""),predictions.astype(int),header = 'ACCURACY: {:.4f}%\nBugID,Prediction'.format(models[i][2]*100),delimiter=',')


def PlotROCs(models,x,y):
    """This function plots the ROCs of all models."""

    #calculate the fpr and tpf for all models and add it to the plot
    plt.figure(figsize=(12,9))
    for i in range(0,len(models)-1):
        prob = models[i][1].predict_proba(x)[:,1]
        fpr, tpr, thresholds = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, alpha=0.3,label='{} (AUC = {:0.4f})'.format(models[i][0],roc_auc))

    prob = models[5][1].predict(x)
    fpr, tpr, thresholds = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, alpha=0.3,label='{} (AUC = {:0.4f})'.format(models[5][0],roc_auc))

    #format plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of the Models')
    plt.legend(loc="lower right")
    plt.savefig('../docs/Latex/pictures/rocs.png',bbox_inches='tight')
    plt.savefig('../results/rocs.png',bbox_inches='tight')
    plt.show()
