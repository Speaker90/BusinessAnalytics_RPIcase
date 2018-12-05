import sys
import sqlite3
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.metrics import roc_curve, auc

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
    """Encodes the values of a df-header according to a dictionary. If exhaustive is set false, then values not in the dictionary are set to 0."""

    if exhaustive:
        encodedValues = df[header].map(dict)
    else:
        encodedValues = df[header].map(dict).fillna(0)

    return encodedValues


def FuzzyFeatureEncoder(df,header,dict):
    """Encodes the values of a df-header according to a fuzzy dictionary. Sets undefined values to 0."""

    encodedValues = np.zeros((df.shape[0],1))
    #loop through the rows and match the dict keys
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


def PlotROCs(models,x,y):
    """This function plots the ROCs of all models."""

    #calculate the fpr and tpf for all models and add it to the plot
    for i in range(0,len(models)-1):
        prob = models[i][1].predict_proba(x)[:,1]
        fpr, tpr, thresholds = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, alpha=0.3,label='{:<25s} {:>7s}{:0.4f})'.format(models[i][0],'(AUC = ',roc_auc))

    prob = models[5][1].predict(x)
    fpr, tpr, thresholds = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, alpha=0.3,label='{:<25s} {:>7s}{:0.4f})'.format(models[5][0],'(AUC = ',roc_auc))

    #format plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of the Models')
    plt.legend(loc="lower right")
    plt.show()
