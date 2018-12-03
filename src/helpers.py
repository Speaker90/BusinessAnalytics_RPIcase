import sys
import sqlite3
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD

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

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metric=['accuracy'])

    return model
