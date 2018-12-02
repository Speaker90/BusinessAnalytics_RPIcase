import pandas as pd
import sqlite3
import re
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense,Dropout

conn = None


conn = sqlite3.connect("../db/Bugs.db")
df = pd.read_sql_query("SELECT Bugs.*, Assignees.SuccessAssignee, Reporters.SuccessReporter FROM Bugs INNER JOIN Assignees ON Bugs.AssigneeID = Assignees.AssigneeID INNER JOIN Reporters ON Bugs.ReporterID = Reporters.ReporterID;", conn)

#print(df)
conn.close

N=df.shape[0]
labels = np.zeros((N,1))
features = np.zeros((N,7))

def os_replace(dict_os, text):

    regex = re.compile("(%s)" % "|".join(map(re.escape, dict_os.keys())))

    if regex.search(text):
        ret = regex.search(text)
        return dict_os[ret.group()]
    else:
        return 0

dict_fixed = {"FIXED": 1}
labels[:,0]=df['Outcome'].map(dict_fixed).fillna(0)

features[:,0]=df['Closing']-df['Opening']

dict_priority = {"P1": 1,"P2": 2,"P3": 3,"P4": 4,"P5": 5}
features[:,1]=df['Priority'].map(dict_priority).fillna(0)

features[:,2]=df['CC']

products = df["Product"].unique()
dict_product = dict(zip(products, range(len(products))))
features[:,3]=df['Product'].map(dict_product)

dict_os = {"Windows": 1, "Mac": 2, "Linux": 3, "Unix": 4, "Solaris": 5, "All": 6}

for i,row in df.iterrows():
    features[i,4]=os_replace(dict_os, row['OS'])

features[:,5] = df['SuccessAssignee']
features[:,6] = df['SuccessReporter']


print(features)
features_norm=normalize(features,axis=0)
print(features_norm)
print(np.sum(labels)/N)

x_train, x_test, y_train, y_test = train_test_split(features_norm, labels, test_size=0.5, random_state=29)
x_cv, x_test, y_cv, y_test = train_test_split(x_test,y_test, test_size=0.5, random_state=29)

print(x_train.shape)
print(x_cv.shape)

y_train=y_train.flatten()
y_cv = y_cv.flatten()
y_test = y_test.flatten()


NB = GaussianNB().fit(x_train, y_train)
print(NB.score(x_cv,y_cv))
prob = NB.predict_proba(x_cv)[:,1]

fpr, tpr, thresholds = roc_curve(y_cv, prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold (AUC = %0.2f)' % roc_auc)
LM = LogisticRegression(random_state=29,solver='lbfgs').fit(x_train,y_train)
print(LM.score(x_cv,y_cv))

prob = LM.predict_proba(x_cv)[:,1]

fpr, tpr, thresholds = roc_curve(y_cv, prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold (AUC = %0.2f)' % roc_auc)
#DT = DecisionTreeClassifier(random_state=29).fit(x_train, y_train)
#print(DT.score(x_cv, y_cv))

RF = RandomForestClassifier(n_estimators=1000, max_depth=30, random_state=29).fit(x_train, y_train)
print(RF.score(x_cv,y_cv))

GB = GradientBoostingClassifier(n_estimators= 1000, max_depth= 5, random_state= 29,
                   ).fit(x_train,y_train)
print(GB.score(x_cv,y_cv))

SVM = SVC(gamma='auto', probability=True).fit(x_train,y_train)
print(SVM.score(x_cv,y_cv))

model = Sequential()
model.add(Dense(128, input_dim=7, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
print(model.evaluate(x_cv, y_cv, batch_size=128))
plt.show()
