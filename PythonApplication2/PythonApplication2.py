import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from tqdm import tqdm
from joblib import dump, load
import json


"""

list_dir = os.listdir("./Samples")
print(list_dir)

templates = []

for dir in list_dir:
    file_list = os.listdir("./Samples/" + dir + "")
    for f in file_list:
        ts = {"id": [], "time": [], "m_x": [], "m_y": []}
        sample = np.array(np.loadtxt("Samples/" + dir + "/" + f + "", dtype=float))
        for temp in sample:
            ts["m_x"].append(temp[5])
            ts["m_y"].append(temp[6])
            ts["time"].append(temp[0])
            ts["id"].append(dir)
        if len(sample) > 0:
            templates.append(pd.DataFrame(ts))

data = {"label": [], "template": []}

ctr = 0
for elem in templates:
    extracted_features = impute(
        extract_features(
            elem,
            column_id="id",
            column_sort="time",
            n_jobs=1,
            show_warnings=False,
            disable_progressbar=False,
            profile=False,
        )
    )
    data["label"].append(elem["id"][0])
    l = []
    for e in extracted_features.to_numpy()[0]:
        l.append(e)
    data["template"].append(l)
    print(ctr)
    ctr += 1
    


with open('Templates/templates', 'w') as convert_file:
     convert_file.write(json.dumps(data))

 
"""

#---------------------------------------------



data = {}
with open('Templates/templates') as json_file:
    data = json.load(json_file)

Xfeature = data['template']
Ylabel = data['label']

X_train, X_test, y_train, y_test = train_test_split(Xfeature, Ylabel, test_size=0.1, random_state=42)

scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
# apply same transformation to test data
X_test = scaler.transform(X_test)

dump(scaler, 'StandardScaler.joblib')

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=10000)
clf.fit(X_train, y_train)

dump(clf, 'clf.joblib')

print("Length of the model classes: ", len(clf.classes_))
print("Type of the model classes  : ", clf.classes_)
print("Model accuracy for CLF: ", clf.score(X_test, y_test)) 


lrModel = LogisticRegression(max_iter = 10000)
lrModel.fit(X_train, y_train)

dump(lrModel, 'lrModel.joblib')

print("Model accuracy for Logistic Regression: ", lrModel.score(X_test, y_test)) 

k_range =  range(1, 8)
scores = {}
scores_list = []

for k in tqdm(k_range):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    scores_list.append(neigh.score(X_test, y_test))


dump(neigh, 'KNN.joblib')


plt.plot(k_range, scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

#from sklearn.naive_bayes import MultinomialNB

#nvModel = MultinomialNB()
#nvModel.fit(X_train, y_train)
#print("Model accuracy Naive Bayes: ", nvModel.score(X_test, y_test))


svmModel = svm.SVC()
svmModel.fit(X_train, y_train)

dump(lrModel, 'svmModel.joblib')

print("Model accuracy for SVM: ", svmModel.score(X_test, y_test)) 


#-----------------------------------------------



ts = {'id':[],'time':[], 'm_x':[], 'm_y':[]}
file_path = "Test/Walk_Ex_mavi_1"
sample = np.array(np.loadtxt(file_path, dtype=float))
for temp in sample:
  ts["m_x"].append(temp[5])
  ts["m_y"].append(temp[6])
  ts["time"].append(temp[0])
  ts["id"].append(file_path)

ts = pd.DataFrame(ts)

templates = []

extracted_features = impute(extract_features(ts, column_id="id", column_sort="time", n_jobs = 1, show_warnings=False, disable_progressbar=False, profile=False))
l = []
for e in extracted_features.to_numpy()[0]:  
  l.append(e)
templates.append(l)

scaler = load("StandardScaler.joblib")
clf = load("clf.joblib")
lrModel = load("lrModel.joblib")
neigh = load("KNN.joblib")
svmModel = load("svmModel.joblib")

templates_clf = scaler.transform(templates)  

print()
print("clf: ", clf.predict(templates_clf))
print("lrModel: ", lrModel.predict(templates_clf))
print("neigh: ", neigh.predict(templates))
print("svmModel: ", svmModel.predict(templates_clf))
