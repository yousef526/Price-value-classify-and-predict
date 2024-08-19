import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from sklearn.metrics import r2_score
from sklearn import svm, datasets
import pickle
import time
from sklearn.linear_model import LogisticRegression




data = pd.read_csv('player-tas-classification-test.csv')
#data = pd.read_csv('sliced.csv')
#data.drop(data.columns[0],axis=1,inplace=True)


#data = pd.read_csv('player-test-samples.csv')
#data.dropna(how='any', inplace=True)
##########################################################
# bassem
# data = pd.read_csv('player-value-prediction.csv')
data22 = data.iloc[:, 13:29]  # This is my columns

indexes = {}

for i in list(data.columns):
    indexes[i] = list(data.columns).index(i)


def where(column):
    return indexes[column]


position_groups = [
    ['CB'],
    ['LB', 'RB', 'LWB', 'RWB'],
    ['LM', 'RM', 'CM', 'CDM'],
    ['CAM'],
    ['LW', 'RW', ],
    ['ST', 'CF']
]

group_length = len(position_groups)

null_contract_end_year = data['contract_end_year'].isna()
null_club_join_date = data['club_join_date'].isna()

data['national_team'].fillna('NONE', inplace=True)
data['national_rating'].fillna(0, inplace=True)
data['national_team_position'].fillna('NONE', inplace=True)
data['national_jersey_number'].fillna(0, inplace=True)

# Defence begins as High and goes to Low
# Attack begins Low goes to High
rates = {'High': 3, 'Medium': 2, 'Low': 1}


def work_rate(pos, rate):
    if pos == 'GK':
        return 15
    atk, dfs = rate.split('/ ')
    atk, dfs = rates[atk], rates[dfs]
    for i in range(group_length):
        if pos in position_groups[i]:
            return int((group_length - i - 1) * dfs + i * atk)  # group_length - i begins as topper


rate_index = where('work_rate')

for i in range(len(data)):
    row = data.loc[i].astype(str)
    if null_contract_end_year[i]:
        data.iloc[i, where('contract_end_year')] = '2022'

    elif len(row['contract_end_year']) != 4:
        data.iloc[i, where('contract_end_year')] = '20' + row['contract_end_year'][-2:]

    if null_club_join_date[i]:
        end = int(data.iloc[i, where('contract_end_year')])
        join = end - 2
        join_date = f"1/1/{join}"
        data.iloc[i, where('club_join_date')] = join_date

    if row['body_type'] not in ['Normal', 'Stocky', 'Lean']:
        data.iloc[i, 4] = 'Normal'

    rate = 0
    num_of_positions = len(row['positions'].split(','))
    for position in row['positions'].split(','):
        position = position.strip()
        rate += work_rate(position, row['work_rate'])
    data.iloc[i, rate_index] = round(rate / num_of_positions, 1)

data = Feature_Encoder(data, ['body_type', 'club_position', 'national_team', 'national_team_position'])
###########################################################################

##############################################################################
# yousef
Pos_list = {'LB': 2, 'CB': 2, 'RB': 2, 'RWB': 2, 'LWB': 2,
            'LM': 3, 'CM': 3, 'CDM': 3, 'RM': 3, 'CAM': 3,
            'LW': 4, 'LF': 4, 'ST': 4, 'CF': 4, 'RF': 4, 'RW': 4, 'GK': 1}
col = data['positions']
# i for the col
# j for key,w for the value in the Pos_list dict
# k for str list
for i in range(len(data['positions'])):
    str = ""
    str = data['positions'][i]
    str = str.split(',')
    result1 = 0
    for k in str:
        result1 += int(Pos_list[k])
    data['positions'][i] = int(result1)

# prefreed foot preporcess
for i in range(len(data['preferred_foot'])):
    if data['preferred_foot'][i] == 'Right':
        data['preferred_foot'][i] =int(1)
    else:
        data['preferred_foot'][i] = int(2)

##########################################################################

########################################################################

# galall

# Load players data
# data = pd.read_csv('player-value-prediction.csv')
X = data.iloc[:, 29:64]  # Features
for i in X.iloc[:, :33]:
    data[i] = data[i].fillna(data[i].mean)
data['tags'] = data['tags'].fillna('unknown')

cols = ('tags',)
data = Feature_Encoder(data, cols)


#########################################################################

##########################################################################
# mostafa
def postinsPreprocessing(s):
    if (s is np.nan):
        return s;
    strList = s.split('+')
    if (len(strList) > 1):
        return int(strList[0])
    else:
        return int(strList[0] + strList[1]);


def traitsPreprocessig(s):
    if (s is np.nan):
        return 0;
    strList = s.split(',')
    res = len(strList) * 100;
    if ('Injury Prone' in strList):
        res = res - 200;
    return res;


# data = pd.read_csv('player-value-prediction.csv')
data.dropna(how='all', inplace=True)
colms = (
'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB')
for c in colms:
    data[c] = data[c].apply(postinsPreprocessing)
data['traits'] = data['traits'].apply(traitsPreprocessig)

for i in range(data.shape[0]):
    if (data.loc[i]['LS'] is np.nan):
        data.iloc[i, 65:91] = data.loc[i, 'overall_rating']
# print(data.loc[0]['id'])
# X = data.iloc[:,64:91]


########################################################################################
########################################################################################


data = Feature_Encoder(data, ['club_team','nationality'])

data = data.drop(['id', 'name', 'full_name', 'birth_date','preferred_foot','positions','work_rate',

                  'release_clause_euro', 'club_jersey_number', 'club_join_date', 'contract_end_year'

                 , 'national_team_position', 'national_jersey_number','composure','reactions','short_passing','long_passing'], axis=1)

########################################################################################
def f (s):
    if (s=='Normal'):
        return 25
    return int(s)
data['age']=data['age'].apply(f)
data.fillna(value=data.mean(), inplace=True)
#data.dropna(how='any',inplace=True)
list =['age', 'overall_rating', 'potential', 'wage', 'international_reputation(1-5)', 'club_rating', 'LCM', 'CM', 'RCM']

X = data.iloc[:, 0:74] ;
X = X[list]
Y = data["PlayerLevel"]
#X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=5)

import joblib



################################################################################################
#model classifiction
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier

# SVM one vs one
"""C = 0.001  # SVM regularization parameter
time_train_svm1 = time.time()
svc = OneVsOneClassifier(SVC(kernel='linear', C=0.0001,max_iter=100000000)).fit(X_train, y_train)
time_train_svm2 = time.time()

time_test_svm1 = time.time()
predictions = svc.predict(X_test)
accuracy1 = np.mean(predictions == y_test)
time_test_svm2 = time.time()
print(accuracy1,"one vs one")# %87
print("time traing svm",time_train_svm2 - time_train_svm1)
print("time test svm",time_test_svm2 - time_test_svm1)"""




#with open('modelSVC_pkl', 'wb') as files:
 #   pickle.dump(svc, files)

with open('modelSVC_pkl', 'rb') as f:
    lr = pickle.load(f)
predictions = lr.predict(X)
accuracy = np.mean(predictions == Y)
print("One vs one:",accuracy)#87%
#######################################################################




#adaboost and Descision tree
"""from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),
                         algorithm="SAMME",
                         n_estimators=120)
time_train_ada1 = time.time()
bdt.fit(X_train,y_train);
time_train_ada2 = time.time()

time_test_ada1 = time.time()
predictions = bdt.predict(X_test)
accuracy2 = np.mean(predictions == y_test)
time_test_ada2 = time.time()
print(accuracy2," Descion tree with adaboost")#%90
print("time traing ada",time_train_ada2 - time_train_ada1)
print("time test ada",time_test_ada2 - time_test_ada1)"""

#with open('modeladaboost_pkl', 'wb') as files:
 #   pickle.dump(bdt, files)

with open('modeladaboost_pkl', 'rb') as f:
    lr = pickle.load(f)
predictions = lr.predict(X)
accuracy = np.mean(predictions == Y)
print(" Adaboost and Descision tree: ",accuracy)
##################################################################
#KNN

"""knn = KNeighborsClassifier(n_neighbors=5,weights="distance",algorithm='kd_tree')

time_train_knn1 = time.time()
knn.fit(X_train, y_train)
time_train_knn2 = time.time()

time_test_knn1 = time.time()
predictions = knn.predict(X_test)
accuracy3 = np.mean(predictions == y_test)
time_test_knn2 = time.time()
print(accuracy3,"KNN model acc")
print("time traing knn",time_train_knn2 - time_train_knn1)
print("time test knn",time_test_knn2 -time_test_knn1)"""

#with open('modelknn_pkl', 'wb') as files:
 #   pickle.dump(knn, files)

with open('modelknn_pkl', 'rb') as f:
    lr = pickle.load(f)
predictions = lr.predict(X)
accuracy = np.mean(predictions == Y)
print(" Knn model: ",accuracy)

################################
##Accuracy bar graph for each model

"""data45 = {'Accuracy SVM':accuracy1 , 'Accuracy adaboost':accuracy2, 'accuracy knn':accuracy3}


fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(data45.keys(), data45.values(), color='maroon',
        width=0.4)

#plt.xlabel("Courses offered")
plt.ylabel("Accuracy")
plt.title("Models Accuracy")
plt.show()



###########################
##training time bar graph for each model

data45 = {'SVM traning time': time_train_svm2 - time_train_svm1, 'adaboost traning time': time_train_ada2 - time_train_ada1, 'KNN traning time': time_train_knn2 - time_train_knn1}


fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(data45.keys(), data45.values(), color='maroon',
        width=0.4)

plt.ylabel("Training time")
plt.title("Training time for models")
plt.show()




#################################3
## testing time bar graph for each model
data45 = {'SVM test time': time_test_svm2 - time_test_svm1, 'adaboost test time': time_test_ada2 - time_test_ada1, 'KNN test time': time_test_knn2 - time_test_knn1}


fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(data45.keys(), data45.values(), color='maroon',
        width=0.4)

plt.ylabel("Testing  time")
plt.title("Testing time for models")
plt.show()"""