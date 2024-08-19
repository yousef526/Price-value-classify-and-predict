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
import time
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris


data = pd.read_csv('player-value-prediction.csv')

null_Check = pd.notnull(data)
data[null_Check]
# after check no null values are found found in


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
data.dropna(how='any',inplace=True)
X = data.iloc[:, 0:77]  # Features
Y = data['value']  # Label
fifa_data = data.iloc[:, :]
data.to_csv("new fifa.csv",index=False)
# cols=('Nationality','Club','Position')
# X=Feature_Encoder(X,cols)

# Feature Selection
# Get the correlation between the features
#print(X.info())
corr = fifa_data.corr()

# Top 50% Correlation training features with the Value

top_feature = corr.index[abs(corr['value']) > 0.5]


# Correlation plot

plt.subplots(figsize=(12, 8))
top_corr = fifa_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)
#print(top_feature)
list = ['age']
list.extend(top_feature)
#print(list)
X = X[list]
X = featureScaling(X, 0, 100)
#################################################################
#polynoimal

# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=10)

# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.50,shuffle=True)

poly_features = PolynomialFeatures(degree=4)

#time for model to start
start = time.time()

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
ypred = poly_model.predict(poly_features.transform(X_test))

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

#time after end
stop = time.time()

# print('Co-efficient of linear regression',poly_model.coef_)
# print('Intercept of linear regression model',poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
print('Accuracy: ',r2_score(y_test, prediction))
print(f"Training time: {stop - start}s")

true_player_value=np.asarray(y_test)[5]
predicted_player_value=prediction[5]
print("true player value: ",true_player_value)
print('predicted player value :',abs(predicted_player_value))

##############################################################################
print('#######################################################')
print('#######################################################')
print('#######################################################')

#mutivarite

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10)


#time for model to start
start = time.time()

# creating an object of LinearRegression class
LR = linear_model.LinearRegression()
# fitting the training data
LR.fit(X_train,y_train)

y_prediction =  LR.predict(x_test)
#y_prediction
#score=r2_score(y_test,y_prediction)

#time after end
stop = time.time()

print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
print('Accuracy: ',r2_score(y_test, prediction))
print(f"Training time: {stop - start}s")

true_player_value=np.asarray(y_test)[5]
predicted_player_value=prediction[5]
print("true player value: ",true_player_value)
print('predicted player value :',abs(predicted_player_value))

