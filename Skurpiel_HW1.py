#Imports
import pandas as pd
import numpy as np
import csv as csv

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

#Visuals(If I have time to do this)
#import seaborn as sns

#Loads training set and test file
print('--------Load train & test file------')
train_dataset = pd.read_csv('HW1\input/train.csv')
test_dataset = pd.read_csv('HW1\input/test.csv')

print('----Train dataset column types information-------')
dtype_df = train_dataset.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()

print('----Train Dataset Information-------')
dtype_df

# Class vs Survived
print(train_dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# sex vs Survived
print(train_dataset[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# SibSp vs Survived
print(train_dataset[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Parch vs Survived
print(train_dataset[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Data sets cleaning, fill nan (null) where needed and delete uneeded columns
print('----------cleaning ------------')

#manage Age
train_random_ages = np.random.randint(train_dataset["Age"].mean() - train_dataset["Age"].std(),
                                          train_dataset["Age"].mean() + train_dataset["Age"].std(),
                                          size = train_dataset["Age"].isnull().sum())

test_random_ages = np.random.randint(test_dataset["Age"].mean() - test_dataset["Age"].std(),
                                          test_dataset["Age"].mean() + test_dataset["Age"].std(),
                                          size = test_dataset["Age"].isnull().sum())

train_dataset["Age"][np.isnan(train_dataset["Age"])] = train_random_ages
test_dataset["Age"][np.isnan(test_dataset["Age"])] = test_random_ages
train_dataset['Age'] = train_dataset['Age'].astype(int)
test_dataset['Age']    = test_dataset['Age'].astype(int)

#Port embrakation
train_dataset["Embarked"].fillna('S', inplace=True)
test_dataset["Embarked"].fillna('S', inplace=True)
train_dataset['Port'] = train_dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_dataset['Port'] = test_dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
del train_dataset['Embarked']
del test_dataset['Embarked']

#Fare Price
test_dataset["Fare"].fillna(test_dataset["Fare"].median(), inplace=True)

#Passanger has a cabin
train_dataset['Has_Cabin'] = train_dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_dataset['Has_Cabin'] = test_dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)\
# group them
full_dataset = [train_dataset, test_dataset]

#Featute for the Family_Size
for dataset in full_dataset:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
### new try 

# Feature for if they're travelling alone
for dataset in full_dataset:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
# Get titles from the names
train_dataset['Title'] = train_dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_dataset['Title'] = test_dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
  
## Create new column "FamilySizeGroup" and assign "Alone", "Small" and "Big"
for dataset in full_dataset:
    dataset['FamilySizeGroup'] = 'Small'
    dataset.loc[dataset['FamilySize'] == 1, 'FamilySizeGroup'] = 'Alone'
    dataset.loc[dataset['FamilySize'] >= 5, 'FamilySizeGroup'] = 'Big'

## Get the average survival rate of different FamilySizes
train_dataset[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()

for dataset in full_dataset:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
        
for dataset in full_dataset:    
    dataset.loc[ dataset['Age'] <= 14, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

for dataset in full_dataset:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

 #Family size and titles feature
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
family_mapping = {"Small": 0, "Alone": 1, "Big": 2}
for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['FamilySizeGroup'] = dataset['FamilySizeGroup'].map(family_mapping)

#Ischildandrich, my theory that if rich and a child you had a higher chance of survival
for dataset in full_dataset:
    dataset['IsChildandRich'] = 0
    dataset.loc[(dataset['Age'] <= 0) & (dataset['Pclass'] == 1 ),'IsChildandRich'] = 1  
    dataset.loc[(dataset['Age'] <= 0) & (dataset['Pclass'] == 2 ),'IsChildandRich'] = 1  
    

for data in full_dataset:
    # classify Cabins by fare price
    data['Cabin'] = data['Cabin'].fillna('X')
    data['Cabin'] = data['Cabin'].apply(lambda x: str(x)[0])
    data['Cabin'] = data['Cabin'].replace(['A', 'D', 'E', 'T'], 'M')
    data['Cabin'] = data['Cabin'].replace(['B', 'C'], 'H')
    data['Cabin'] = data['Cabin'].replace(['F', 'G'], 'L')
    data['Cabin'] = data['Cabin'].map({'X': 0, 'L': 1, 'M': 2, 'H': 3}).astype(int) 

    
#Can maunally check the names later...
del train_dataset['Name']
del test_dataset['Name']

del train_dataset['SibSp']
del test_dataset['SibSp']

del train_dataset['Parch']
del test_dataset['Parch']

del train_dataset['FamilySize']
del test_dataset['FamilySize']

del train_dataset['Cabin']
del test_dataset['Cabin']

# Dont need the tickets numbers
del train_dataset['Ticket']
del test_dataset['Ticket']

del train_dataset['Port']
del test_dataset['Port']

print('----Cleaning the final data ------------')

print('train dataset: %s, test dataset %s' %(str(train_dataset.shape), str(test_dataset.shape)) )
train_dataset.head()

del train_dataset['PassengerId']

X_train = train_dataset.drop("Survived",axis=1)
Y_train = train_dataset["Survived"]
X_test  = test_dataset.drop("PassengerId",axis=1).copy()

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

svc = SVC(C = 0.1, gamma=0.1)
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": test_dataset["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanicResults.csv', index=False)
print('File Exported, check if its right')