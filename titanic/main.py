#!/usr/bin/env python


from __future__ import print_function

import warnings
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt
import matplotlib
import numpy as np


######################################
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 100
matplotlib.style.use('ggplot')
pd.options.display.max_rows = 100
######################################



train = pd.read_csv('../misc/data/train.csv')
test = pd.read_csv('../misc/data/test.csv')

# Prints the head of data prettily :)
# print(tabulate(train.head(), headers='keys', tablefmt='psql'))

# Describes the data stats
# print(tabulate(train.describe(), headers='keys', tablefmt='psql'))

# Imputing 'Age' column with median values
train['Age'].fillna(train['Age'].median(), inplace=True)

surv_sex = train[train['Survived'] == 1]['Sex'].value_counts()
dead_sex = train[train['Survived'] == 0]['Sex'].value_counts()

# Create graph for SurvivalRate w.r.t Gender
# df = pd.DataFrame([surv_sex, dead_sex])
# df.index = ['Survived', 'Dead']
# df.plot(kind='bar', stacked=True, figsize=(15, 8))
# plt.show()



surv_age = train[train['Survived'] == 1]['Age']
dead_age = train[train['Survived'] == 0]['Age']

# In order to tabulate a 1D array,
# reshape the array into 2D array as
# tabulate only allows 2D arrays as input
# surv_age = np.reshape(surv_age, (-1, 1))
# print(tabulate(surv_age[:20, :], headers='keys', tablefmt='psql'))

# Create a graph for SurvivalRate w.r.t Age
# plt.hist([surv_age, dead_age], stacked=True, color=['g', 'r'], bins=30, label=['Survived', 'Dead'])
# plt.xlabel('Age')
# plt.ylabel('Number of Passengers')
# plt.legend()
# plt.show()

surv_fare = train[train['Survived'] == 1]['Fare']
dead_fare = train[train['Survived'] == 0]['Fare']

# Create a graph for SurvivalRate w.r.t Fare
# plt.hist([surv_fare, dead_fare], stacked=True, color=['g', 'r'], bins=30, label=['Survived', 'Dead'])
# plt.xlabel('Fare')
# plt.ylabel('Number of Passengers')
# plt.legend()
# plt.show()


# Graph
# plt.figure(figsize=(15, 8))
# ax = plt.subplot()
# ax.scatter(surv_age, surv_fare, c='green', s=40)
# ax.scatter(dead_age, dead_fare, c='red', s=40)

# Graph
# ax.set_xlabel('Age')
# ax.set_ylabel('Fare')
# ax.legend(('survived', 'dead'), scatterpoints=1, loc='upper right', fontsize=15)
# plt.show()

# Graph
# ax = plt.subplot()
# ax.set_ylabel('Average Fare')
# train.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(15, 8), ax=ax)
# plt.show()


surv_embark = train[train['Survived'] == 1]['Embarked'].value_counts()
dead_embark = train[train['Survived'] == 0]['Embarked'].value_counts()


# Create a graph for SurvivalRate w.r.t EmbarkedPosition
# df = pd.DataFrame([surv_embark, dead_embark])
# df.index = ['Survived', 'Dead']
# df.plot(kind='bar', stacked=True, figsize=(15, 8))
# plt.show()


def status(feature):
    print('processing', feature, ': OK')

# Feature Engineering
def getCombinedData():
    test = pd.read_csv('../misc/data/test.csv')
    train = pd.read_csv('../misc/data/train.csv')

    # Extracting, then removing targets from training data
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)

    # merging train and test data for feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)

    return combined


combined = getCombinedData()

# pretty-print combined data
# print(combined.shape)
# print(tabulate(combined.describe(), headers='keys', tablefmt='psql'))
# print(tabulate(combined[:100][:], headers='keys', tablefmt='psql'))

def getTitles():
    global combined

    # extract title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    # mapping titles
    Title_Dictionary = {
        "Capt":         "Officer",
        "Col":          "Officer",
        "Major":        "Officer",
        "Jonkheer":     "Royalty",
        "Don":          "Royalty",
        "Sir" :         "Royalty",
        "Dr":           "Officer",
        "Rev":          "Officer",
        "the Countess": "Royalty",
        "Dona":         "Royalty",
        "Mme":          "Mrs",
        "Mlle":         "Miss",
        "Ms":           "Mrs",
        "Mr" :          "Mr",
        "Mrs" :         "Mrs",
        "Miss" :        "Miss",
        "Master" :      "Master",
        "Lady" :        "Royalty"
    }

    # mapping title to dictionary_val
    combined['Title'] = combined.Title.map(Title_Dictionary)

getTitles()

# pretty-print combined data
# print(combined.shape)
# print(tabulate(combined.describe(), headers='keys', tablefmt='psql'))
# print(tabulate(combined[:100][:], headers='keys', tablefmt='psql'))

# imputing 'Age' values according to the section the person belongs
# instead of taking median of values
# in order to understand the reason for this method,
# run the following commands :
#####################################################################
# features = ['Sex', 'Pclass', 'Title']
# grouped = combined.groupby(features)
# print(tabulate(grouped.median(), headers='keys', tablefmt='psql'))
#####################################################################
# notice that different sections of people [differentiated by `features`]
# have different medians of age

def processAge():
    global combined

    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26

    combined.Age = combined.apply(lambda r: fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    status('age')

processAge()
# print(combined.info())

def processNames():
    global combined

    # clean-up of `Name` variable
    combined.drop('Name', axis=1, inplace=True)

    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)
    combined.drop('Title', axis=1, inplace=True)

    status('names')

processNames()
# print(tabulate(combined.head(), headers='keys', tablefmt='psql'))

def processFares():
    global combined
    combined.Fare.fillna(combined.Fare.mean(), inplace=True)

    status('fare')

processFares()

def processEmbarked():
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.Embarked.fillna('S',inplace=True)

    # dummy encoding
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)

    status('embarked')

processEmbarked()

def processCabin():
    global combined

    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U',inplace=True)
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')
    combined = pd.concat([combined,cabin_dummies],axis=1)
    combined.drop('Cabin',axis=1,inplace=True)

    status('cabin')

processCabin()
# print(combined.info())

def processSex():
    global combined
    # mapping string values to numerical one
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})

    status('sex')

# creates a 2d matrix out of 2 lists of same shape
# pclass_sample = np.column_stack((combined['Pclass'][:50], combined['Sex'][:50]))

# print(tabulate(pclass_sample, headers='keys', tablefmt='psql'))

processSex()


def processPclass():
    global combined
    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
    combined = pd.concat([combined,pclass_dummies],axis=1)
    combined.drop('Pclass',axis=1,inplace=True)

    status('pclass')

processPclass()

# print all columns' names
# print(combined.head().columns.values)

def processTicket():
    global combined

    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'


    # Extracting dummy variables from tickets:
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)

    status('ticket')

processTicket()

def processFamily():
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)

    status('family')

processFamily()

def scaleAllFeatures():
    global combined

    features = list(combined.columns)
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)

    print('Features scaled successfully !')

scaleAllFeatures()



#######################################################
#           Building a Predictive Model               #
#######################################################

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


def computeScore(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)
    return np.mean(xval)


def recoverTrainTestTarget():
    global combined

    train0 = pd.read_csv('../misc/data/train.csv')
    targets = train0.Survived
    train = combined.ix[0:890]
    test = combined.ix[891:]

    return train, test, targets

train, test, targets = recoverTrainTestTarget()

clf = ExtraTreesClassifier(n_estimators=200)
clf.fit(train, targets)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_

# prints features according to importance in prediction
# print(tabulate(features.sort(['importance'], ascending=False), headers='keys', tablefmt='psql'))


model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
test_new = model.transform(test)

# print(train_new.shape)
# print(test_new.shape)
# print(tabulate(train_new[:10], headers='keys', tablefmt='psql'))
# print(tabulate(test_new[:10], headers='keys', tablefmt='psql'))



#######################################################
#              Hyperparameter Tuning                  #
#######################################################

forest = RandomForestClassifier(max_features='sqrt')
parameter_grid = {
    'max_depth': [4, 5, 6, 7, 8],
    'n_estimators': [200, 210, 240, 250],
    'criterion': ['gini', 'entropy']
}

skf = StratifiedKFold(n_splits=5)
cross_validation = skf.get_n_splits(targets)
grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=cross_validation)
grid_search.fit(train_new, targets)

print('Best Score: {}'.format(grid_search.best_score_))
print('Best Parameters: {}'.format(grid_search.best_params_))





output = grid_search.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId', 'Survived']].to_csv('../misc/output/output.csv', index=False)
