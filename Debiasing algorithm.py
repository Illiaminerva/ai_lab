#preprocessing the data
import random
import pandas as pd
df = pd.read_csv(r"adult.data")
#saving the weights we will use for later
weights = df[df.columns[2]]
df = df.drop(df.columns[2], axis=1)
df.dropna()
new_column_names = {'39': 'age',' State-gov': 'workclass', ' Bachelors': 'education', ' 13': 'education-num', ' Never-married':'martial-status',  ' Adm-clerical':'occupation', ' Not-in-family':'relationship', ' White':'race', ' Male':'sex', ' 2174':'capital-gain', ' 0':'capital-loss', ' 40':'hours-per-week', ' United-States':'native-country', ' <=50K':'income'}
# rename columns
df = df.rename(columns=new_column_names)
df['sex'] = df['sex'].replace({' Female': 0, ' Male': 1})
df['relationship'] = df['relationship'].replace({' Wife' : 0, ' Own-child' : 1, ' Husband' : 2, " Not-in-family" : 3, " Other-relative" : 4," Unmarried" : 5})
df['income'] = df['income'].replace({' <=50K': 0, ' >50K' : 1})
df['education'] = df['education'].replace({" Bachelors" : 0, " Some-college" : 1, " 11th" : 2, " HS-grad" : 3, " Prof-school" : 4," Assoc-acdm" : 5, " Assoc-voc" : 6, " 9th" : 7, " 7th-8th" : 8, " 12th" : 9," Masters" : 10, " 1st-4th" : 11," 10th" : 12, " Doctorate" : 13, " 5th-6th" : 14, " Preschool" : 15})
df['race'] = df['race'].replace({" White" : 0, " Asian-Pac-Islander" : 1, " Amer-Indian-Eskimo" : 2, " Other" : 3, " Black" : 4})
print(df)


#leaving desired variables
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = df.drop(['workclass', 'martial-status', 'occupation', 'capital-gain', 'capital-loss', 'native-country'], axis=1)
X = df.drop('income', axis=1)
y = df['income']
print(df)

#Linear Regression without reweighting
from sklearn.metrics import accuracy_score
#splitting the dataset between test and training data

all_indexes = list(range(0, len(df)))
test_indexes = random.sample(all_indexes, int(0.3*len(all_indexes)))
training_indexes = [i for i in all_indexes if i not in test_indexes]

#defining test and training datasets – both input and output variables
test = df.iloc[test_indexes]
X_test = test.drop(columns=['income'])
y_test = test['income']
training = df.iloc[training_indexes]
X_train = training.drop(columns=['income'])
y_train = training['income']

#setting the sensitive attribute we will debias for
sensitive_attribute = 'sex'

#training our data
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

#making predictions
y_pred = logistic_regression.predict(X_test)
prediction_no_weights = X_test.copy()
prediction_no_weights['income'] = y_pred

#calculating and outputing the counts for priviliged/unpriviliged groups
counts = prediction_no_weights.groupby([sensitive_attribute, 'income']).size().unstack()
print(counts)
base_rate = counts[1] / counts.sum(axis=1)
fpr = counts.loc[0, 1] / counts.loc[0].sum()
fnr = counts.loc[1, 0] / counts.loc[1].sum()

#calculating statistics
disparate_impact = np.max(base_rate) / np.min(base_rate)
average_odds_difference = 0.5 * (np.abs(fpr - base_rate[1]) + np.abs(fnr - base_rate[0]))
accuracy = accuracy_score(y_test, y_pred)

#outputing the statistics
print("The disparate impcat is", round(disparate_impact,3))
print("The average odds difference is", round(average_odds_difference,3))
print("The accuracy of the model is", round(accuracy,3))

#Linear Regression with reweighting

#setting up the weights
weights_train = [weights[i] ** 2 for i in training_indexes]

#training our data
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train, sample_weight=weights_train)

#making predictions
y_pred = logistic_regression.predict(X_test)
prediction_weights = X_test.copy()
prediction_weights['income'] = y_pred

#calculating and outputing the counts for priviliged/unpriviliged groups
counts = prediction_weights.groupby([sensitive_attribute, 'income']).size().unstack()
print(counts)
base_rate = counts[1] / counts.sum(axis=1)
fpr = counts.loc[0, 1] / counts.loc[0].sum()
fnr = counts.loc[1, 0] / counts.loc[1].sum()

#calculating statistics
disparate_impact = np.max(base_rate) / np.min(base_rate)
average_odds_difference = 0.5 * (np.abs(fpr - base_rate[1]) + np.abs(fnr - base_rate[0]))
accuracy = accuracy_score(y_test, y_pred)

#outputing the statistics
print("The disparate impcat is", round(disparate_impact,3))
print("The average odds difference is", round(average_odds_difference,3))
print("The accuracy of the model is", round(accuracy,3))