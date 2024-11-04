import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from itertools import compress
from sklearn import preprocessing as pp
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Kaggle_competitions/Predict_Bank_Churn/train.csv')
df1 = pd.read_csv('/Users/shankarmanoharan/VSCode/Kaggle_competitions/Predict_Bank_Churn/test.csv')

#Exploratory Analysis
df1.drop(columns=['CustomerId','Surname'],axis=1,inplace=True)
df1.set_index('id',inplace=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X.set_index('id',inplace=True)
X.drop(columns=['CustomerId','Surname'],axis=1,inplace=True)
col_names = X.columns.to_list()

lbl_encod = pp.LabelEncoder()
X['Geography'] = lbl_encod.fit_transform(X['Geography'])
X['Gender'] = lbl_encod.fit_transform(X['Gender'])
df1['Geography'] = lbl_encod.fit_transform(df1['Geography'])
df1['Gender'] = lbl_encod.fit_transform(df1['Gender'])

# Remove constant Variance
# Retained all of the variance
#var_tresh = VarianceThreshold(0.1)
#transformed_data = var_tresh.fit_transform(X)
#df1 = pd.DataFrame(data=transformed_data,columns=[list(compress(col_names, var_tresh.get_support(indices=False)))])
#print(df1.columns.to_list())


# Univariate Feature selection
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

class UnivariateFeatureSelection:
    def __init__(self,n_features,problem_type,scoring):
        ### n_features : selectpercentile if float else SelectKbest
        ### Problem Type : classification or Regression
        ### scoring : scoring function
        if problem_type == "classification":
            valid_scoring = {
                "f_classif" : f_classif,                
                "mutual_info_classif" : mutual_info_classif
            }        
        if scoring not in valid_scoring:
            raise Exception("Invalid Scoring function")        
        if isinstance(n_features, int):
            self.selection = SelectKBest(valid_scoring[scoring],k=n_features)        
        else: raise Exception("Invalid Type of feature")

    def fit(self,X,y):
        return self.selection.fit(X,y)
        
    def transform(self,X):
        return self.selection.transform(X)

    def fit_transform(self,X,y):
        return self.selection.fit_transform(X,y)


ufs = UnivariateFeatureSelection(n_features=9,problem_type='classification',scoring='mutual_info_classif')
ufs.fit(X,y)
scores = ufs.selection.scores_
#-np.log10(ufs.selection.pvalues_)
print(scores)
print(col_names)
print(ufs.selection.get_support(False))


import matplotlib.pyplot as plt
X_indices = np.arange(X.shape[-1])
plt.figure(1)
plt.clf()
plt.bar(X_indices - 0.05, scores, width=0.2)
plt.title("Feature univariate score")
plt.xlabel("Feature number")
plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
plt.show()


# Basic Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

#Split the data
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2)

lin_reg = LogisticRegression()
lin_reg.fit(X_train,y_train)
y_pred = pd.Series(lin_reg.predict_proba(X_test)[:,1])

y_pred = y_pred.apply(lambda x: 0 if x <=0.50 else 1)

print(y_test.value_counts())
print('Base Model accuracy',accuracy_score(y_test,y_pred))
print('Base Model confusion Matrix\n', confusion_matrix(y_test,y_pred))
print('Base Model Classification report\n', classification_report(y_test,y_pred))


model = XGBClassifier(booster='gbtree', objective='binary:logistic',random_state=2)
model.fit(X_train,y_train)
y_pred = pd.Series(model.predict_proba(X_test)[:,1])
y_pred = y_pred.apply(lambda x: 0 if x <=0.50 else 1)
print('Xgboost accuracy',accuracy_score(y_test,y_pred))
print('Xgboost confusion Matrix\n', confusion_matrix(y_test,y_pred))
print('Xgboost Classification report\n', classification_report(y_test,y_pred))

# imbalanced Data
weight = 3072/15000 

# Xgboost Hyper parameter Tuning
from sklearn.model_selection import StratifiedKFold, cross_val_score
kfold = StratifiedKFold(n_splits=2,shuffle=True,random_state=2)
model = XGBClassifier(booster='gbtree', objective='binary:logistic',scale_pos_weight=weight,random_state=2)

scores = cross_val_score(model,X,y,cv=kfold,scoring='accuracy')
print('Accuracy', scores)
print('Accuracy Mean', scores.mean())

# custom function
#import sys
#sys.path.append("/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch7")


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
def grid_search(kfold,X,y,params,random=False):
    xgb = XGBClassifier(booster='gbtree',random_state=2)
    if random:
        grid = RandomizedSearchCV(xgb,params,cv=kfold,scoring='recall',n_jobs=-1,n_iter=25)
    else:
        grid = GridSearchCV(xgb,params,cv=kfold,scoring='recall',n_jobs=-1)
    grid.fit(X,y)
    best_params = grid.best_params_
    print('Best Params:',best_params)
    best_score = grid.best_score_    
    print('Training Best Score',best_score)
    return grid.best_estimator_

#iter 2
best_model = grid_search(params={'n_estimators':[800],
                    'learning_rate':[0.3],
                    'max_depth':[7],
                    'subsample':[0.5],
                    'gamma':[5],
                    'min_child_weight':[2],
                    'colsample_bytree':[0.7],
                    'colsample_bynode':[0.8],
                    'colsample_bylevel':[0.7]}, 
                     kfold=kfold,X=X,y=y)

#Best Params: {'colsample_bylevel': 0.7, 'colsample_bynode': 0.8, 'colsample_bytree': 0.8, 'gamma': 5, 'learning_rate': 0.3, 'max_depth': 7, 'min_child_weight': 2, 'n_estimators': 800, 'subsample': 0.5}

from xgboost import XGBClassifier
lbl_encod = pp.LabelEncoder()
y_pred = pd.Series(best_model.predict_proba(df1)[:,1])

submission = pd.DataFrame(df1.reset_index().id).assign(Exited=y_pred)
submission.set_index('id',inplace=True)

submission.to_csv('/Users/shankarmanoharan/VSCode/Kaggle_competitions/Predict_Bank_Churn/submission.csv')

