import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from itertools import compress
from sklearn import preprocessing as pp
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Kaggle_competitions/Santander_Cust_tran_Pred/train.csv')
df_test = pd.read_csv('/Users/shankarmanoharan/VSCode/Kaggle_competitions/Santander_Cust_tran_Pred/test.csv')


# This dataset is bank data. it has got no description of the data 
# Provided with an anonymized dataset containing numeric feature variables, the binary target column, and a string ID_code column
# Imbalanced data
    #  0    179902
    #  1     20098
# All numeric data  - no cat columns
# Metric is AUC

print(df.info())
print(df.shape)
#print(df.describe())
print(df['target'].value_counts())

#set the ID as index
df.set_index('ID_code',inplace=True)

#verify if the target is extracted correctly
#seperate features & target
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
#print(y.name)

# which Model works best with all the data - this gives direction 
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# Standard Scalar for all the numeric features
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)
X_transformed = pd.DataFrame(X_transformed)
print('Transformed Shape',X_transformed.shape)

# Logistic regression
model = LogisticRegression(max_iter=200)
scores = cross_val_score(model,X_transformed,y,cv=8,scoring='roc_auc')
print("Before Logist Accuracy:",np.round(scores,2))
print("Before Logistic Accuracy mean:, %0.2f" %(scores.mean()))

"""
#Random Forest
rf = RandomForestClassifier(n_estimators=200,random_state=2,n_jobs=-1)
scores = cross_val_score(rf,X_transformed,y,cv=8,scoring='roc_auc')
print('roc_auc', np.round(scores,2))
print("AUC mean:, %0.2f" %(scores.mean())) 

# Xgboost
model = XGBClassifier()
scores = cross_val_score(model,X_transformed,y,cv=8,scoring='roc_auc')
print("XGB Accuracy:",np.round(scores,2))
print("XGB Accuracy mean:, %0.2f" %(scores.mean())) """


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


#started with 200 features -  remove 15 non contributing features
ufs = UnivariateFeatureSelection(n_features=190,problem_type='classification',scoring='f_classif')
ufs.fit(X_transformed,y)
scores = -np.log10(ufs.selection.pvalues_)
X_iter1 = X_transformed.iloc[:,ufs.selection.get_support(indices=True)]
print(X_iter1.shape)

"""
import matplotlib.pyplot as plt
X_indices = np.arange(X.shape[-1])
plt.figure(1)
plt.clf()
plt.bar(X_indices - 0.05, scores, width=0.2)
plt.title("Feature univariate score")
plt.xlabel("Feature number")
plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
plt.show()
"""

# Logistic regression - model performance after 190 features
model = LogisticRegression(max_iter=200,solver='sag')
scores = cross_val_score(model,X_transformed,y,cv=8,scoring='roc_auc')
print("After Logist Accuracy:",np.round(scores,2))
print("After Logistic Accuracy mean:, %0.2f" %(scores.mean()))


# Coefficients and Odds Ratios
model.fit(X_transformed,y)
coefficients = model.coef_[0]
odds_ratios = np.exp(coefficients)


# Display feature importance using coefficients and odds ratios
feature_importance = pd.DataFrame({
    'Feature': X_transformed.columns,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios
})

print("File Write")
feature_importance.sort_values(by='Coefficient', ascending=False).to_csv('/Users/shankarmanoharan/VSCode/Kaggle_competitions/Santander_Cust_tran_Pred/Santender.txt.csv')
print("File Write Compelete")

df_features = pd.read_csv('/Users/shankarmanoharan/VSCode/Kaggle_competitions/Santander_Cust_tran_Pred/Santender.txt.csv')
print(df_features.head())

features = df_features[df_features['Odds Ratio'] >= 0.85]['Feature']

print(features.to_list())

X_transformed1 = X_transformed[features]

model = LogisticRegression(max_iter=200,solver='sag')
scores = cross_val_score(model,X_transformed1,y,cv=8,scoring='roc_auc')
print("After_F Logist Accuracy:",np.round(scores,2))
print("After_F Logistic Accuracy mean:, %0.2f" %(scores.mean()))

# Xgboost
model = XGBClassifier()
scores = cross_val_score(model,X_transformed1,y,cv=8,scoring='roc_auc')
print("XGB_F Accuracy:",np.round(scores,2))
print("XGB-F Accuracy mean:, %0.2f" %(scores.mean())) 


print("before Polynomial Features",X_transformed1.shape)
from sklearn import preprocessing
pf = preprocessing.PolynomialFeatures(
    degree=2,
    interaction_only=False,
    include_bias=False
)
pf.fit(X_transformed1)
Poly = pf.transform(X_transformed1)
num_features = Poly.shape[1]
X_transformed1_Poly = pd.DataFrame(Poly,columns=[f"V_{i}" for i in range(1,num_features+1)])
print("after Polynomial Features",X_transformed1_Poly.shape)

X_transformed1_Poly.columns.to_list().to_csv('/Users/shankarmanoharan/VSCode/Kaggle_competitions/Santander_Cust_tran_Pred/Santender_Features.csv')
