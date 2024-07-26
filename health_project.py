# -*- coding: utf-8 -*-
"""
Created on Wed May  8 19:41:14 2024

@author: Sobala
https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset
"""

import pandas as pd

df=pd.read_csv(r"C:\Users\Sobala\Desktop\ŁUKASZ\nauka\projekty\2 health - python (ML)\Heart Attack.csv")

#----------------------------
### preprocessing

#examining the dataset
dsc=df.describe(include='all')

for col in df.columns:
    print (col,df[col].isna().sum())

df.info()
df.shape

#handling outliers
import matplotlib.pyplot as plt


for i in range(len(df.columns)-1):
    bp=plt.boxplot(df.iloc[:,i])
    plt.title(df.iloc[:,i].name)
    plt.show()
    

df.drop(df[df['impluse'].gt(200)].index,inplace=True)

df.drop(df[df['pressurehight'].lt(65)].index,inplace=True)

df.drop(df[df['pressurelow'].gt(109)].index,inplace=True)
df.shape

#analyzing specific features 
positive=df[df['class']=='positive']
negative=df[df['class']=='negative']

fig,ax=plt.subplots(3)
ax[0].hist(positive['age'],6)
ax[1].hist(negative['age'],6)
ax[2].hist(df['age'],6)

ax[0].set(ylim=(0,400))
ax[1].set(ylim=(0,400))
ax[2].set(ylim=(0,700))

ax[0].set_title('age - positive')
ax[1].set_title('age - negative')
ax[2].set_title('age - all classes')
plt.tight_layout()
plt.show()

positive['troponin_cut']=pd.cut(positive['troponin'],3,labels=['low','medium','high'])
negative['troponin_cut']=pd.cut(negative['troponin'],3,labels=['low','medium','high'])
val_count_pos=positive['troponin_cut'].value_counts()
val_count_neg=negative['troponin_cut'].value_counts()

pos_count=pd.DataFrame({'troponin_level':val_count_pos.index,'count':val_count_pos.values})
neg_count=pd.DataFrame({'troponin_level':val_count_neg.index,'count':val_count_neg.values})


pos_count['troponin_level'] = pd.Categorical(pos_count['troponin_level'], categories=['low','medium','high'], ordered=True)
pos_count.sort_values('troponin_level', inplace=True)

neg_count['troponin_level'] = pd.Categorical(neg_count['troponin_level'], categories=['low','medium','high'], ordered=True)
neg_count.sort_values('troponin_level', inplace=True)

pos_filtred=pos_count[pos_count['troponin_level'].isin(['high','medium'])]
neg_filtred=neg_count[neg_count['troponin_level'].isin(['high','medium'])]

plt.barh(pos_filtred.iloc[:,0],pos_filtred.iloc[:,1],label='positive')
plt.barh(neg_filtred.iloc[:,0],neg_filtred.iloc[:,1],label='negative')
plt.title('troponin level')
plt.legend()
plt.show()

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

y.value_counts()

x_kcm=df['class']
y_kcm=df['kcm']

plt.scatter(x_kcm,y_kcm)
plt.title('kcm-distribution')

#analyzing correlation

y.replace({'positive':1,'negative':0},inplace=True)

    
from scipy.stats import pointbiserialr
for i in X.columns:
    correlation, p_value = pointbiserialr(X[i], y)
    print(f"Point Biserial Correlation between {i} and class is {correlation} with a p-value of {p_value}")


#--------------------------

### Machine learning

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import xgboost

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report,ConfusionMatrixDisplay

classificators = [
     ('RF10',RandomForestClassifier(max_depth=2,n_estimators=10)),
     ('RF100',RandomForestClassifier(n_estimators=100)),
     ('Grad_Boost100',GradientBoostingClassifier(n_estimators=100)),
     ('Ada_Boost100',AdaBoostClassifier(n_estimators=100)),
     ('XG_Boost',xgboost.XGBClassifier())
     ]


tsize=0.25
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=tsize)


for name,clf in classificators:

    clf.fit(X_train,y_train)
    
    y_pred=clf.predict(X_test)   
    print(name)
    print("Accuracy:",accuracy_score(y_test, y_pred))
    
    print(confusion_matrix(y_test, y_pred))
    
    print(classification_report(y_test, y_pred))
    
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = ['positive','negative'])
    fig, ax = plt.subplots(figsize=(4,4))
    cm_display.plot(ax=ax)
    plt.title(name)
    plt.show()



##polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_poly,y,test_size=tsize)

    
for name,clf_poly in classificators:

    clf_poly.fit(X_train,y_train)
    
    y_pred=clf_poly.predict(X_test)   
    print(name)
    print("Accuracy:",accuracy_score(y_test, y_pred))
    
    print(confusion_matrix(y_test, y_pred))
    
    print(classification_report(y_test, y_pred))
    
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = ['positive','negative'])
    fig, ax = plt.subplots(figsize=(4,4))
    cm_display.plot(ax=ax)
    plt.title(name)
    plt.show()



##gridsearch

#feature selection 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

#selecting the top 3 features using the chi-squared test
k_best = SelectKBest(score_func=chi2, k=3)
X_best = k_best.fit_transform(X, y)
#using f_classif
k_best2 = SelectKBest(score_func=f_classif, k=3)
X_best2 = k_best2.fit_transform(X, y)

#displaying the results of selected features
print('selected columns: ',X.columns[k_best.get_support()])
print('selected columns v2: ',X.columns[k_best2.get_support()])


X_reduced=X[['age','kcm','troponin']]
X_train,X_test,y_train,y_test=train_test_split(X_reduced,y,test_size=tsize)

from sklearn.model_selection import GridSearchCV

param= {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 4, 5],            
        'learning_rate': [0.01, 0.1, 0.2], 
        'subsample': [0.5,0.8, 1.0],
        'colsample_bytree':[0.5,0.8, 1.0],
        'alpha':[0.1,1,3],
        'lambda':[0.1,1,3]
        }
gs=GridSearchCV(xgboost.XGBClassifier(), param_grid=param)

gs.fit(X_train,y_train)
print('best parameters: ',gs.best_params_)
print('best score: ',gs.best_score_)

print('grid search score:' ,gs.score(X_test,y_test))


y_pred=gs.predict(X_test)
cr=classification_report(y_test, y_pred)



#-----------------------

###Training the chosen model on the entire dataset
clf.fit(X,y)

### using the chosen model

import numpy as np

def get_parameters():
    while True:
        gender= input("type your gender(1-Male, 0-Female): ")
        if gender in ('1','0'):
            gender=int(gender)
            break
        else:
            print('type 1 or 0')
    
    age= input("type your age: ")
    impulse= input("type your impulse: ")
    pressure_h= input("type your pressure - higher value: ")
    pressure_l= input("type your pressure - lower value: ")
    glucose= input("type your glucose: ")
    kcm= input("type your kcm: ")
    troponin= input("type your troponin: ")
    return age,gender,impulse,pressure_h,pressure_l,glucose,kcm,troponin



param=get_parameters()
data=np.array([param],dtype=float)
result=clf.predict(data)
print(result)



