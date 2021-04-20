# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:32:56 2020

@author: Uni361004

"""

### 1 LIBRARIES ### ----------------------------------------------------------

import os 
import pandas as pd 
import numpy as np 
import matplotlib # to change backend for 3d PCA visualization
original_backend = matplotlib.rcParams['backend']
original_backend
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.svm import SVC, SVR
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score, make_scorer, adjusted_rand_score, classification_report, confusion_matrix

# after imbalanced-learn installation 
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline # equal to sklearn pipeline but allows for SMOTE step
 

### 2 EXPLORATORY DATA ANALYSIS ### ------------------------------------------

# 2.1 DATASET PREVIEW #

data_file_path = os.path.abspath('./winequality-white.csv') 
data = pd.read_csv(data_file_path, sep=";", header=0)
pd.set_option('display.max_columns',100) 
#pd.set_option('display.width', 150) # dependently on console window size
data
data.info()

# 2.2 QUALITY #

plt.figure(figsize=(12,6))
sns.countplot(y='quality', data=data, orient='h',order=range(9,2,-1))
data['quality'].value_counts()

# 2.3 FIXED ACIDITY #

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
density = sns.kdeplot(data=data, x='fixed acidity', shade = True, ax=ax1)
density.annotate(
    "\nMean:\n%.4f\n\nDev. Std:\n%.4f\n\nSkewness:\n%.4f\n\nKurtosis:\n%.4f\n"%(
        data['fixed acidity'].mean(),
        data['fixed acidity'].std(),
        data['fixed acidity'].skew(),
        data['fixed acidity'].kurtosis()
        ),
    xy=(12,0.4),bbox=dict(boxstyle='round', fc='0.8'),va='center')
density
violin=sns.violinplot(data=data, x='fixed acidity', y='quality', orient="h", order=range(9,2,-1), ax=ax2,type='violin')
corr,_=pearsonr(data["fixed acidity"], data["quality"])
MI1 = mutual_info_classif(data["fixed acidity"].to_numpy().reshape(-1,1), data["quality"],random_state=0)
violin.annotate("\nPearson:\n%.4f\n\nMutual Info:\n%.4f\n"%(corr,MI1),xy=(12,1),bbox=dict(boxstyle='round', fc='0.8'),va='center')
violin

# 2.4 VOLATILE ACIDITY #

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
density = sns.kdeplot(data=data, x='volatile acidity', shade = True, ax=ax1)
density.annotate("\nMean:\n%.4f\n\nDev. Std:\n%.4f\n\nSkewness:\n%.4f\n\nKurtosis:\n%.4f\n"%(data['volatile acidity'].mean(),data['volatile acidity'].std(),data['volatile acidity'].skew(),data['volatile acidity'].kurtosis()),xy=(0.8,3),bbox=dict(boxstyle='round', fc='0.8'),va='center')
density
violin=sns.violinplot(data=data, x='volatile acidity', y='quality', orient="h", order=range(9,2,-1), ax=ax2,type='violin')
corr,_=pearsonr(data["volatile acidity"], data["quality"])
MI2 = mutual_info_classif(data["volatile acidity"].to_numpy().reshape(-1,1), data["quality"],random_state=0)
violin.annotate("\nPearson:\n%.4f\n\nMutual Info:\n%.4f\n"%(corr,MI2),xy=(1,1),bbox=dict(boxstyle='round', fc='0.8'),va='center')
violin

# 2.5 CITRIC ACID #

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
density = sns.kdeplot(data=data, x='citric acid', shade = True, ax=ax1)
density.annotate("\nMean:\n%.4f\n\nDev. Std:\n%.4f\n\nSkewness:\n%.4f\n\nKurtosis:\n%.4f\n"%(data['citric acid'].mean(),data['citric acid'].std(),data['citric acid'].skew(),data['citric acid'].kurtosis()),xy=(1.25,3),bbox=dict(boxstyle='round', fc='0.8'),va='center')
density
violin=sns.violinplot(data=data, x='citric acid', y='quality', orient="h", order=range(9,2,-1), ax=ax2,type='violin')
corr,_=pearsonr(data["citric acid"], data["quality"])
MI3 = mutual_info_classif(data["citric acid"].to_numpy().reshape(-1,1), data["quality"],random_state=0)
violin.annotate("\nPearson:\n%.4f\n\nMutual Info:\n%.4f\n"%(corr,MI3),xy=(1.25,1),bbox=dict(boxstyle='round', fc='0.8'),va='center')
violin

# 2.6 RESIDUAL SUGAR #

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
density = sns.kdeplot(data=data, x='residual sugar', shade = True, ax=ax1)
density.annotate("\nMean:\n%.4f\n\nDev. Std:\n%.4f\n\nSkewness:\n%.4f\n\nKurtosis:\n%.4f\n"%(data['residual sugar'].mean(),data['residual sugar'].std(),data['residual sugar'].skew(),data['residual sugar'].kurtosis()),xy=(50,0.10),bbox=dict(boxstyle='round', fc='0.8'),va='center')
density
violin=sns.violinplot(data=data, x='residual sugar', y='quality', orient="h", order=range(9,2,-1), ax=ax2,type='violin')
corr,_=pearsonr(data["residual sugar"], data["quality"])
MI4 = mutual_info_classif(data["residual sugar"].to_numpy().reshape(-1,1), data["quality"],random_state=0)
violin.annotate("\nPearson:\n%.4f\n\nMutual Info:\n%.4f\n"%(corr,MI4),xy=(50,1),bbox=dict(boxstyle='round', fc='0.8'),va='center')
violin

# 2.7 CHLORIDES #

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
density = sns.kdeplot(data=data, x='chlorides', shade = True, ax=ax1)
density.annotate("\nMean:\n%.4f\n\nDev. Std:\n%.4f\n\nSkewness:\n%.4f\n\nKurtosis:\n%.4f\n"%(data['chlorides'].mean(),data['chlorides'].std(),data['chlorides'].skew(),data['chlorides'].kurtosis()),xy=(0.25,25),bbox=dict(boxstyle='round', fc='0.8'),va='center')
density
violin=sns.violinplot(data=data, x='chlorides', y='quality', orient="h", order=range(9,2,-1), ax=ax2,type='violin')
corr,_=pearsonr(data["chlorides"], data["quality"])
MI5 = mutual_info_classif(data["chlorides"].to_numpy().reshape(-1,1), data["quality"],random_state=0)
violin.annotate("\nPearson:\n%.4f\n\nMutual Info:\n%.4f\n"%(corr,MI5),xy=(0.25,1),bbox=dict(boxstyle='round', fc='0.8'),va='center')
violin

# 2.8 FREE SULFUR DIOXIDE #

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
density = sns.kdeplot(data=data, x='free sulfur dioxide', shade = True, ax=ax1)
density.annotate("\nMean:\n%.4f\n\nDev. Std:\n%.4f\n\nSkewness:\n%.4f\n\nKurtosis:\n%.4f\n"%(data['free sulfur dioxide'].mean(),data['free sulfur dioxide'].std(),data['free sulfur dioxide'].skew(),data['free sulfur dioxide'].kurtosis()),xy=(200,0.015),bbox=dict(boxstyle='round', fc='0.8'),va='center')
density
violin=sns.violinplot(data=data, x='free sulfur dioxide', y='quality', orient="h", order=range(9,2,-1), ax=ax2,type='violin')
corr,_=pearsonr(data["free sulfur dioxide"], data["quality"])
MI6 = mutual_info_classif(data["free sulfur dioxide"].to_numpy().reshape(-1,1), data["quality"],random_state=0)
violin.annotate("\nPearson:\n%.4f\n\nMutual Info:\n%.4f\n"%(corr,MI6),xy=(250,1),bbox=dict(boxstyle='round', fc='0.8'),va='center')
violin

# 2.9 TOTAL SULFUR DIOXIDE #

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
density = sns.kdeplot(data=data, x='total sulfur dioxide', shade = True, ax=ax1)
density.annotate("\nMean:\n%.4f\n\nDev. Std:\n%.4f\n\nSkewness:\n%.4f\n\nKurtosis:\n%.4f\n"%(data['total sulfur dioxide'].mean(),data['total sulfur dioxide'].std(),data['total sulfur dioxide'].skew(),data['total sulfur dioxide'].kurtosis()),xy=(300,0.006),bbox=dict(boxstyle='round', fc='0.8'),va='center')
density
violin=sns.violinplot(data=data, x='total sulfur dioxide', y='quality', orient="h", order=range(9,2,-1), ax=ax2,type='violin')
corr,_=pearsonr(data["total sulfur dioxide"], data["quality"])
MI7 = mutual_info_classif(data["total sulfur dioxide"].to_numpy().reshape(-1,1), data["quality"],random_state=0)
violin.annotate("\nPearson:\n%.4f\n\nMutual Info:\n%.4f\n"%(corr,MI7),xy=(400,1),bbox=dict(boxstyle='round', fc='0.8'),va='center')
violin

# 2.10 DENSITY #

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
density = sns.kdeplot(data=data, x='density', shade = True, ax=ax1)
density.annotate("\nMean:\n%.4f\n\nDev. Std:\n%.4f\n\nSkewness:\n%.4f\n\nKurtosis:\n%.4f\n"%(data['density'].mean(),data['density'].std(),data['density'].skew(),data['density'].kurtosis()),xy=(1.02,80),bbox=dict(boxstyle='round', fc='0.8'),va='center')
density
violin=sns.violinplot(data=data, x='density', y='quality', orient="h", order=range(9,2,-1), ax=ax2,type='violin')
corr,_=pearsonr(data["density"], data["quality"])
MI8 = mutual_info_classif(data["density"].to_numpy().reshape(-1,1), data["quality"],random_state=0)
violin.annotate("\nPearson:\n%.4f\n\nMutual Info:\n%.4f\n"%(corr,MI8),xy=(1.02,1),bbox=dict(boxstyle='round', fc='0.8'),va='center')
violin

# 2.11 PH #

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
pH = sns.kdeplot(data=data, x='pH', shade = True, ax=ax1)
pH.annotate("\nMean:\n%.4f\n\nDev. Std:\n%.4f\n\nSkewness:\n%.4f\n\nKurtosis:\n%.4f\n"%(data['pH'].mean(),data['pH'].std(),data['pH'].skew(),data['pH'].kurtosis()),xy=(3.6,2),bbox=dict(boxstyle='round', fc='0.8'),va='center')
pH
violin=sns.violinplot(data=data, x='pH', y='quality', orient="h", order=range(9,2,-1), ax=ax2,type='violin')
corr,_=pearsonr(data["pH"], data["quality"])
MI9 = mutual_info_classif(data["pH"].to_numpy().reshape(-1,1), data["quality"],random_state=0)
violin.annotate("Pearson:\n%.4f\n\nMutual Info:\n%.4f"%(corr,MI9),xy=(2.6,0),bbox=dict(boxstyle='round', fc='0.8'),va='center')
violin

# 2.12 SULPHATES #

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
density = sns.kdeplot(data=data, x='sulphates', shade = True, ax=ax1)
density.annotate("\nMean:\n%.4f\n\nDev. Std:\n%.4f\n\nSkewness:\n%.4f\n\nKurtosis:\n%.4f\n"%(data['sulphates'].mean(),data['sulphates'].std(),data['sulphates'].skew(),data['sulphates'].kurtosis()),xy=(0.8,2.5),bbox=dict(boxstyle='round', fc='0.8'),va='center')
density
violin=sns.violinplot(data=data, x='sulphates', y='quality', orient="h", order=range(9,2,-1), ax=ax2,type='violin')
corr,_=pearsonr(data["sulphates"], data["quality"])
MI10 = mutual_info_classif(data["sulphates"].to_numpy().reshape(-1,1), data["quality"],random_state=0)
violin.annotate("Pearson:\n%.4f\n\nMutual Info:\n%.4f"%(corr,MI10),xy=(1,0),bbox=dict(boxstyle='round', fc='0.8'),va='center')
violin

# 2.13 ALCOHOL #

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
density = sns.kdeplot(data=data, x='alcohol', shade = True, ax=ax1)
density.annotate("\nMean:\n%.4f\n\nDev. Std:\n%.4f\n\nSkewness:\n%.4f\n\nKurtosis:\n%.4f\n"%(data['alcohol'].mean(),data['alcohol'].std(),data['alcohol'].skew(),data['alcohol'].kurtosis()),xy=(13,0.275),bbox=dict(boxstyle='round', fc='0.8'),va='center')
density
violin=sns.violinplot(data=data, x='alcohol', y='quality', orient="h", order=range(9,2,-1), ax=ax2,type='violin')
corr,_=pearsonr(data["alcohol"], data["quality"])
MI11 = mutual_info_classif(data["alcohol"].to_numpy().reshape(-1,1), data["quality"],random_state=0)
violin.annotate("Pearson:\n%.4f\n\nMutual Info:\n%.4f"%(corr,MI11),xy=(6.5,0),bbox=dict(boxstyle='round', fc='0.8'),va='center')
violin

# 2.14 NORMALIZATION AND PRINCIPAL COMPONENT ANALYSIS #

X = MinMaxScaler().fit_transform(data.loc[:,:"alcohol"])
y = data['quality']
pd.DataFrame(X,columns=data.columns[:-1])

pd.DataFrame({"PC cumulative variance explained" : np.cumsum(PCA().fit(X, y).explained_variance_ratio_)},
             index = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11'])

pca = PCA().fit_transform(X, y)
matplotlib.use('Qt5Agg')  # provide GUI for 3d interactive plots, Qt4Agg is an alternative
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pca[:,0],
                pca[:,1],
                pca[:,2],
                c=y,
                s=75,
                cmap='seismic',
                linewidth=0.5,
                edgecolor='black')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
plt.show(block=True)

matplotlib.use(original_backend) # returning to original backend, normal plotting
pd.DataFrame({
    'correlation with PC1' : pd.DataFrame(X).corrwith(pd.Series(pca[:,0])).to_numpy(),
    'correlation with quality' : pd.DataFrame(X).corrwith(pd.Series(y)).to_numpy(),
    'M.I. with quality' : np.hstack([MI1,MI2,MI3,MI4,MI5,MI6,MI7,MI8,MI9,MI10,MI11]),},
    index=data.columns[:-1]
    )


### 3 TRAINING ### -------------------------------------------------------------

# 3.1 TRAINING/TEST SPLIT #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state=0,
    stratify=y)

# 3.2 EXTRA TREES CLASSIFIER #

parameters = {
    'n_estimators':[2,5,10,20,50,100,200,500,1000],
    'bootstrap':[True, False],
    'criterion':['gini','entropy'],
    'max_features':[2,'sqrt', None],
    'max_depth':[5, 10, None],
    'class_weight':[None,'balanced','balanced_subsample']
}

ensemble = GridSearchCV(
    ExtraTreesClassifier(random_state=0),
    parameters,
    return_train_score=True,
    cv=3,
    verbose=2,
    n_jobs=-1).fit(X_train, y_train) # n_jobs=-1 uses all CPU; 6.5 min on i7-8750H 

print("The best ensemble is: ")
print(ensemble.best_params_)
print("The accuracy obtained by the best ensemble is: %.5f"%ensemble.best_score_)

pd.DataFrame({
    'feature importance' : ensemble.best_estimator_.feature_importances_,
    'corr. with quality' : pd.DataFrame(X).corrwith(pd.Series(y)).to_numpy(),
    'M.I. with quality' : np.hstack([MI1,MI2,MI3,MI4,MI5,MI6,MI7,MI8,MI9,MI10,MI11]),
    'corr. with PC1' : pd.DataFrame(X).corrwith(pd.Series(pca[:,0])).to_numpy()},
    index=data.columns[:-1]
    ).sort_values(by='feature importance',ascending=False)

RFECV(ensemble.best_estimator_,cv=3,min_features_to_select=2).fit(X_train,y_train).support_

# 3.3 K-NEAREST NEIGHBORS #

parameters = {
    'n_neighbors':[1,2,3,4,5,6,7,8,9,10,20,40,50,80,100,120,150,180,200],
    'weights':['uniform', 'distance'],
    'p':[1,2]
}

KNN = GridSearchCV(KNeighborsClassifier(),
                     parameters,
                     return_train_score=True,
                     cv=3,
                     verbose=2,
                     n_jobs=-1).fit(X_train, y_train) # 45 sec. on i7-8750H 


print("The best k-nearest neighbors parameters are: ")
print(KNN.best_params_)
print("The accuracy obtained by the best k-nearest neighbors model is: %.5f"%KNN.best_score_)

# 3.4 SUPPORT VECTOR MACHINES #

pipeline = Pipeline([('clf', SVC())])

parameters = [
    {
     'clf': (SVC(),),
     'clf__C': [0.001, 0.01, 0.1, 1,10],
     'clf__kernel': ['linear'],
     'clf__class_weight':[None,'balanced']
     },
    {
     'clf': (SVC(),),
     'clf__C': [0.001, 0.01, 0.1, 1,10],
     'clf__kernel': ['poly'],
     'clf__degree': [2,3,5],
     'clf__class_weight':[None,'balanced']
     },
    {
     'clf': (SVC(),),
     'clf__C': [0.001, 0.01, 0.1, 1,10],
     'clf__kernel': ['rbf'],
     'clf__gamma': [0.5,1,2,'scale','auto'],
     'clf__class_weight':[None,'balanced']
     }
]

SVM = GridSearchCV(pipeline,         
                     parameters,
                     return_train_score=True,
                     cv=3,
                     verbose=2,
                     n_jobs=-1).fit(X_train, y_train)   # 1.3min on i7-8750H 

print("The best SVM parameters are: ")
print(SVM.best_params_)
print('The accuracy obtained by the best SVM is: %.5f'%SVM.best_score_)

# 3.5 MODEL COMPARISON #

test_scores = [ensemble.best_score_, KNN.best_score_, SVM.best_score_]
training_scores = [
    ensemble.cv_results_['mean_train_score'][ensemble.best_index_],
    KNN.cv_results_['mean_train_score'][KNN.best_index_],
    SVM.cv_results_['mean_train_score'][SVM.best_index_]
    ]
model = ['Extra Trees','K-Nearest Neighbors','SVM']
barwidth = 0.25        
barset1=np.arange(len(training_scores))  
barset2=[value + barwidth for value in barset1]
plt.figure(figsize=(10,8))
plt.bar(barset1,training_scores,width=barwidth, color="tab:orange", edgecolor="white", label="training set")
plt.bar(barset2,test_scores, width=barwidth, color="tab:blue",edgecolor="white", label="test set")
plt.xticks([value + 0.5*barwidth for value in range(len(training_scores))], model) 
plt.ylim(0.5,1)
plt.title("Accuracy")
plt.legend()
plt.figure()

# 3.6 TRAINING IN LDA SPACE #

LDA = LinearDiscriminantAnalysis(n_components=2).fit_transform(X_train,y_train)
plt.figure(figsize=(10,8))
ax = sns.scatterplot(x=LDA[:,0], y=LDA[:,1],hue=y_train,palette='coolwarm')
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')

pipeline = Pipeline([
    ('lda', LinearDiscriminantAnalysis()),
    ('clf', ExtraTreesClassifier(random_state=0))])
# the choice of algorithm in 'clf' step in pipeline definition is irrelevant, it will be re-writed each time

parameters = [{
    'lda__n_components':[2,6],
    'clf' : (ExtraTreesClassifier(random_state=0),),
    'clf__n_estimators':[2,5,10,20,50,100,200,500,1000],
    'clf__bootstrap':[True, False],
    'clf__criterion':['gini','entropy'],
    'clf__max_features':[2,'sqrt', None],
    'clf__max_depth':[5, 10, None],
    'clf__class_weight':[None,'balanced','balanced_subsample']
},{
    'lda__n_components':[2,6],
    'clf' : (KNeighborsClassifier(),),
    'clf__n_neighbors':[1,2,3,4,5,6,7,8,9,10,20,40,50,80,100,120,150,180,200],
    'clf__weights':['uniform', 'distance'],
    'clf__p':[1,2]
},{
   'lda__n_components':[2,6],
   'clf': (SVC(),),
   'clf__C': [0.001, 0.01, 0.1, 1, 10],
   'clf__kernel': ['linear'],
   'clf__class_weight':[None,'balanced']
},{
   'lda__n_components':[2,6],
   'clf': (SVC(),),
   'clf__C': [0.001, 0.01, 0.1, 1, 10],
   'clf__kernel': ['poly'],
   'clf__degree': [2,3,5],
   'clf__class_weight':[None,'balanced']
},{
   'lda__n_components':[2,6],
   'clf': (SVC(),),
   'clf__C': [0.001, 0.01, 0.1, 1, 10],
   'clf__kernel': ['rbf'],
   'clf__gamma': [0.5,1,2,'scale','auto'],
   'clf__class_weight':[None,'balanced']
}]

dim_red_classifier = GridSearchCV(pipeline,         
                     parameters,
                     cv=3,
                     verbose=2,
                     n_jobs=-1).fit(X_train, y_train) # 13.2 min on i7-8750H 

print("The best classifier in LDA space is: ")
print(dim_red_classifier.best_params_)
print('The accuracy obtained by the best classifier in LDA space is: %.5f'%dim_red_classifier.best_score_)

results_df = pd.DataFrame.from_dict(dim_red_classifier.cv_results_)
LDA_2components_df = results_df.loc[results_df['param_lda__n_components'] == 2]
LDA_2components_best_index = LDA_2components_df.loc[LDA_2components_df.rank_test_score == LDA_2components_df.rank_test_score.min(),('params', 'mean_test_score')].first_valid_index()
print('The best classifier in 2-dimensional LDA space is: ')
print(results_df.loc[LDA_2components_best_index,'params'])
print('The accuracy obtained by the best classifier in 2-dimensional LDA space is: %.5f'%results_df.loc[LDA_2components_best_index,'mean_test_score'])

# 3.7 REGRESSION #

pipeline = Pipeline([('reg', ExtraTreesRegressor(random_state=0))])

parameters = [{
    'reg' : (ExtraTreesRegressor(random_state=0),),
    'reg__n_estimators':[2,5,10,20,50,100,200,500,1000],
    'reg__bootstrap':[True, False],
    'reg__max_features':[2,'sqrt', None],
    'reg__max_depth':[5, 10, None]
},{
    'reg' : (KNeighborsRegressor(),),
    'reg__n_neighbors':[1,2,3,4,5,6,7,8,9,10,20,40,50,80,100,120,150,180,200],
    'reg__weights':['uniform', 'distance'],
    'reg__p':[1,2]
},{
   'reg': (SVR(),),
   'reg__C': [0.001, 0.01, 0.1, 1, 10],
   'reg__kernel': ['linear']
},{
   'reg': (SVR(),),
   'reg__C': [0.001, 0.01, 0.1, 1, 10],
   'reg__kernel': ['poly'],
   'reg__degree': [2,3,5]
},{
   'reg': (SVR(),),
   'reg__C': [0.001, 0.01, 0.1, 1, 10],
   'reg__kernel': ['rbf'],
   'reg__gamma': [0.5,1,2,'scale','auto']
}]

def mod_accuracy(y,y_hat):   
    y_hat[y_hat<3]=3
    y_hat[y_hat>8]=8
    y_hat=np.round(y_hat).astype(int)
    return accuracy_score(y,y_hat)

regressor = GridSearchCV(pipeline,         
                     parameters,
                     cv=3,
                     verbose=2,
                     n_jobs=-1,
                     scoring=make_scorer(mod_accuracy)).fit(X_train, y_train) # 6.5 min. on i7-8750H 

print("The best regressor is: ")
print(regressor.best_params_)
print('The accuracy obtained by the best regressor is: %.5f'%regressor.best_score_)
print('The root mean squared error obtained by the best regressor is: %.5f'%-np.average(cross_val_score(regressor.best_estimator_ ,X_train ,y_train , scoring='neg_root_mean_squared_error', cv=3)))
print("The R2 score obtained by the best regressor is: %.5f"%np.average(cross_val_score(regressor.best_estimator_ ,X_train ,y_train , scoring='r2', cv=3)))

# 3.8 SMOTE AND BALANCED ACCURACY #

pipeline = Pipeline(
    [('SMOTE', BorderlineSMOTE(k_neighbors=1,random_state=0)),
     ('clf', ExtraTreesClassifier(random_state=0))]
)

parameters = [{
    'SMOTE__sampling_strategy':[{},{8:300, 4:300, 3:300, 9:300}, 'auto'],
    'clf' : (ExtraTreesClassifier(random_state=0),),
    'clf__n_estimators':[2,5,10,20,50,100,200,500,1000],
    'clf__bootstrap':[True, False],
    'clf__criterion':['gini','entropy'],
    'clf__max_features':[2,'sqrt', None],
    'clf__max_depth':[5, 10, None],
    'clf__class_weight':[None,'balanced','balanced_subsample']
},{
    'SMOTE__sampling_strategy':[{},{8:300, 4:300, 3:300, 9:300}, 'auto'],
    'clf' : (KNeighborsClassifier(),),
    'clf__n_neighbors':[1,2,3,4,5,6,7,8,9,10,20,40,50,80,100,120,150,180,200],
    'clf__weights':['uniform', 'distance'],
    'clf__p':[1,2]
},{
   'SMOTE__sampling_strategy':[{},{8:300, 4:300, 3:300, 9:300}, 'auto'],
   'clf': (SVC(),),
   'clf__C': [0.001, 0.01, 0.1, 1, 10],
   'clf__kernel': ['linear'],
   'clf__class_weight':[None,'balanced']
},{
   'SMOTE__sampling_strategy':[{},{8:300, 4:300, 3:300, 9:300}, 'auto'],
   'clf': (SVC(),),
   'clf__C': [0.001, 0.01, 0.1, 1, 10],
   'clf__kernel': ['poly'],
   'clf__degree': [2,3,5],
   'clf__class_weight':[None,'balanced']
},{
   'SMOTE__sampling_strategy':[{},{8:300, 4:300, 3:300, 9:300}, 'auto'],
   'clf': (SVC(),),
   'clf__C': [0.001, 0.01, 0.1, 1, 10],
   'clf__kernel': ['rbf'],
   'clf__gamma': [0.5,1,2,'scale','auto'],
   'clf__class_weight':[None,'balanced']
}]

balanced_classifier = GridSearchCV(pipeline,         
                     parameters,
                     cv=3,
                     verbose=2,
                     n_jobs=-1,
                     scoring='balanced_accuracy').fit(X_train, y_train) # 27.1 min on i7-8750H 

print("The best balanced classifier is: ")
print(balanced_classifier.best_params_)
print('The balanced accuracy obtained by the best balanced classifier is: %.5f'%balanced_classifier.best_score_)
balanced_classifier_standard_accuracy = np.average(cross_val_score(balanced_classifier.best_estimator_ ,X_train ,y_train ,cv=3, scoring='accuracy'))
print("The accuracy of the best balanced classifier is: %.5f"%balanced_classifier_standard_accuracy)

pd.DataFrame(
    {'accuracy':[ensemble.best_score_,
                 balanced_classifier_standard_accuracy],
     'balanced accuracy':[np.average(cross_val_score(ensemble.best_estimator_ ,X_train ,y_train , scoring='balanced_accuracy', cv=3)),
                          balanced_classifier.best_score_],
     'macro f1':[np.average(cross_val_score(ensemble.best_estimator_ ,X_train ,y_train , scoring='f1_macro', cv=3)),
                         np.average(cross_val_score(balanced_classifier.best_estimator_ ,X_train ,y_train , scoring='f1_macro', cv=3))],
     'weighted f1':[np.average(cross_val_score(ensemble.best_estimator_ ,X_train ,y_train , scoring='f1_weighted', cv=3)),
                         np.average(cross_val_score(balanced_classifier.best_estimator_ ,X_train ,y_train , scoring='f1_weighted', cv=3))]},
     index=['classifier','balanced classifier']
     )

# 3.9 LOW/MID/HIGH TARGET CLASSIFICATION #

y_three_train = np.select(
    [y_train<6, y_train==6, y_train>6],
    ['low','mid','high'])

plt.figure(figsize=(12,6))
sns.countplot(y=y_three_train,orient='h').set_title('quality (three levels)')

pd.DataFrame(np.unique(y_three_train,return_counts=True)[1],
             index=np.unique(y_three_train,return_counts=True)[0],
             columns=['quality (three levels)'])

pipeline = Pipeline([('clf', ExtraTreesClassifier(random_state=0))])

parameters = [{
    'clf' : (ExtraTreesClassifier(random_state=0),),
    'clf__n_estimators':[2,5,10,20,50,100,200,500,1000],
    'clf__bootstrap':[True, False],
    'clf__criterion':['gini','entropy'],
    'clf__max_features':[2,'sqrt', None],
    'clf__max_depth':[5, 10, None],
    'clf__class_weight':[None,'balanced','balanced_subsample']
},{
    'clf' : (KNeighborsClassifier(),),
    'clf__n_neighbors':[1,2,3,4,5,6,7,8,9,10,20,40,50,80,100,120,150,180,200],
    'clf__weights':['uniform', 'distance'],
    'clf__p':[1,2]
},{
   'clf': (SVC(),),
   'clf__C': [0.001, 0.01, 0.1, 1, 10],
   'clf__kernel': ['linear'],
   'clf__class_weight':[None,'balanced']
},{
   'clf': (SVC(),),
   'clf__C': [0.001, 0.01, 0.1, 1, 10],
   'clf__kernel': ['poly'],
   'clf__degree': [2,3,5],
   'clf__class_weight':[None,'balanced']
},{
   'clf': (SVC(),),
   'clf__C': [0.001, 0.01, 0.1, 1, 10],
   'clf__kernel': ['rbf'],
   'clf__gamma': [0.5,1,2,'scale','auto'],
   'clf__class_weight':[None,'balanced']
}]

three_values_classifier = GridSearchCV(pipeline,         
                     parameters,
                     cv=3,
                     verbose=2,
                     n_jobs=-1).fit(X_train, y_three_train) # 6.5 min on i7-8750H 

print("For the simplified problem, The best classifier is: ")
print(three_values_classifier.best_params_)
print('For the simplified problem, the accuracy obtained by the best classifier is: %.5f'%three_values_classifier.best_score_)

# 3.10 UNSUPERVISED AND SEMI-SUPERVISED LEARNING #

names = []
rand = []
for linkage in ['single','complete','average','ward']:
    for affinity in ['euclidean', 'manhattan']:
        if linkage=='ward' and affinity=='manhattan': # substitute this combination with KMeans
            names.append('KMeans')
            rand.append(adjusted_rand_score(
            y_train,
            KMeans(random_state=0,n_clusters=6).fit(X_train).labels_)
                )
            continue
        names.append('AgglomerativeClustering, with ' + linkage + ' linkage and ' + affinity + ' affinity')
        rand.append(adjusted_rand_score(
            y_train,
            AgglomerativeClustering(n_clusters=6,linkage=linkage,affinity=affinity).fit(X_train).labels_
            ))
print('With Adjusted Rand index equal to %.5f, the best clustering algorithm is '%max(rand) + names[rand.index(max(rand))])

names3 = []
rand3 = []
for linkage in ['single','complete','average','ward']:
    for affinity in ['euclidean', 'manhattan']:
        if linkage=='ward' and affinity=='manhattan': # substitute this combination with KMeans
            names3.append('KMeans')
            rand3.append(adjusted_rand_score(
            y_three_train,
            KMeans(random_state=0,n_clusters=3).fit(X_train).labels_)
                )
            continue
        names3.append('AgglomerativeClustering, with ' + linkage + ' linkage and ' + affinity + ' affinity')
        rand3.append(adjusted_rand_score(
            y_three_train,
            AgglomerativeClustering(n_clusters=3,linkage=linkage,affinity=affinity).fit(X_train).labels_
            ))

print('With Adjusted Rand index equal to %.5f, the best clustering algorithm for three levels quality is '%max(rand3) + names3[rand3.index(max(rand3))])

X_label, X_unlabel, y_label,  y_true = train_test_split(
    X_train,
    y_train,
    train_size=0.2, # labeled size
    random_state=0,
    stratify=y_train)

y_unlabel = np.full((len(y_true),),-1)
X_train_SSL = np.r_[X_label,X_unlabel]
y_train_SSL = np.r_[y_label,y_unlabel]

names = []
accuracy = []
for classifier in [LabelPropagation(), LabelSpreading()]:
    for kernel in ['knn','rbf']:
        if kernel=='knn':
            for n_neighbors in [60,70,90,100,150]:
                if str(type(classifier)).split(".")[-1][:-2]=='LabelPropagation':
                    names.append('LabelPropagation with KNN kernel and K equal to %i'%n_neighbors)
                    accuracy.append(
                        accuracy_score(
                            y_true,
                            LabelPropagation(kernel=kernel,
                                             n_neighbors=n_neighbors
                                             ).fit(X_train_SSL,y_train_SSL).transduction_[len(y_label):]
                            ))
                    print('Computed LabelPropagation with KNN kernel and K equal to %i'%n_neighbors)
                else:
                    for alpha in [0.1,0.2,0.3,0.5,0.7]:
                        names.append('LabelSpreading with KNN kernel, K equal to %i and alpha equal to %.1f'%(n_neighbors,alpha))
                        accuracy.append(
                            accuracy_score(
                                y_true,
                                LabelSpreading(kernel=kernel,
                                               n_neighbors=n_neighbors,
                                               alpha=alpha
                                               ).fit(X_train_SSL,y_train_SSL).transduction_[len(y_label):]
                            ))
                        print('Computed LabelSpreading with KNN kernel, K equal to %i and alpha equal to %.1f'%(n_neighbors,alpha))           
        else:
            for gamma in [0.1,0.2,0.5,1,2,5,10,20,50]:
                if str(type(classifier)).split(".")[-1][:-2]=='LabelPropagation':
                    names.append('LabelPropagation with rbf kernel and gamma equal to %.1f'%gamma)
                    accuracy.append(
                        accuracy_score(
                            y_true,
                            LabelPropagation(kernel=kernel,
                                             gamma=gamma
                                             ).fit(X_train_SSL,y_train_SSL).transduction_[len(y_label):]
                            ))
                    print('Computed LabelPropagation with rbf kernel and gamma equal to %.1f'%gamma)
                else:
                    for alpha in [0.1,0.2,0.3,0.5,0.7]:
                        names.append('LabelSpreading with rbf kernel, gamma equal to %.1f and alpha equal to %.1f'%(gamma,alpha))
                        accuracy.append(
                            accuracy_score(
                                y_true,
                                LabelSpreading(kernel=kernel,
                                               gamma=gamma,
                                               alpha=alpha
                                               ).fit(X_train_SSL,y_train_SSL).transduction_[len(y_label):]
                            ))
                        print('Computed LabelSpreading with rbf kernel, gamma equal to %.1f and alpha equal to %.1f'%(gamma,alpha))    

print('With accuracy score equal to %.5f, the best semi-supervised algorithm is '%max(accuracy) + names[accuracy.index(max(accuracy))])

y_three_label = np.select(
    [y_label<6, y_label==6, y_label>6],
    [0,1,2]) # stands for low, mid, high

y_three_true = np.select(
    [y_true<6, y_true==6, y_true>6],
    [0,1,2])

y_three_train_SSL = np.r_[y_three_label,y_unlabel]

names3 = []
accuracy3 = []
for classifier in [LabelPropagation(), LabelSpreading()]:
    for kernel in ['knn','rbf']:
        if kernel=='knn':
            for n_neighbors in [60,70,90,100,150]:
                if str(type(classifier)).split(".")[-1][:-2]=='LabelPropagation':
                    names3.append('LabelPropagation with KNN kernel and K equal to %i'%n_neighbors)
                    accuracy3.append(
                        accuracy_score(
                            y_three_true,
                            LabelPropagation(kernel=kernel,
                                             n_neighbors=n_neighbors
                                             ).fit(X_train_SSL,y_three_train_SSL).transduction_[len(y_label):]
                            ))
                    print('Computed LabelPropagation with KNN kernel and K equal to %i'%n_neighbors)
                else:
                    for alpha in [0.1,0.2,0.3,0.5,0.7]:
                        names3.append('LabelSpreading with KNN kernel, K equal to %i and alpha equal to %.1f'%(n_neighbors,alpha))
                        accuracy3.append(
                            accuracy_score(
                                y_three_true,
                                LabelSpreading(kernel=kernel,
                                               n_neighbors=n_neighbors,
                                               alpha=alpha
                                               ).fit(X_train_SSL,y_three_train_SSL).transduction_[len(y_label):]
                            ))
                        print('Computed LabelSpreading with KNN kernel, K equal to %i and alpha equal to %.1f'%(n_neighbors,alpha))           
        else:
            for gamma in [0.1,0.2,0.5,1,2,5,10,20,50]:
                if str(type(classifier)).split(".")[-1][:-2]=='LabelPropagation':
                    names3.append('LabelPropagation with rbf kernel and gamma equal to %.1f'%gamma)
                    accuracy3.append(
                        accuracy_score(
                            y_three_true,
                            LabelPropagation(kernel=kernel,
                                             gamma=gamma
                                             ).fit(X_train_SSL,y_three_train_SSL).transduction_[len(y_label):]
                            ))
                    print('Computed LabelPropagation with rbf kernel and gamma equal to %.1f'%gamma)
                else:
                    for alpha in [0.1,0.2,0.3,0.5,0.7]:
                        names3.append('LabelSpreading with rbf kernel, gamma equal to %.1f and alpha equal to %.1f'%(gamma,alpha))
                        accuracy3.append(
                            accuracy_score(
                                y_three_true,
                                LabelSpreading(kernel=kernel,
                                               gamma=gamma,
                                               alpha=alpha
                                               ).fit(X_train_SSL,y_three_train_SSL).transduction_[len(y_label):]
                            ))
                        print('Computed LabelSpreading with rbf kernel, gamma equal to %.1f and alpha equal to %.1f'%(gamma,alpha))    

print('With accuracy score equal to %.5f, the best semi-supervised algorithm for three levels quality is '%max(accuracy3) + names3[accuracy3.index(max(accuracy3))])


### 4 TEST AND CONCLUSION ### ------------------------------------------------

y_pred = ensemble.best_estimator_.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

