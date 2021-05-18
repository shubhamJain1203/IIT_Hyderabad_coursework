#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn. preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


# In[2]:


df1 = pd.read_csv('train_input.csv')
df1.head()


# In[3]:


nulls = df1.isnull().sum()
nulls[nulls > 0]


# In[4]:


def meanImputer(df):
    for i in df:
        df[i].fillna((df[i].median()), inplace=True)
    return df


# In[5]:


df = meanImputer(df1)


# In[6]:


df.head(5)


# In[7]:



non_discrete_feature_cols = ["Feature 9","Feature 10",
                                           "Feature 11","Feature 12","Feature 13","Feature 14","Feature 15",
                                           "Feature 16","Feature 17","Feature 18","Feature 24" ]


# In[8]:


discrete_feature_cols= ["Feature 1 (Discrete)","Feature 2 (Discrete)","Feature 3 (Discrete)",
                                            "Feature 4 (Discrete)","Feature 5 (Discrete)","Feature 6 (Discrete)",
                                            "Feature 7 (Discrete)","Feature 8 (Discrete)","Feature 19 (Discrete)",
                                            "Feature 20 (Discrete)","Feature 21 (Discrete)","Feature 22 (Discrete)",
                                            "Feature 23 (Discrete)"]


# In[9]:


#For outliers
# for i in non_discrete_feature_cols:
#     median = df[i].median()
#     std = df[i].std()
#         #print(std)
#     mean = df[i].mean()
#     #print(std,mean)
#     df.loc[((df[i] > (3*std + mean)) | (df[i] < ( mean - 3*std))), i] = np.nan
#     print(df.isnull().sum(axis = 0))
#     #df.loc[df["Feature 9"] < (2*std - mean), "Feature 9"] = np.nan
#     df.fillna(median,inplace=True)
# df.head()


# In[ ]:





# In[10]:


X = df.drop(["Target Variable (Discrete)"],axis=1)
y = df["Target Variable (Discrete)"]


# In[11]:


y.unique()


# In[ ]:





# In[ ]:





# In[12]:


min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
sc_X = pd.DataFrame(x_scaled)
sc_X.head()


# In[13]:





# In[14]:


X_train, X_valid, y_train, y_valid = train_test_split(sc_X, y, test_size=0.2, random_state=2)


# In[15]:


# Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_valid = sc.transform(X_valid)


# In[ ]:





# In[16]:


#Defining the machine learning models
# model1 = LogisticRegression(max_iter=300)
model2 = DecisionTreeClassifier(criterion="gini", max_depth=9,random_state=1)
model3 = SVC(kernel = 'rbf', C = 5,random_state = 1)
# model4 = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 2)
model5 = RandomForestClassifier(n_estimators=25, max_features="auto",random_state=0, max_depth=11)


# In[17]:


#Training the machine learning models
# model1.fit(X_train, y_train)


# In[18]:


model2.fit(X_train, y_train)


# In[19]:


model3.fit(X_train, y_train)


# In[20]:


#model4.fit(X_train, y_train)


# In[21]:


model5.fit(X_train, y_train)


# In[22]:


# y_pred1 = model1.predict(X_valid)
y_pred2 = model2.predict(X_valid)
y_pred3 = model3.predict(X_valid)
#y_pred4 = model4.predict(X_valid)
y_pred5 = model5.predict(X_valid)


# In[23]:


# cm_LogisticRegression = confusion_matrix(y_valid, y_pred1)
cm_DecisionTree = confusion_matrix(y_valid, y_pred2)
cm_SupportVectorClass = confusion_matrix(y_valid, y_pred3)
#cm_KNN = confusion_matrix(y_valid, y_pred4)
cm_RF = confusion_matrix(y_valid, y_pred5)


# In[24]:


# kfold = model_selection.KFold(n_splits=1, random_state = 0, shuffle=True)
# result1 = model_selection.cross_val_score(model1, X_train, y_train)#, cv=kfold)
result2 = model_selection.cross_val_score(model2, X_train, y_train)#, cv=kfold)
result3 = model_selection.cross_val_score(model3, X_train, y_train)#, cv=kfold)
#result4 = model_selection.cross_val_score(model4, X_train, y_train)#, cv=kfold)
result5 = model_selection.cross_val_score(model5, X_train, y_train)#, cv=kfold)


# In[25]:


#Printing the accuracies achieved in cross-validation
# print('Accuracy of Logistic Regression Model = ',result1.mean())
print('Accuracy of Decision Tree Model = ',result2.mean())
print('Accuracy of Support Vector Machine = ',result3.mean())
# print('Accuracy of k-NN Model = ',result4.mean())
print('Accuracy of Random Forest Model = ',result5.mean())


# In[26]:


#Defining Hybrid Ensemble Learning Model
# create the sub-models
estimators = []

#Defining 5 Logistic Regression Models
# model11 = LogisticRegression(penalty = 'l2', random_state = 0, max_iter=300)
# estimators.append(('logistic1', model11))
# model12 = LogisticRegression(penalty = 'l2', random_state = 0, max_iter=300)
# estimators.append(('logistic2', model12))
# model13 = LogisticRegression(penalty = 'l2', random_state = 0, max_iter=300)
# estimators.append(('logistic3', model13))
# model14 = LogisticRegression(penalty = 'l2', random_state = 0, max_iter=300)
# estimators.append(('logistic4', model14))
# model15 = LogisticRegression(penalty = 'l2', random_state = 0,max_iter=300)
# estimators.append(('logistic5', model15))

#Defining 5 Decision Tree Classifiers
model16 = DecisionTreeClassifier(max_depth = 9,criterion="gini", random_state=1)
estimators.append(('cart1', model16))
model17 = DecisionTreeClassifier(max_depth = 8,criterion="gini", random_state=1)
estimators.append(('cart2', model17))
model18 = DecisionTreeClassifier(max_depth = 7,criterion="entropy",  random_state=1)
estimators.append(('cart3', model18))
model19 = DecisionTreeClassifier(max_depth = 8,criterion="entropy",  random_state=1)
estimators.append(('cart4', model19))
model20 = DecisionTreeClassifier(max_depth = 9,criterion="entropy",  random_state=1)
estimators.append(('cart5', model20))

#Defining 5 Support Vector Classifiers
model21 = SVC(kernel = 'rbf', C = 5,random_state = 42)
estimators.append(('svm1', model21))
model22 = SVC(kernel = 'rbf', C = 5,random_state = 42)
estimators.append(('svm2', model22))
model23 = SVC(kernel='rbf', C = 6,random_state = 42)
estimators.append(('svm3', model23))
model24 = SVC(kernel = 'rbf', C = 4,random_state = 42)
estimators.append(('svm4', model24))
model25 = SVC(kernel = 'rbf', C = 5,random_state = 42)
estimators.append(('svm5', model25))

#Defining 5 K-NN classifiers
# model26 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# estimators.append(('knn1', model26))
# model27 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# estimators.append(('knn2', model27))
# model28 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
# estimators.append(('knn3', model28))
# model29 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 1)
# estimators.append(('knn4', model29))
# model30 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 1)
# estimators.append(('knn5', model30))

#Defining 5 Random Forest
model31 = RandomForestClassifier(n_estimators=25, max_features="auto",random_state=0, max_depth=11)
estimators.append(('RF1', model31))
model32 = RandomForestClassifier(n_estimators=25, max_features="auto",random_state=0, max_depth=11)
estimators.append(('RF2', model32))
model33 = RandomForestClassifier(n_estimators=25, max_features="auto",random_state=0, max_depth=11)
estimators.append(('RF3', model33))
model34 = RandomForestClassifier(n_estimators=25, max_features="auto",random_state=0, max_depth=11)
estimators.append(('RF4',model34))
model35 = RandomForestClassifier(n_estimators=25, max_features="auto",random_state=0, max_depth=11)
estimators.append(('RF5',model35))

# Defining the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_valid)

#Confisuin matrix
cm_HybridEnsembler = confusion_matrix(y_valid, y_pred)
cm_HybridEnsembler


# In[27]:


#Cross-Validation
seed = 7
# kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
results = model_selection.cross_val_score(ensemble, X_train, y_train)#, cv=kfold)
print(results.mean())


# In[28]:


test = pd.read_csv("test_input.csv")


# In[29]:


test.head()


# In[30]:


nulls = test.isnull().sum()
nulls[nulls > 0]


# In[31]:


test_set = meanImputer(test)


# In[32]:


test_set.head()


# In[33]:


min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(test_set)
X_test = pd.DataFrame(x_scaled)
X_test.head()


# In[34]:


# X_test = sc.fit_transform(test_set)


# In[35]:


y_pred_test = ensemble.predict(X_test)
y_pred_test


# In[36]:


output = pd.DataFrame(y_pred_test)
output.insert(0,"ID" , range(1,len(output)+1))
output.rename(columns = {0:"Category"},inplace =True)
output.to_csv('test_ouput.csv', index = False)


# In[ ]:




