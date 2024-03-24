#!/usr/bin/env python
# coding: utf-8

# # Heart Attack Prediction Using Different ML Models

# ### Importing necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### Importing and reading the data.

# In[2]:


data = pd.read_csv('heart.csv')
data.head()


# In[3]:


data.info()


# In[4]:


data.shape


# In[5]:


data.describe()


# ## Information about the data set we got.
# 
# ### Attribute Information
# 

# 1) age

# 
# 2) sex

# 
# 3) chest pain type (4 values)

# 
# 4) resting blood pressure

# 
# 5) serum cholestoral in mg/dl

# 
# 6) fasting blood sugar > 120 mg/dl

# 
# 7) resting electrocardiographic results (values 0,1,2)

# 
# 8) maximum heart rate achieved

# 
# 9) exercise induced angina
# 

# 
# 10) oldpeak = ST depression induced by exercise relative to rest

# 11) the slope of the peak exercise ST segment
# 

# 12) number of major vessels (0-3) colored by flourosopy

# 13) thal: 0 = normal; 1 = fixed defect; 2 = reversable defect

# 14) target: 0= less chance of heart attack 1= more chance of heart attack

# ## EDA

# Finding null values

# In[6]:


print(data.isnull().sum())
# sum will sums up the number of True values (i.e., missing values) along each column.
#the count shows how much missing values are there


# In[7]:


print(data.isnull())
#True means have null values and False means no null values


# In[8]:


print(data['age'].isnull().sum())


# In[9]:


print(data['sex'].isnull().sum())


# Finding Duplicate values

# In[10]:


# Check for duplicate rows in the entire DataFrame
duplicate_rows = data[data.duplicated()]
print("Duplicate rows:")
print(duplicate_rows)


# ## Understanding data with the help of visualization.

# In[11]:


#histogram
data.hist()


# In[12]:


#boxplot
sns.boxplot(x='age', data=data)


# In[13]:


sns.boxplot(x='chol', data=data)


# In[14]:


#coorelation
data.corr()


# ### Important insights we got from correlation.

# There are moderate positive correlations between the target variable (target) and features such as cp (chest pain type), thalach (maximum heart rate achieved), and slope (slope of the peak exercise ST segment).

# Moderate negative correlations exist between the target variable and features like exang (exercise-induced angina) and oldpeak (ST depression induced by exercise relative to rest).

# Some features exhibit correlations among themselves, such as age with trestbps (resting blood pressure) and chol (serum cholesterol levels).

# Features with weak correlations with the target variable may be considered less influential in predicting the target variable.

# In[15]:


#heatmap 
sns.heatmap(data.corr(), annot=True)


# In[16]:


#frequency count
data['age'].value_counts()


# In[17]:


#barplot
sns.countplot(x='age', data=data)


# In[18]:


#scatterplot
plt.scatter(x='age', y='sex', data=data)


# In[19]:


#pairplot
sns.pairplot(data)


# ## Pandas Profiling for Automated EDA

# In[20]:


pip install pandas-profiling


# In[21]:


import pandas_profiling as pp


# In[22]:


pp.ProfileReport(data)


# ### Preparing the data

# In[23]:


x = data.drop('target',axis=1)
y = data["target"]


# ### Splitting the data.

# In[24]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 0)


# ### Feature scaling

# In[25]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# ### Fitting and transforming the data.

# In[26]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# ## Applying different ML algorithms to find best algorithm with higher prediction

# ### Logist regression.

# In[27]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()


# In[28]:


#fiting the model
model = LR.fit(x_train, y_train)


# In[29]:


#predictions
LR_predict = LR.predict(x_test)


# In[30]:


from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report


# In[31]:


#confusion matrix
LR_conf_matrix = confusion_matrix(y_test, LR_predict)
print("confussion matrix")
print(LR_conf_matrix)


# In[32]:


plt.figure(figsize=(8, 6))
sns.heatmap(LR_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[33]:


LR_acc_score = accuracy_score(y_test, LR_predict)
print("Accuracy of Logistic Regression:",LR_acc_score*100,'\n')


# In[34]:


print(classification_report(y_test,LR_predict))


# ## Gaussian Naive Bayes

# In[35]:


from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()


# In[36]:


#fitting the model
NB.fit(x_train,y_train)


# In[37]:


#predictions
NBpred = NB.predict(x_test)


# In[38]:


#confusion matrix
NB_conf_matrix = confusion_matrix(y_test, NBpred)
print("confussion matrix")
print(NB_conf_matrix)


# In[39]:


plt.figure(figsize=(8, 6))
sns.heatmap(NB_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[40]:


#accuracy score
NB_acc_score = accuracy_score(y_test, NBpred)
print("Accuracy of Naive Bayes model:",NB_acc_score*100,'\n')


# In[41]:


print(classification_report(y_test,NBpred))


# ## Extreme Gradient Boost

# In[42]:


from xgboost import XGBClassifier
xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
                    reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)


# In[43]:


#fitting the data
xgb.fit(x_train, y_train)


# In[44]:


#prediction
xgb_predicted = xgb.predict(x_test)


# In[45]:


#confusion matrix
xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)
print("confussion matrix")
print(xgb_conf_matrix)


# In[46]:


plt.figure(figsize=(8, 6))
sns.heatmap(xgb_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[47]:


#accuracy score
xgb_acc_score = accuracy_score(y_test, xgb_predicted)
print("Accuracy of Extreme Gradient Boost:",xgb_acc_score*100,'\n')


# In[48]:


print(classification_report(y_test,xgb_predicted))


# ## Random Forest

# In[49]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5)


# In[50]:


#fittig the data 
RF.fit(x_train,y_train)


# In[51]:


#prediction
RF_predicted = RF.predict(x_test)


# In[52]:


#confusion matrix
RF_conf_matrix = confusion_matrix(y_test, RF_predicted)
print("confussion matrix")
print(RF_conf_matrix)


# In[53]:


plt.figure(figsize=(8, 6))
sns.heatmap(RF_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[54]:


#accuracy score
RF_acc_score = accuracy_score(y_test, RF_predicted)
print("Accuracy of Random Forest:",RF_acc_score*100,'\n')


# In[55]:


#classification report
print(classification_report(y_test,RF_predicted))


# ## Decision Tree

# In[56]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)


# In[57]:


#fitting the data
DT.fit(x_train, y_train)


# In[58]:


#prediction
DT_predicted = DT.predict(x_test)


# In[59]:


#confusion matrix
DT_conf_matrix = confusion_matrix(y_test, DT_predicted)
print("confussion matrix")
print(DT_conf_matrix)


# In[60]:


plt.figure(figsize=(8, 6))
sns.heatmap(DT_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[61]:


#accuracy score
DT_acc_score = accuracy_score(y_test, DT_predicted)
print("Accuracy of DecisionTreeClassifier:",DT_acc_score*100,'\n')


# In[62]:


print(classification_report(y_test,DT_predicted))


# ## Support Vector Machine

# In[63]:


from sklearn.svm import SVC
svc =  SVC(kernel='rbf', C=2)


# In[64]:


# fitting the data
svc.fit(x_train, y_train)


# In[65]:


#prediction
svc_predicted = svc.predict(x_test)


# In[66]:


#confusion matrix
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
print("confussion matrix")
print(svc_conf_matrix)


# In[67]:


plt.figure(figsize=(8, 6))
sns.heatmap(svc_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[68]:


#accuracy score
svc_acc_score = accuracy_score(y_test, svc_predicted)
print("Accuracy of Support Vector Classifier:",svc_acc_score*100,'\n')


# In[69]:


#classification report
print(classification_report(y_test,svc_predicted))


# ## K-NeighborsClassifier

# In[70]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)


# In[71]:


#fitting the data
knn.fit(x_train, y_train)


# In[72]:


#prediction
knn_predicted = knn.predict(x_test)


# In[73]:


#confusion matrix
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
print("confussion matrix")
print(knn_conf_matrix)


# In[74]:


plt.figure(figsize=(8, 6))
sns.heatmap(knn_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[75]:


#accuracy score
knn_acc_score = accuracy_score(y_test, knn_predicted)
print("Accuracy of K-NeighborsClassifier:",knn_acc_score*100,'\n')


# In[76]:


#classification report
print(classification_report(y_test,knn_predicted))


# ## Identifying the importance of each feature.

# In[77]:


#using xgb.feature_importances_ feature 
colors = ['red', 'green', 'blue', 'black', 'yellow', 'magenta', 'cyan']
important_features = pd.DataFrame({'Features': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'], 'Importance': xgb.feature_importances_})
plt.figure(figsize=(10,4))
plt.title("Importance of each feature ")
plt.xlabel("Importance ")
plt.ylabel("Features")
plt.barh(important_features['Features'],important_features['Importance'],color = colors)
plt.show()


# "ca" appears to be the most important feature, suggesting that it strongly influences the model's predictions.

# Conversely, "chol" has the lowest importance, implying that it contributes less to the model's predictions compared to other features.

# ## ROC Curve

# Finding the false positive rate, true positive rate, the threshold value

# In[78]:


LR_false_positive_rate,LR_true_positive_rate,LR_threshold = roc_curve(y_test, LR_predict)
NB_false_positive_rate,NB_true_positive_rate,NB_threshold = roc_curve(y_test,NBpred)
RF_false_positive_rate,RF_true_positive_rate,RF_threshold = roc_curve(y_test,RF_predicted)                                                             
xgb_false_positive_rate,xgb_true_positive_rate,xgb_threshold = roc_curve(y_test,xgb_predicted)
knn_false_positive_rate,knn_true_positive_rate,knn_threshold = roc_curve(y_test,knn_predicted)
DT_false_positive_rate,DT_true_positive_rate,DT_threshold = roc_curve(y_test,DT_predicted)
svc_false_positive_rate,svc_true_positive_rate,svc_threshold = roc_curve(y_test,svc_predicted)


# In[79]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(LR_false_positive_rate,LR_true_positive_rate,label='Logistic Regression')
plt.plot(NB_false_positive_rate,NB_true_positive_rate,label='Naive Bayes')
plt.plot(RF_false_positive_rate,RF_true_positive_rate,label='Random Forest')
plt.plot(xgb_false_positive_rate,xgb_true_positive_rate,label='Extreme Gradient Boost')
plt.plot(knn_false_positive_rate,knn_true_positive_rate,label='K-Nearest Neighbor')
plt.plot(DT_false_positive_rate,DT_true_positive_rate,label='Desion Tree')
plt.plot(svc_false_positive_rate,svc_true_positive_rate,label='Support Vector Classifier')
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.legend()
plt.show()


# The results suggest that KNN and SVC may be preferred choices for this classification task due to their superior performance compared to other classifiers.

# In[80]:


model_evaluation = pd.DataFrame({'Model': ['Logistic Regression','Naive Bayes','Random Forest','Extreme Gradient Boost',
                    'K-Nearest Neighbour','Decision Tree','Support Vector Machine'], 'Accuracy': [LR_acc_score*100,
                    NB_acc_score*100,RF_acc_score*100,xgb_acc_score*100,knn_acc_score*100,DT_acc_score*100,svc_acc_score*100]})
model_evaluation


# In[81]:


model_evaluation_sorted = model_evaluation.sort_values(by='Accuracy', ascending=False)
model_evaluation_sorted


# KNN and SVM stand out as the top-performing models with the highest accuracy, while Extreme Gradient Boost lags behind with the lowest accuracy. 

# This analysis provides insights into the comparative performance of different models, guiding the selection of the most suitable model for the classification task at hand.

# ### Graphically representing the performance of different models.

# In[82]:


colors = ['red','green','blue','gold','silver','yellow','orange',]
plt.figure(figsize=(12,5))
plt.title("barplot Represent Accuracy of different models")
plt.xlabel("Accuracy %")
plt.ylabel("Algorithms")
plt.bar(model_evaluation['Model'],model_evaluation['Accuracy'],color = colors)
plt.show()


# ## Using ensemble learning method in order to try to enhance the performance and accuracy of the model

# In[83]:


pip install mlxtend


# ### Using the stacking technique.

# In[84]:


from mlxtend.classifier import StackingCVClassifier
SCV=StackingCVClassifier(classifiers=[xgb,knn,svc],meta_classifier= svc,random_state=42)


# In[85]:


#fitting the data
SCV.fit(x_train,y_train)


# In[86]:


#pridiction
SCV_predicted = SCV.predict(x_test)


# In[87]:


SCV_conf_matrix = confusion_matrix(y_test, SCV_predicted)
print("confussion matrix")
print(SCV_conf_matrix)


# In[88]:


#accuracy score
SCV_acc_score = accuracy_score(y_test, SCV_predicted)
print("Accuracy of StackingCVClassifier:",SCV_acc_score*100,'\n')


# In[89]:


#classification report
print(classification_report(y_test,SCV_predicted))


# In[90]:


plt.figure(figsize=(8, 6))
sns.heatmap(SCV_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[91]:


#clssification report
SCV_report = classification_report(y_test, SCV_predicted, output_dict=True)
df_report = pd.DataFrame(SCV_report).transpose()


# In[92]:


plt.figure(figsize=(10, 6))
sns.barplot(x=df_report.index, y=df_report['f1-score'])
plt.title('F1-Score by Class')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.xticks(rotation=45)
plt.show()


# ## Representing the roc curve.

# In[93]:


# Calculate ROC curves for all classifiers
classifiers = {
    'Logistic Regression': (LR_predict, 'blue'),
    'Naive Bayes': (NBpred, 'orange'),
    'Random Forest': (RF_predicted, 'green'),
    'Extreme Gradient Boost': (xgb_predicted, 'red'),
    'K-Nearest Neighbor': (knn_predicted, 'purple'),
    'Decision Tree': (DT_predicted, 'brown'),
    'Support Vector Classifier': (svc_predicted, 'cyan'),
    'StackingCVClassifier': (SCV_predicted, 'magenta')
}


# In[94]:


plt.figure(figsize=(10, 5))
plt.title('Receiver Operating Characteristic Curve')
for clf_name, (y_pred, color) in classifiers.items():
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_pred)
    plt.plot(false_positive_rate, true_positive_rate, label=clf_name, color=color)

plt.plot([0, 1], ls='--', color='gray')
plt.plot([0, 0], [1, 0], c='.5')
plt.plot([1, 1], c='.5')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()


# ## visualizing the accuracy of different models.

# In[96]:


# Define the colors for the bar plot
colors = ['red', 'green', 'blue', 'gold', 'silver', 'yellow', 'orange', 'magenta']

# Add the ensemble method result to the model_evaluation DataFrame
model_evaluation.loc[len(model_evaluation)] = ['StackingCVClassifier', SCV_acc_score * 100]

# Plot the bar plot
plt.figure(figsize=(12, 5))
plt.title("Barplot Representing Accuracy of Different Models")
plt.xlabel("Accuracy %")
plt.ylabel("Algorithms")
plt.bar(model_evaluation['Model'], model_evaluation['Accuracy'], color=colors)
plt.xticks(rotation=45)
plt.show()


# In[98]:


model_evaluation = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'Extreme Gradient Boost',
              'K-Nearest Neighbour', 'Decision Tree', 'Support Vector Machine', 'StackingCVClassifier'],
    'Accuracy': [LR_acc_score * 100, NB_acc_score * 100, RF_acc_score * 100, xgb_acc_score * 100,
                 knn_acc_score * 100, DT_acc_score * 100, svc_acc_score * 100, SCV_acc_score * 100]
})

# Display the model evaluation DataFrame
model_evaluation


# In[99]:


#soretd
model_evaluation_sorted = model_evaluation.sort_values(by='Accuracy', ascending=False)
model_evaluation_sorted 


# ### Based on the accuracy scores alone, K-Nearest Neighbour and Support Vector Machine appear to be the top-performing models

# In[ ]:





# In[ ]:




