#!/usr/bin/env python
# coding: utf-8

# # Directing Customers To Subscription Through Financial App Behaviour Analysis

# ## Goal of Project

# The "FinTech" company launch there android and IOS mobile base app and want to grow there business.But there is problem how to 
# recommended this app and offer who really want to use it.So for that company decided to give free trial each and every customer
# for 24 hours and collect data from the customers.In this scenario some customer purchase the app and someone not.According to 
# this data company want to give special offer to the customer who are not interested to buy without offer and grow the business
# 
# This is an classification type problem

# ##  Importing essential libraries

# In[5]:


import numpy as np # for numeric calculation
import pandas as pd # for data analysis and manupulation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization
from dateutil import parser # convert time in date time data type


# ## Import Dataset & Explore

# In[6]:


finTech_appData=pd.read_csv("D:\mlai project\FineTech_appData.csv")


# In[7]:


finTech_appData.shape


# In[8]:


finTech_appData.head(6) # Show First 6 rows of finTech_appData DataFrame  


# In[9]:


finTech_appData.tail(6) # Show Last 6 rows of finTech_appData


# In[10]:


for i in [1,2,3,4,5]:
    print(finTech_appData.loc[i,'screen_list'],'\n')


# ## Know about Dataset
As you can see in fineTech_appData DataFrame, there are 50,000 users data with 12 different features.Description each and every feature in brief.

1. user: Unique ID for each user.

2. first_open: Date (yy-mm-dd) and time (Hour:Minute:Seconds:Milliseconds) of login on app first time.

3. dayofweek: On which day user login.

0: Sunday
1: Monday
2: Tuesday
3: Wednesday
4: Thursday
5: Friday
6: Saturday
4. Hour: Time of a day in 24-hour format customer logon. It is correlated with dayofweek column.

5. age: The age of the registered user.

6. screen_list: The name of multiple screens seen by customers, which are separated by a comma.

7. numscreens: The total number of screens seen by customers.

8. minigame: Tha app contains small games related to finance. If the customer played mini-game then 1 otherwise 0.

9. used_premium_feature: If the customer used the premium feature of the app then 1 otherwise 0.

10. enrolled: If the user bought a premium feature app then 1 otherwise 0.

11. enrolled_date: On the date (yy-mm-dd) and time (Hour:Minute:Seconds:Milliseconds) the user bought a premium features app.

12. liked: The each screen of the app has a like button if the customer likes it then 1 otherwise 0.
# ### Find the null value in DataFrame using DataFrame.isnull() method and take summation by sum() method.

# In[11]:


finTech_appData.isnull().sum() # taking summation of null values


# ### Brief inforamtion about Dataset

# In[12]:


finTech_appData.info() 


# ### Getting the distribution of numerical variables

# In[13]:



finTech_appData.describe() 


# ### Getting the unique value of each columns and it's length

# In[14]:


features = finTech_appData.columns
for i in features:
    print("""Unique value of {}\n{}\nlen is {} \n........................\n
          """.format(i, finTech_appData[i].unique(), len(finTech_appData[i].unique())))


# ### Converting hour data convert string to int

# In[15]:


finTech_appData['hour'] = finTech_appData.hour.str.slice(1,3).astype(int) 


# ### Getting data type of each columns
# 

# In[16]:


finTech_appData.dtypes


# ### Dropping object dtype columns

# In[17]:


finTech_appData2 = finTech_appData.drop(['user', 'first_open', 'screen_list', 'enrolled_date'], axis = 1)


# ### Head of numeric dataFrame

# In[18]:


finTech_appData2.head(6)


# ## Data Visualisation

# ### Heat Map using Correlation Matrix

# In[19]:


plt.figure(figsize=(16,9)) # heatmap size in ratio 16:9
 
sns.heatmap(finTech_appData2.corr(), annot = True, cmap ='coolwarm') # show heatmap
 
plt.title("Heatmap using correlation matrix of finTech_appData2", fontsize = 25) # title of heatmap


# ### Pair plot of fineTech_appData

# In[20]:


sns.pairplot(finTech_appData2, hue  = 'enrolled')
plt.show()


# ### Showing counterplot of 'enrolled' feature

# In[21]:


sns.countplot(finTech_appData.enrolled)


# ### Value enrolled and not enrolled customers

# In[22]:


print("Not enrolled user = ", (finTech_appData.enrolled < 1).sum(), "out of 50000")
print("Enrolled user = ",50000-(finTech_appData.enrolled < 1).sum(),  "out of 50000")


# ### Plotting Histogram

# In[23]:


plt.figure(figsize = (16,9)) # figure size in ratio 16:9
features = finTech_appData2.columns # list of columns name
for i,j in enumerate(features): 
    plt.subplot(3,3,i+1) # create subplot for histogram
    plt.title("Histogram of {}".format(j), fontsize = 15) # title of histogram
     
    bins = len(finTech_appData2[j].unique()) # bins for histogram
    plt.hist(finTech_appData2[j], bins = bins, rwidth = 0.8, edgecolor = "y", linewidth = 2, ) # plot histogram
     
plt.subplots_adjust(hspace=0.5) # space between horixontal axes (subplots)


# ### Showing Correlation barplot with ‘enrolled’ feature

# In[24]:


sns.set() # set background dark grid
plt.figure(figsize = (14,5))
plt.title("Correlation all features with 'enrolled' ", fontsize = 20)
finTech_appData3 = finTech_appData2.drop(['enrolled'], axis = 1) # drop 'enrolled' feature
ax =sns.barplot(finTech_appData3.columns,finTech_appData3.corrwith(finTech_appData2.enrolled)) # plot barplot 
ax.tick_params(labelsize=15, labelrotation = 20, color ="k") # decorate x & y ticks font


# ### Parsing ‘first_open’ and ‘enrolled_date’ object data in data and time format.

# In[25]:


finTech_appData['first_open'] =[parser.parse(i) for i in finTech_appData['first_open']]
 
finTech_appData['enrolled_date'] =[parser.parse(i) if isinstance(i, str) else i for i in finTech_appData['enrolled_date']]
 
finTech_appData.dtypes


# ### Finding how much time the customer takes to get enrolled in the premium feature app after registration

# In[26]:


finTech_appData['time_to_enrolled']  = (finTech_appData.enrolled_date - finTech_appData.first_open).astype('timedelta64[h]')


# In[27]:


# Plot histogram
plt.hist(finTech_appData['time_to_enrolled'].dropna())


# ### Distribution in range 0 to 100 hours.

# In[28]:


# Plot histogram
plt.hist(finTech_appData['time_to_enrolled'].dropna(), range = (0,100)) 


# ## Feature Selection

# ### Those customers have enrolled after 48 hours setting them as 0

# In[29]:


finTech_appData.loc[finTech_appData.time_to_enrolled > 48, 'enrolled'] = 0
finTech_appData.drop(columns = ['time_to_enrolled', 'enrolled_date', 'first_open'], inplace=True)


# ### Reading top screen csv file and convert it into numpy array

# In[30]:



finTech_app_screen_Data = pd.read_csv("D:/mlai project/top_screens.csv").top_screens.values
 
finTech_app_screen_Data


# In[31]:


finTech_appData['screen_list'] = finTech_appData.screen_list.astype(str) + ','


# ### String to number and getting shape

# In[32]:


for screen_name in finTech_app_screen_Data:
    finTech_appData[screen_name] = finTech_appData.screen_list.str.contains(screen_name).astype(int)
    finTech_appData['screen_list'] = finTech_appData.screen_list.str.replace(screen_name+",", "")

finTech_appData.shape


# In[33]:


# head of DataFrame
finTech_appData.head(6)


# 

# In[34]:


# remain screen in 'screen_List'
finTech_appData.loc[0,'screen_list']


# In[35]:


# count remain screen list and store counted number in 'remain_screen_list'
finTech_appData['remain_screen_list'] = finTech_appData.screen_list.str.count(",")


# In[36]:


# Drop the 'screen_list'
finTech_appData.drop(columns = ['screen_list'], inplace=True)


# In[37]:


# total columns
finTech_appData.columns


# ### taking sum of all saving screen in one place

# In[38]:




saving_screens = ['Saving1',
                  'Saving2',
                  'Saving2Amount',
                  'Saving4',
                  'Saving5',
                  'Saving6',
                  'Saving7',
                  'Saving8',
                  'Saving9',
                  'Saving10',
                 ]
finTech_appData['saving_screens_count'] = finTech_appData[saving_screens].sum(axis = 1)
finTech_appData.drop(columns = saving_screens, inplace = True)


# ### Similarly for credit, CC1 and loan screens.

# In[39]:


credit_screens = ['Credit1',
                  'Credit2',
                  'Credit3',
                  'Credit3Container',
                  'Credit3Dashboard',
                 ]
finTech_appData['credit_screens_count'] = finTech_appData[credit_screens].sum(axis = 1)
finTech_appData.drop(columns = credit_screens, axis = 1, inplace = True)


# In[40]:


cc_screens = ['CC1',
              'CC1Category',
              'CC3',
             ]
finTech_appData['cc_screens_count'] = finTech_appData[cc_screens].sum(axis = 1)
finTech_appData.drop(columns = cc_screens, inplace = True)


# In[41]:


loan_screens = ['Loan',
                'Loan2',
                'Loan3',
                'Loan4',
               ]
finTech_appData['loan_screens_count'] = finTech_appData[loan_screens].sum(axis = 1)
finTech_appData.drop(columns = loan_screens, inplace = True)


# ### Now shape of Data Frame is reduce

# In[42]:


finTech_appData.shape


# In[43]:


finTech_appData.info()


# In[44]:


# Numerical distribution of fineTech_appData
finTech_appData.describe()


# ### Heatmap with correlation Matrix of  new finTech_appData

# In[45]:


plt.figure(figsize = (25,16)) 
sns.heatmap(finTech_appData.corr(), annot = True, linewidth =2)


# ## Data Preprocessing

# ### Splitting Data Set in Train and Test

# In[46]:


clean_finTech_appData = finTech_appData
target = finTech_appData['enrolled']
finTech_appData.drop(columns = 'enrolled', inplace = True)
1
2
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(finTech_appData, target, test_size = 0.2, random_state = 0)


# In[47]:


print('Shape of X_train = ', X_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of y_test = ', y_test.shape)


# ### Taking User Id in another Variable

# In[48]:


train_userID = X_train['user']
X_train.drop(columns= 'user', inplace =True)
test_userID = X_test['user']
X_test.drop(columns= 'user', inplace =True)


# In[49]:


print('Shape of X_train = ', X_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of train_userID = ', train_userID.shape)
print('Shape of test_userID = ', test_userID.shape)


# ## Feature Scaling

# ### The multiple features in the different units so for the best accuracy need to convert all features in a single unit.

# In[50]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# ## Machine Learning Model Building

# ### The target variable is categorical type 0 and 1, so I will use supervised classification algorithms

# ### importing required packages

# In[51]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ### Decision Tree Classifier

# In[52]:


from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_score(y_test, y_pred_dt)


# ### Train with Standard Scaling dataset

# In[53]:


dt_model2 = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
dt_model2.fit(X_train_sc, y_train)
y_pred_dt_sc = dt_model2.predict(X_test_sc)
accuracy_score(y_test, y_pred_dt_sc)


# ### K – Nearest Neighbor Classifier

# In[54]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2,)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
 
accuracy_score(y_test, y_pred_knn)


# ### Train with Standard Scaling dataset

# In[55]:


knn_model2 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2,)
knn_model2.fit(X_train_sc, y_train)
y_pred_knn_sc = knn_model2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_knn_sc)


# ### Naive Bayes Classifier

# In[56]:


from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
 
accuracy_score(y_test, y_pred_nb)


# ### Train with Standard Scaling dataset

# In[57]:


nb_model2 = GaussianNB()
nb_model2.fit(X_train_sc, y_train)
y_pred_nb_sc = nb_model2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_nb_sc)


# ### Random Forest Classifier

# In[58]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
 
accuracy_score(y_test, y_pred_rf)


# ### Train with Standard Scaling dataset

# In[59]:



rf_model2 = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_model2.fit(X_train_sc, y_train)
y_pred_rf_sc = rf_model2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_rf_sc)


# ### Logistic Regression
# 

# In[60]:


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state = 0, penalty = 'l1',solver='liblinear')
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
 
accuracy_score(y_test, y_pred_lr)


# ### Train with Standard Scaling dataset

# In[61]:



lr_model2 = LogisticRegression(random_state = 0, penalty = 'l1',solver='liblinear')
lr_model2.fit(X_train_sc, y_train)
y_pred_lr_sc = lr_model2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_lr_sc)


# ### Support Vector Classifier

# In[62]:


from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)
 
accuracy_score(y_test, y_pred_svc)


# ### Train with Standard Scaling dataset

# In[63]:


svc_model2 = SVC()
svc_model2.fit(X_train_sc, y_train)
y_pred_svc_sc = svc_model2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_svc_sc)


# In[64]:


import xgboost as xgb


# In[65]:


import xgboost as xgb


# ### XGBoost Classifier

# In[66]:


from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracy_score(y_test, y_pred_xgb)


# ### Train with Standard Scaling dataset

# In[67]:


xgb_model2 = XGBClassifier()
xgb_model2.fit(X_train_sc, y_train)
y_pred_xgb_sc = xgb_model2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_xgb_sc)


# ### XGB classifier with parameter tuning

# In[68]:


xgb_model_pt1 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
 
xgb_model_pt1.fit(X_train, y_train)
y_pred_xgb_pt1 = xgb_model_pt1.predict(X_test)
 
accuracy_score(y_test, y_pred_xgb_pt1)


# ### XGB classifier with parameter tuning + Train with Standard Scaling dataset

# In[69]:


xgb_model_pt2 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
 
xgb_model_pt2.fit(X_train_sc, y_train)
y_pred_xgb_sc_pt2 = xgb_model_pt2.predict(X_test_sc)
 
accuracy_score(y_test, y_pred_xgb_sc_pt2)


# ### We observe that Support Vector Classifier(SVC) and XGBoost Classifier give best accuracy than ohter ML algorithm.
# ### But we will continue with XGBoost classifier because the accuracy is slightly higher than SVC.

# ### Confusion Matrix

# In[70]:


cm_xgb_pt2 = confusion_matrix(y_test, y_pred_xgb_sc_pt2)
sns.heatmap(cm_xgb_pt2, annot = True, fmt = 'g')
plt.title("Confussion Matrix", fontsize = 20) 


# ### Classification report of ML model

# In[71]:


cr_xgb_pt2 = classification_report(y_test, y_pred_xgb_sc_pt2)
 
print("Classification report >>> \n", cr_xgb_pt2)


# ### Cross Validation of the ML Model

# In[73]:



from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_model_pt2, X = X_train_sc, y = y_train, cv = 10)
print("Cross validation of XGBoost model = ",cross_validation)
print("Cross validation of XGBoost model (in mean) = ",cross_validation.mean())


# ### The mean value cross-validation and XGBoost model accuracy is 78%. That means our XGBoost model is a generalized model.

# ### Mapping predicted output to the target

# In[75]:


final_result = pd.concat([test_userID, y_test], axis = 1)
final_result['predicted result'] = y_pred_xgb_sc_pt2
 
print(final_result)


# ### This is my Machine Learning Project  with 78.87% accuracy which is good for 'Directing Customers to Subscription Through Financial App Behavior Analysis' 

# ### To get more accuracy, we train all supervised classification algorithms but I have  try out a few of them which are always popular. After training all algorithms, I found that SVC and XGBoost classifiers are given high accuracy than remaining but we have chosen XGBoost.
