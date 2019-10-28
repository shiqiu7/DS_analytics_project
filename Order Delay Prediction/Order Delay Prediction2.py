
# coding: utf-8

# Step1: define the problem
# 1. The business objective: identify factors and 2. build a model to predict if order will be fulfilled on time or delay.
# 2. The problem is supurvised learning, and a classfication problem
# 3. performance evaluation: ROC/AUC,recall/precision

# In[1]:


#import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#import data
df=pd.read_csv('Consolidated data.csv')
print('Dataframe dimensions:', df.shape)


# Step2: Data Cleaning and manipulation before data exploration

# In[3]:


#subset only delivered and know labels' data
df_d=df[(df['SHP_PACK_DLVRY_STS_KEY']==6)&(df['SPDPD_KEY']!=1)&(df['SPDPD_KEY']!=10)]
df_d=df_d.dropna(subset=['SPDPD_KEY'])
df_d['ontime_ornot']=df_d['SPDPD_KEY']
df_d['ontime_ornot']=df_d['ontime_ornot'].replace([2,4],1).replace([3,5,6,7,8,9],0)


# In[4]:


#ratio of delay:
1-sum(df_d['ontime_ornot'])/len(df_d)


# In[6]:


#drop clearly not relevant columns
df_d=df_d.drop(['CREATED_TS','CREATED_USER','UPDATED_TS','UPDATED_USER','FRAUD_CHECK_REQD_IND','CORP_ORDER_IND','CONSUMER_CNTCT_ID','TENANT_ASSOC_ID'],axis=1)


# In[7]:


#unify null
df_d=df_d.replace('?',np.nan)
df_d=df_d.replace('NULL',np.nan)


# In[8]:


#clean time stamp
for i in (5,6,7,9,10,12,15,16,17):
    for j in range(len(df_d)):
        if type(df_d.iloc[j,i]) is str:
            df_d.iloc[j,i]=df_d.iloc[j,i][:-13]


# In[9]:


#clean SHIPPED_PACKAGE_PRICE_AMT
df_d['SHIPPED_PACKAGE_PRICE_AMT']=df_d['SHIPPED_PACKAGE_PRICE_AMT'].str.replace(",","").astype(float)
df_d['SHIPPED_PACKAGE_PRICE_AMT']=df_d['SHIPPED_PACKAGE_PRICE_AMT'].astype(float)


# In[10]:


#change time stamp date type

for i in ('ORDER_TS','EXPECTED_DELIVERY_TS','LEAST_DELIVERY_TS','DETERMINED_DELIVERY_TS','EXPECTED_SHIPMENT_TS','CARRIER_EXPECTED_DELIVERY_TS','CARGO_DEPARTED_TS','INITIAL_DLVR_ATMPT_TS','FINAL_DELIVERY_ATMPT_TS','ACTUAL_DELIVERY_TS','ORDER_REL_TS','ORDER_PLACED_TS'):
    df_d[i]=pd.to_datetime(df_d[i])
    


# In[11]:


#change index
df_d.index=range(len(df_d))


# In[12]:


#create test data: stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit
sp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
for train_index, test_index in sp.split(df_d, df_d['ontime_ornot']):
        df_train = df_d.loc[train_index]
        df_test = df_d.loc[test_index]
print(df_train.shape)
print(df_test.shape)
combine=[df_train,df_test]


# Step3: Data exploration
# (more on the report)

# In[13]:


# explore analysis
#1. distributor feature:
a=1-df_train['ontime_ornot'].groupby(df_train['DSTRBTR_KEY']).apply(np.mean)
print(a)
sns.barplot(a.index,a.values)
#conclusion: distributor is a factor


# In[14]:


#2. feature engineering: order month (half month)
combine=[df_train,df_test]
for data in combine:
    data['order_month']=''
    for j in range(len(data)):
        if data.iloc[j,5].day in range(15):
            a=0
        else:
            a=0.5
        data.iloc[j,51]=data.iloc[j,5].month+a
a=1-df_train['ontime_ornot'].groupby(df_train['order_month']).apply(np.mean)
sns.barplot(a.index,a.values)
a
#conclusion: seasonality is an important feature


# In[15]:


#3. SHIPPED_PACKAGE_PRICE_AMT : more on tableau
df_train[['SHIPPED_PACKAGE_PRICE_AMT','TOT_AMT']].groupby(df_train['ontime_ornot']).apply(np.mean)



# In[16]:


#4. feature engineering use timestamp-----expected delivery time
df_train['expected delivery time']=df_train['EXPECTED_DELIVERY_TS']-df_train['ORDER_TS']
df_test['expected delivery time']=df_test['EXPECTED_DELIVERY_TS']-df_test['ORDER_TS']
print("Expected delivery time for on time")
print(df_train[df_train['ontime_ornot']==1]['expected delivery time'].describe())
print('------------')
print("Expected delivery time for delays")
print(df_train[df_train['ontime_ornot']==0]['expected delivery time'].describe())


# In[17]:


#5. states----on tableau
#rename state column

df_train=df_train.rename(columns={'CITY_STATE_INFO_Vcol':'state'})  
df_test=df_test.rename(columns={'CITY_STATE_INFO_Vcol':'state'}) 


# In[18]:


#determined-least
df_train['delivery time range']=df_train['DETERMINED_DELIVERY_TS']-df_train['LEAST_DELIVERY_TS']
df_test['delivery time range']=df_test['DETERMINED_DELIVERY_TS']-df_train['LEAST_DELIVERY_TS']
print("Expected delivery range for on time")
print(df_train[df_train['ontime_ornot']==1]['delivery time range'].describe())
print('------------')
print("Expected delivery range for delay")
print(df_train[df_train['ontime_ornot']==0]['delivery time range'].describe())
#analysis: this is an effective factor, but need to check correlation
#correlation between quantitative attributes
df_train[(df_train['delivery time range'].notnull())&(df_train['TOT_AMT'].notnull())][['delivery time range','expected delivery time','ontime_ornot','SHIPPED_PACKAGE_PRICE_AMT','TOT_AMT']].astype(int).corr().sort_values(by='ontime_ornot',ascending=False)


# Step4: data manipulation after data exploration
#     1. one-hot encoding for categorical features
#     2. feature scaling for quantitative features
#     3. complete null values

# In[19]:


#extract factors
df_train_model=df_train[['DSTRBTR_KEY','state','order_month','SHIPPED_PACKAGE_PRICE_AMT','expected delivery time','delivery time range','TOT_AMT','ontime_ornot']]
df_test_model=df_test[['DSTRBTR_KEY','state','order_month','SHIPPED_PACKAGE_PRICE_AMT','expected delivery time','delivery time range','TOT_AMT','ontime_ornot']]


# In[20]:


df_train_model.info()


# In[21]:


#one-hot encoding for distributor, state and month
df_train_model=pd.get_dummies(df_train_model, columns=['DSTRBTR_KEY','order_month','state'])
df_test_model=pd.get_dummies(df_test_model, columns=['DSTRBTR_KEY','order_month','state'])


# In[22]:


df_train_model['TOT_AMT']=df_train_model['TOT_AMT'].fillna(df_train_model['TOT_AMT'].median())
df_test_model['TOT_AMT']=df_test_model['TOT_AMT'].fillna(df_test_model['TOT_AMT'].median())


# In[23]:


#scalling for SHIPPED_PACKAGE_PRICE_AMT,OT_AMT, and expected delivery time
df_train_model['expected delivery time']=df_train_model['expected delivery time'].astype(int)
df_test_model['expected delivery time']=df_test_model['expected delivery time'].astype(int)
df_train_model['delivery time range']=df_train_model['delivery time range'].astype(int)
df_test_model['delivery time range']=df_test_model['delivery time range'].astype(int)
from sklearn.preprocessing import StandardScaler
df_train_model.iloc[:,0:4]= StandardScaler().fit_transform(df_train_model.iloc[:,0:4])
df_test_model.iloc[:,0:4]= StandardScaler().fit_transform(df_test_model.iloc[:,0:4])


# In[24]:


#remove states with too few observations
a=pd.Series(df_train.index).groupby(df_train['state']).agg('count')
print(a[a<10])
df_train_model=df_train_model.drop(['state_AE','state_AK','state_HI','state_PR','state_VI','state_WY'],axis=1)
df_test_model=df_test_model.drop(['state_AK','state_HI','state_VI','state_WY'],axis=1)
#add IA so that train and test data can be same
df_test_model['state_IA']=0
df_test_model['state_DC']=0
df_test_model=df_test_model[list(df_train_model.columns)]



# In[111]:


df_train_model=df_train_model.drop('Unnamed: 0',axis=1)
df_test_model=df_test_model.drop('Unnamed: 0',axis=1)


# In[25]:


df_train_model.to_csv('df_train_model.csv')
df_test_model.to_csv('df_test_model.csv')


# In[26]:


df_train_model=pd.read_csv('df_train_model.csv')
df_test_model=pd.read_csv('df_test_model.csv')


# Step 5:  find the best model
# 1. Models used: Random Forest, Xgboost,KNN, Logistic regression
# 2. Evalution metrics: I use AUC/ROC because it is animblance classification problem
# 3. Avoid overfitting techniques: K-fold cross validation, tuning hyperparameters
# 
# 

# In[132]:


#split into X and y
X=df_train_model.drop('ontime_ornot',axis=1)
y=df_train_model['ontime_ornot']


# In[134]:


#tune randanm forest
#hyperparameter: number of trees, max depth, and max features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
model = ensemble.RandomForestClassifier()
n_estimators = range(50, 500, 50)
max_depth=range(3,20,1)
max_features=('sqrt','log2')
param_grid = dict(n_estimators=n_estimators,max_features =max_features,max_depth=max_depth)
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[135]:


#tune xgboost
#tune number of trees, max depth
import xgboost as xgb
model = xgb.XGBClassifier()
n_estimators = range(50, 500, 50)
max_depth=range(3,10,1)
param_grid = dict(n_estimators=n_estimators,max_depth=max_depth)
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[136]:


#tune knn
#tune number of neigbors
from sklearn.neighbors import KNeighborsClassifier
model =KNeighborsClassifier()
n_neighbors = range(5, 20, 1)
param_grid = dict(n_neighbors=n_neighbors)
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[120]:


#tune logistic regression
#tune l1 or l2 regularization
from sklearn.linear_model import LogisticRegression
model =LogisticRegression()
penalty=('l1','l2')
param_grid = dict(penalty=penalty)
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[137]:


#based on the cross validation result, the best model is as follow:
model=ensemble.RandomForestClassifier(max_depth=19,max_features='log2',n_estimators=450)


# Precision-recall trade off: find the best threshold
# 1. In this problem, we want to capture the delays as much as possible. False positives are less important than false negatives. Since recall is more important than precision, we use an adjusted F1 score as metric, which gives more weight to recall, to find the best threshold (no accurate meaning, need more business data to have more specific metrics)

# In[138]:


#precision recall trade off:
# 
from sklearn.metrics import confusion_matrix

Performance=[]
for thr in np.arange(0.1,0.5,0.05):
    re=[]
    pr=[]
    kf = StratifiedKFold(n_splits=4, random_state=42)
    for train_index, test_index in kf.split(X.values,y.values):
        X_train_folds = X.values[train_index]
        y_train_folds = (y.values[train_index])
        X_test_fold = X.values[test_index]
        y_test_fold = (y.values[test_index])
        model.fit(X_train_folds, y_train_folds)
        y_predict=pd.DataFrame(model.predict_proba(X_test_fold))
        b=y_predict.iloc[:,0]
        y_predict['prediction']=np.where(b>=thr,0,1)
        y_predict=y_predict['prediction'].values
        m=confusion_matrix(y_test_fold,y_predict)
        re.append(m[0,0]/(m[0,0]+m[0,1]))
        pr.append(m[0,0]/(m[0,0]+m[1,0]))
    PR=sum(pr)/len(pr)
    RE=sum(re)/len(re)
    performance=['Random Forest',thr,RE,PR,2/(2/RE+1/PR)]
    Performance.append(performance)
Performance=pd.DataFrame(Performance)
Performance.columns=['model','threshold','recall','precision','metrics_defined']
Performance.sort_values(by='metrics_defined',ascending=False)

#Based on the metric, the best threshold is 0.15


# In[139]:


#test on test data now

X_test=df_test_model.drop('ontime_ornot',axis=1)
y_test=df_test_model['ontime_ornot']
model.fit(X,y)
y_predict=pd.DataFrame(model.predict_proba(X_test))
b=y_predict.iloc[:,0]
y_predict['prediction']=np.where(b>=0.15,0,1)
m=confusion_matrix(y_test,y_predict['prediction'])
re=m[0,0]/(m[0,0]+m[0,1])
pr=m[0,0]/(m[0,0]+m[1,0])
ac=(m[0,0]+m[1,1])/(m[0,0]+m[1,1]+m[0,1]+m[1,0])

print('test result: accuracy','{:.3%}'.format(ac),'recall','{:.3%}'.format(re),'precision','{:.3%}'.format(pr))


