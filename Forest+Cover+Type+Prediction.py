
# coding: utf-8

# ### It is a competition offered by Kaggle site wherein this competition you are asked to predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables (as opposed to remotely sensed data)

# ### Importing Libraries

# In[1]:

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# ### Reading Datasets

# In[6]:

train= pd.read_csv('train.csv')
test= pd.read_csv('test.csv')


# In[7]:

full_data=[train,test]


# In[8]:

train_copy=train.copy()
test_copy=test.copy()


# In[9]:

train.head()


# ## EDA and Feature Engineering

# In[10]:

del train['Id']


# In[11]:

subm=test['Id']
del test["Id"]


# In[12]:

train.ix[:,:10].hist(bins=70,figsize=(16,10))
plt.show()


# In[13]:

def angle(x):
    if x>180:
        return 180-x
    else:
        return x


# In[14]:

train.columns


# In[15]:

for data in full_data:
    data['Aspect']=data['Aspect'].apply(angle)


# In[16]:

data.Aspect.describe()


# Imputing missing values of Hillshade_3pm with GBRT

# In[17]:

from sklearn.ensemble import GradientBoostingRegressor
impute=GradientBoostingRegressor(n_estimators=1000)


# In[18]:

temp_train=train.copy()
cols=train.columns.tolist()
cols=cols[:8]+cols[9:]+[cols[8]]
temp_train=temp_train[cols]


# In[19]:

temp_test=test.copy()
cols=test.columns.tolist()
cols=cols[:8]+cols[9:]+[cols[8]]
temp_test=temp_test[cols]


# In[20]:

rxtrain,rytrain= temp_train[temp_train['Hillshade_3pm']!=0].ix[:,:54].values,temp_train[temp_train['Hillshade_3pm']!=0].ix[:,54].values


# In[21]:

impute.fit(rxtrain,rytrain)


# In[22]:

values_train=impute.predict(temp_train[temp_train['Hillshade_3pm']==0].ix[:,:54].values)


# In[23]:

values_test=impute.predict(temp_test[temp_test['Hillshade_3pm']==0].ix[:,:54].values)


# In[24]:

train.loc[train['Hillshade_3pm']==0,'Hillshade_3pm']=values_train


# In[25]:

test.loc[test['Hillshade_3pm']==0,'Hillshade_3pm']=values_test


# In[27]:

plt.scatter(train.Hillshade_3pm,train.Hillshade_Noon)


# Creating new features

# In[28]:

for data in full_data:
    data['dist_to_hydrology']=np.sqrt(data['Vertical_Distance_To_Hydrology']**2 +                                       data['Horizontal_Distance_To_Hydrology']**2)


# In[29]:

for data in full_data:
    data['Highwater']=(data['Vertical_Distance_To_Hydrology']<0)*1


# In[30]:

for data in full_data:
    data['Slope_to_hydrology']=data['Vertical_Distance_To_Hydrology']/ data['Horizontal_Distance_To_Hydrology']


# In[31]:

for data in full_data:
    data.loc[data['Horizontal_Distance_To_Hydrology']==0,'Slope_to_hydrology']=0


# In[32]:

with sns.axes_style('white'):
    matrix=train[['Elevation','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',                  'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']]
    scatter_matrix(matrix,figsize=(20,20),diagonal='kde')
plt.show()    


# Creating more new features

# In[33]:

for data in full_data:
    data['Ele_to_HD']=data.Elevation-0.2*data.Horizontal_Distance_To_Hydrology
    data['Elev_to_VD']=data.Elevation-data.Vertical_Distance_To_Hydrology
    data['Elev_to_HDR']=data.Elevation-0.05*data.Horizontal_Distance_To_Roadways


# In[34]:

for data in full_data:
    data['amenities']=(data.Horizontal_Distance_To_Hydrology+data.Horizontal_Distance_To_Roadways                        +data.Horizontal_Distance_To_Fire_Points)/3


# In[35]:

print train.shape
print test.shape


# ## Modelling

# In[36]:

features=[x for x in train.columns if x not in ['Cover_Type']]


# In[37]:

target=['Cover_Type']


# In[38]:

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(train[features].values,train[target].values,test_size=0.15)
print xtest.shape
print ytest.shape


# In[39]:

from sklearn.ensemble import RandomForestClassifier
clf_1=RandomForestClassifier(n_estimators=1000,random_state=2)


# In[40]:

clf_1.fit(xtrain,ytrain)


# In[41]:

from sklearn.metrics import accuracy_score
print accuracy_score(ytrain,clf_1.predict(xtrain))
print accuracy_score(ytest,clf_1.predict(xtest))


# In[42]:

importances=clf_1.feature_importances_
importances.astype(float)
indices=np.argsort(importances)[::-1]


# In[51]:

features1=np.array(features)


# In[52]:

(pd.Series(importances[indices],index=(features1[indices])))


# In[44]:

clf_1.fit(train[features].values,np.ravel(train[target].values))


# In[ ]:

predict=clf_1.predict(test[features].values)


# In[ ]:

subm=pd.DataFrame(test['Id'])
subm['Cover_Type']=predict
subm.to_csv('Submission.csv',sep=',')



# ### This submission scored 0.74273 on Kaggle LeaderBoard

# In[ ]:



