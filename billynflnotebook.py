#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import boto3
import re
import copy
import time
from time import gmtime, strftime
from sagemaker import get_execution_role

role = get_execution_role()

region = boto3.Session().region_name

bucket='sagemakernflstats1' # Replace with your s3 bucket name
prefix = 'sagemaker/xgboost-mnist' # Used as part of the path in the bucket where you store data
bucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region,bucket) # The URL to access the bucket


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


pass_df = pd.read_csv("https://sagemakernflstats1.s3.amazonaws.com/nflstatistics/Career_Stats_Passing.csv")


# In[4]:


pass_df.head()


# In[5]:


pass_df['Position'].isnull().sum()


# In[6]:


pass_df.isnull().sum()


# In[7]:


df = pass_df[pass_df['Games Played'] != 0]
df.head()


# In[8]:


column_names = pass_df.columns
print(column_names)


# In[9]:


df = df[['Player Id', 'Name', 'Games Played', 'Completion Percentage',
       'Pass Attempts Per Game', 'Passing Yards Per Attempt',
       'Passing Yards Per Game', 'Percentage of TDs per Attempts',
         'Int Rate','Passer Rating']]


# In[10]:


df.head()


# In[11]:


df.head()


# In[12]:


ext = df['Player Id'].str.extract('(\d+)').astype(int)
ext.head()


# In[13]:


df['Player Id'] = ext


# In[14]:


df.head()


# In[15]:


df = df[df.ne('--').all(1)]
df.head()


# In[16]:


df.isnull().sum()


# In[17]:


print (df.dtypes)


# In[18]:


df['Completion Percentage'] = df['Completion Percentage'].astype(str).astype('float32')
df['Passing Yards Per Attempt'] = df['Passing Yards Per Attempt'].astype(str).astype('float32')
df['Passing Yards Per Game'] = df['Passing Yards Per Game'].astype(str).astype('float32')
df['Percentage of TDs per Attempts'] = df['Percentage of TDs per Attempts'].astype(str).astype('float32')
df['Int Rate'] = df['Int Rate'].astype(str).astype('float32')


# In[19]:


print (df.dtypes)


# In[20]:


df = df.groupby(['Player Id', 'Name']).mean().reset_index()


# In[21]:


df.head()


# In[22]:


#no need to split data into training and testing because this will be using an unsupervised training


# In[23]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

mns = MinMaxScaler()
df[['Games Played', 'Completion Percentage', 'Pass Attempts Per Game', 'Passing Yards Per Attempt','Passing Yards Per Game', 'Percentage of TDs per Attempts', 'Int Rate', 'Passer Rating']] = mns.fit_transform(df[['Games Played', 
                                                   'Completion Percentage',
                                                   'Pass Attempts Per Game',
                                                   'Passing Yards Per Attempt',
                                                   'Passing Yards Per Game',
                                                   'Percentage of TDs per Attempts',
                                                   'Int Rate',
                                                   'Passer Rating'
                                                  ]])
df.index = df['Name']
df.drop('Player Id', axis = 1, inplace = True)
df.drop('Name', axis = 1, inplace = True)
df


# In[24]:


df.describe()


# In[25]:


from sagemaker import PCA
bucket_name='sagemakernflstats1'
num_components=7

pca_SM = PCA(role=role,
          train_instance_count=1,
          train_instance_type='ml.c4.xlarge',
          output_path='s3://'+ bucket_name +'/passingstat/',
            num_components=num_components)


# In[26]:


train_data = df.values.astype('float32')


# In[27]:


get_ipython().run_cell_magic('time', '', 'pca_SM.fit(pca_SM.record_set(train_data))\n#already ran and have a training done -> fetch from jobs')


# In[28]:


job_name='pca-2019-06-20-15-43-03-130'
model_key = "passingstat/" + job_name + "/output/model.tar.gz"

boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')


# In[31]:


get_ipython().system('pip install mxnet')


# In[32]:


import mxnet as mx
pca_model_params = mx.ndarray.load('model_algo-1')


# In[33]:


s=pd.DataFrame(pca_model_params['s'].asnumpy())
v=pd.DataFrame(pca_model_params['v'].asnumpy())


# In[34]:


s.iloc[2:,:].apply(lambda x: x*x).sum()/s.apply(lambda x: x*x).sum()
#top 5 component accounts for 98% of the total variance in our dataset


# In[35]:


s_5=s.iloc[2:,:]
v_5=v.iloc[:,2:]
v_5.columns=[0,1,2,3,4]
#take only the 5 largest components from our original matrix


# In[36]:


import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


for i in range(1,6):
    first_comp = v_5[5-i]
    comps = pd.DataFrame(list(zip(first_comp, df.columns)), columns=['weights', 'features'])
    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
    ax=sns.barplot(data=comps.sort_values('abs_weights', ascending=False).head(10), x="weights", y="features", palette="Blues_d")
    ax.set_title("PCA Component Makeup: #" + str(i))
    plt.show()
#display all 5 components and their weights


# In[38]:


#PCA_list=['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5']

PCA_list=["Completion Percentage/Passer Rating", "Games Played/TDs per attempt", "Games Played/Pass attempt per game/pass yard per game",          "Int Rate/Completeion Percentage", "Completion Percentage/Games Played"]


# In[39]:


get_ipython().run_cell_magic('time', '', "pca_predictor = pca_SM.deploy(initial_instance_count=1, \n                                 instance_type='ml.t2.medium')\n#deploy an endpoint for the pca\n#MAKE SURE TO DELETE ENDPOINTS AFTER")


# In[40]:


df.shape


# In[41]:


get_ipython().run_cell_magic('time', '', "result = pca_predictor.predict(train_data)\ndf_transformed=pd.DataFrame()\n\nfor a in result:\n    b=a.label['projection'].float32_tensor.values\n    df_transformed=df_transformed.append([list(b)])")


# In[42]:


df_transformed.shape


# In[43]:


df_transformed.index=df.index
df_transformed=df_transformed.iloc[:,2:]
df_transformed.columns=PCA_list


# In[44]:


df_transformed.head()


# In[45]:


train_data = df_transformed.values.astype('float32')


# In[46]:


from sagemaker import KMeans

num_clusters = 5
kmeans = KMeans(role=role,
                train_instance_count=1,
                train_instance_type='ml.c4.xlarge',
                output_path='s3://'+ bucket_name +'/passingstat/',              
                k=num_clusters)


# In[47]:


get_ipython().run_cell_magic('time', '', 'kmeans.fit(kmeans.record_set(train_data))\n#create the training job for KMeans')


# In[48]:


get_ipython().run_cell_magic('time', '', "kmeans_predictor = kmeans.deploy(initial_instance_count=1, \n                                 instance_type='ml.t2.medium')\n#KMeans endpoint")


# In[49]:


get_ipython().run_cell_magic('time', '', 'result=kmeans_predictor.predict(train_data)')


# In[50]:


cluster_labels = [r.label['closest_cluster'].float32_tensor.values[0] for r in result]
pd.DataFrame(cluster_labels)[0].value_counts()


# In[51]:


ax=plt.subplots(figsize=(6,3))
ax=sns.distplot(cluster_labels, kde=False)
title="Histogram of Cluster Counts"
ax.set_title(title, fontsize=12)
plt.show()


# In[52]:


job_name='kmeans-2019-06-20-15-57-27-826'
model_key = "passingstat/" + job_name + "/output/model.tar.gz"

boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')


# In[53]:


Kmeans_model_params = mx.ndarray.load('model_algo-1')


# In[54]:


cluster_centroids=pd.DataFrame(Kmeans_model_params[0].asnumpy())
cluster_centroids.columns=df_transformed.columns
cluster_centroids


# In[55]:


plt.figure(figsize = (16, 6))
ax = sns.heatmap(cluster_centroids.T, cmap = 'YlGnBu')
ax.set_xlabel("Cluster")
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
ax.set_title("Attribute Value by Centroid")
plt.show()


# In[56]:


df_transformed['labels']=list(map(int, cluster_labels))
df_transformed.head()


# In[57]:


cluster=df_transformed[df_transformed['labels']==1]
cluster.head(5)


# In[58]:


print("DELETE ENDPOINTS AND ENDPOINT CONFIGS AND MODELS")


# In[59]:


cluster


# In[61]:


pass_df


# In[ ]:




