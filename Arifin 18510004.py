#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


movies_df=pd.read_csv("F:/movie-KNN/movies.csv",
                     usecols=['movieId','title'],dtype={'movieId' : 'int32', 'title': 'str'})


# In[4]:


rating_df=pd.read_csv('F:/movie-KNN/ratings.csv',usecols=['userId', 'movieId' , 'rating'], 
                      dtype={'userId' : 'int32' , 'movieId' : 'int32' , 'rating' : 'float32'})


# In[5]:


movies_df.head()


# In[6]:


movies_df.shape


# In[7]:


rating_df.head()


# In[8]:


rating_df.shape


# In[9]:


df = pd.merge(rating_df, movies_df, on='movieId')
df


# In[10]:


combine_movie_rating =df.dropna(axis = 0, subset=['title'])
movie_ratingCount = (combine_movie_rating.
                    groupby(by =['title'])['rating'].
                    count().
                    reset_index().
                    rename(columns = {'rating' : 'totalRatingCount'})
                    [['title' , 'totalRatingCount']]
                    )
movie_ratingCount


# In[11]:


rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on='title', right_on='title', how= 'left')
rating_with_totalRatingCount.head()


# In[12]:


popularity_threshold = 75
rating_popular_movie=rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
rating_popular_movie.head()


# In[13]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(movie_ratingCount['totalRatingCount'].describe())


# In[14]:


rating_popular_movie.shape


# In[15]:


## First lets create pivot matrix

movie_features_df=rating_popular_movie.pivot_table(index='title',columns='userId', values='rating').fillna(0)
movie_features_df.head()


# In[16]:


from scipy.sparse import csr_matrix

movie_features_df_matrix = csr_matrix(movie_features_df.values)

from sklearn.neighbors import NearestNeighboars

model_knn=NearestNeighboard(metric = 'cosine', algorithm='brute')
model_knn.fit(movie_features_df_matrix)


# In[ ]:


movie_features_df.head()


# In[ ]:


query_index = np.random.choice(movie_features_df.shape[0])
print(query_index)
distances, indices = model_knn.kneighbors(movie.features_df.iloc[query_index,:].values.reshape(1,-1), n_neighbors = 10)


# In[ ]:


movie_features_df.head()


# In[1]:


for i in range(0, len (distance.flatten())):
    if i==0:
        print('Recomendation for {0}:\n'.format(movie_features_df.index[query.index]))
    else:
        print('{0}:{1}, with distance of {2}:\n'.format(i, movie_features_df.index[indices.flatten()[i]],distance flatten[i]))


# In[ ]:




