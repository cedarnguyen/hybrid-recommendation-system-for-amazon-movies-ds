#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df1 = pd.read_csv('pretrained_meta_data.csv')

print(df1.head())


# In[3]:


data =df1


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Select features for clustering (exclude 'asin' and 'title' columns)
X = data.drop(['asin', 'title'], axis=1)

# Split the data into train and test sets 
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Fit KMeans model to the training data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)

# Predict clusters for the test data
test_clusters = kmeans.predict(X_test)

# Compute silhouette score
silhouette_avg = silhouette_score(X_test, test_clusters)
print("Silhouette Score:", silhouette_avg)


# In[ ]:


title_mapping = dict(zip(df['asin'], df['title']))


# In[5]:


import random

# Function to recommend similar items based on a user's interaction with one item
def recommend_similar_items(user_item_index, kmeans_model, data, X_test):
    # Select the user's item
    user_item = X_test.iloc[[user_item_index]]
    
    # Predict cluster for the user's item
    user_item_cluster = kmeans_model.predict(user_item)[0]

    # Generate boolean index based on the predicted cluster
    boolean_index = kmeans_model.labels_ == user_item_cluster
    
    # Ensure boolean index has the same length as X_test
    boolean_index = boolean_index[:len(X_test)]
    
    # Find similar items in the same cluster
    similar_items_indices = X_test[boolean_index].index.tolist()

    # Recommend similar items to the user
    similar_items = data.loc[similar_items_indices, 'title'].tolist()

    # Return only the first 10 recommended items
    return '\n'.join(similar_items[:10])

# Reset index of X_test
X_test.reset_index(drop=True, inplace=True)

# Ensure random_item_index is within bounds
random_item_index = random.randint(0, len(X_test) - 1)
user_item_title = data.loc[random_item_index, 'title']

# Recommend similar items to the user based on the simulated interaction
similar_items = recommend_similar_items(random_item_index, kmeans, data, X_test)

print("Simulated User's Item:", user_item_title)
print("Recommended Similar Items:", similar_items)

