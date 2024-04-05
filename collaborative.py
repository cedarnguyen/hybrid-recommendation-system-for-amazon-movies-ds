#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df1 = pd.read_csv('pretrained_viewer_data.csv')

print(df1.head())


# In[2]:


df1.head()


# In[3]:


df =df1


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


asin_mapping = dict(zip(range(len(df['asin'].unique())), df['asin'].unique()))


# In[7]:


df = df.drop(['asin'], axis =1)


# In[8]:


from sklearn.model_selection import train_test_split
def split_train_test_data(df, test_size=0.2, random_state=42):
    """
    Split the dataset into train and test sets.
    """
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_data, test_data


# In[9]:


from sklearn.neighbors import NearestNeighbors

def train_knn_model(train_data, k=5):
    """
    Train the Nearest Neighbors model using the training data.
    """
    knn_model_train = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn_model_train.fit(train_data[['reviewerID', 'asin_t', 'overall', 'helpfulness_ratio']])
    return knn_model_train


# In[10]:


def recommend_for_user(user_id, train_data, test_data, knn_model_train, asin_mapping, N=10):
    """
    Recommend items for the given user.
    """
    if user_id in train_data['reviewerID'].unique():
        user_ratings = test_data[test_data['reviewerID'] == user_id][['reviewerID', 'asin_t', 'overall', 'helpfulness_ratio']]
        if len(user_ratings) > 0:
            distances, indices = knn_model_train.kneighbors(user_ratings)
            neighbor_ratings = train_data.iloc[indices.flatten()]
            neighbor_ratings = neighbor_ratings[neighbor_ratings['reviewerID'] != user_id]
            if not neighbor_ratings.empty:
                top_n_recommendations = neighbor_ratings.groupby('asin_t')['overall'].mean().sort_values(ascending=False).head(N)
                recommended_items_asin = top_n_recommendations.index.tolist()
                recommended_items = [asin_mapping[encoded_asin] for encoded_asin in recommended_items_asin]
                print(f"Top {len(recommended_items)} recommendations for user {user_id}: {recommended_items}")
                return recommended_items
            else:
                print("No neighbors found for the user.")
        else:
            print("No ratings found for the user. Recommending popular items.")
            popularity_recommendations = test_data['asin_t'].value_counts().index[:N].tolist()
            recommends = [asin_mapping[encoded_asin] for encoded_asin in popularity_recommendations]
            print(f"Here are the top {N} most popular items: {recommends}")
            return recommends
    else:
        print("This is a new user.")
        popularity_recommendations = test_data['asin_t'].value_counts().index[:N].tolist()
        recommends = [asin_mapping[encoded_asin] for encoded_asin in popularity_recommendations]
        print(f"Since you are a new user, here are the top {N} most popular items: {recommends}")
        return recommends


# In[11]:


'''
def calculate_precision_and_map(user_id, recommended_items, N, ground_truth):
    """
    Calculate precision at N and Mean Average Precision at N.
    """
    ground_truth = pd.DataFrame({'reviewerID': [user_id] * len(recommended_items), 'asin': recommended_items})
    K = min(N, len(ground_truth))
    relevant_items_at_K = set(ground_truth['asin'][:K])
    recommended_items_at_K = set(recommended_items[:K])
    precision_at_K = len(relevant_items_at_K.intersection(recommended_items_at_K)) / K
    print(f"Precision at {K}: {precision_at_K}")

    relevant_indices = [i for i, item in enumerate(recommended_items) if item in relevant_items_at_K]
    average_precision = 0
    for i, index in enumerate(relevant_indices, 1):
        average_precision += i / (index + 1)
    average_precision /= len(relevant_items_at_K)
    print(f"Mean Average Precision at {K}: {average_precision}")
'''


# In[12]:


# Usage example
import random
train_data, test_data = split_train_test_data(df)
knn_model_train = train_knn_model(train_data)
user_id = random.randint(1, 2088619)
#user_id = int('1520337')
recommended_items = recommend_for_user(user_id, train_data, test_data, knn_model_train, asin_mapping)

'''
if recommended_items:
    calculate_precision_and_map(user_id, recommended_items, 10, None)
'''

