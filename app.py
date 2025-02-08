import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np

df = pd.read_csv("amazon_updated_data.csv")

list(df)

df2 = df

df3 = df2[['product_id', 'main_category', 'sub_category', 'rating', 'rating_count', 'user_id']]

from sklearn.preprocessing import LabelEncoder

print(df3['rating'].unique())

# Remove rows where 'rating' is not numeric
df3 = df3[pd.to_numeric(df3['rating'], errors='coerce').notnull()]

# Convert 'rating' to float
df3['rating'] = df3['rating'].astype(float)

le_category = LabelEncoder()
df3['main_category_encoded'] = le_category.fit_transform(df3['main_category'])
df3['sub_category_encoded'] = le_category.fit_transform(df3['sub_category'])

df3['rating_count'] = df3['rating_count'].replace({',': ''}, regex=True).astype(float)
df3['rating'] = df3['rating'].replace({'.': ''}, regex=True).astype(float)

df3['rating'] = pd.to_numeric(df3['rating'], errors='coerce')
df3['rating_count'] = pd.to_numeric(df3['rating_count'].replace({',': ''}, regex=True), errors='coerce')
df3['main_category_encoded'] = pd.to_numeric(df3['main_category_encoded'], errors='coerce')
df3['sub_category_encoded'] = pd.to_numeric(df3['sub_category_encoded'], errors='coerce')

print(df3[['rating', 'rating_count', 'main_category_encoded', 'sub_category_encoded']].dtypes)

df3['rating_count'].fillna(df3['rating_count'].mean(), inplace=True)

print(df3[['rating', 'rating_count', 'main_category_encoded', 'sub_category_encoded']].dtypes)

content_features = df3[['rating', 'rating_count', 'main_category_encoded', 'sub_category_encoded']].values
content_similarity = cosine_similarity(content_features)

def get_content_based_recommendations(product_id, top_n=5):
    product_idx = df3[df3['product_id'] == product_id].index[0]
    sim_scores = list(enumerate(content_similarity[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [x[0] for x in sim_scores[1:top_n+1]]
    return df3.iloc[top_indices]['product_id'].tolist()

product_id_to_recommend = df3['product_id'].iloc[0]
recommendations = get_content_based_recommendations(product_id_to_recommend, top_n=5)

print("Recommendations for Product ID", product_id_to_recommend)
print(recommendations)

import pickle

class ContentBasedRecommender:
    def __init__(self, similarity_matrix, product_df):
        self.similarity_matrix = similarity_matrix
        self.product_df = product_df

    def recommend(self, product_id, top_n=5):
        product_idx = self.product_df[self.product_df['product_id'] == product_id].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[product_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [x[0] for x in sim_scores[1:top_n+1]]
        return self.product_df.iloc[top_indices]['product_id'].tolist()

# Instantiate and save the model
recommender = ContentBasedRecommender(content_similarity, df3)
with open("content_recommender.pkl", "wb") as f:
    pickle.dump(recommender, f)

from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the recommender model
with open("content_recommender.pkl", "rb") as f:
    recommender = pickle.load(f)

# Create FastAPI app
app = FastAPI()

# Request model
class RecommendationRequest(BaseModel):
    product_id: int
    top_n: int = 5

@app.post("/recommend/")
def recommend_products(request: RecommendationRequest):
    recommendations = recommender.recommend(request.product_id, request.top_n)
    return {"recommendations": recommendations}