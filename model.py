# model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import ast

def load_data():
    # Load the movie dataset
    movies = pd.read_csv('data/movies.csv')
    return movies

def vectorize_genres(movies):
    # Vectorizing genres column
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_vectors = vectorizer.fit_transform(movies['genres'])
    return genre_vectors

def build_knn_model(genre_vectors):
    # Building the KNN model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(genre_vectors)
    return model_knn

def recommend_movie_knn(model_knn, genre_vectors, movies, title, n_neighbors=10):
    # Recommend movies based on KNN model
    idx = movies[movies['title'] == title].index[0]
    distances, indices = model_knn.kneighbors(genre_vectors[idx], n_neighbors=n_neighbors)
    
    recommended_titles = []
    for i in indices.flatten():
        if i != idx:
            recommended_titles.append(movies.iloc[i]['title'])
    
    return recommended_titles
