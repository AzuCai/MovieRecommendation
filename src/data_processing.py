import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(ratings_path='../data/ml-1m/ratings.dat', movies_path='../data/ml-1m/movies.dat'):
    ratings = pd.read_csv(ratings_path, sep='::', engine='python',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies = pd.read_csv(movies_path, sep='::', engine='python',
                         names=['movie_id', 'title', 'genres'], encoding='latin-1')
    return ratings, movies

def preprocess_data(ratings, movies, test_size=0.2):
    merged_data = ratings.merge(movies, on='movie_id')
    user_movie_matrix = merged_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    train_data, test_data = train_test_split(merged_data, test_size=test_size, random_state=42)
    return user_movie_matrix, train_data, test_data
