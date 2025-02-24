import numpy as np
from sklearn.metrics import mean_squared_error


def evaluate_rmse(predictions, test_data, user_movie_matrix):
    user_indices = test_data['user_id'].values - 1
    movie_indices = test_data['movie_id'].map(lambda x: user_movie_matrix.columns.get_loc(x)).values
    true_ratings = test_data['rating'].values

    pred_ratings = predictions[user_indices, movie_indices]
    rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))

    return rmse
