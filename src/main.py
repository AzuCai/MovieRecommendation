# Step 1: Import Libraries

from data_processing import load_data, preprocess_data
from model import collaborative_filtering
from evaluation import evaluate_rmse
from visualization import plot_rating_distribution, plot_actual_vs_predicted

# Step 2: Load and preprocess data
ratings, movies = load_data()
user_movie_matrix, train_data, test_data = preprocess_data(ratings, movies)

# Step 3: Model Training
svd_predictions = collaborative_filtering(user_movie_matrix, n_components=150)

# Step 4: Evaluation
rmse = evaluate_rmse(svd_predictions, test_data, user_movie_matrix)
print(f'RMSE: {rmse:.4f}')

# Step 5: Visualization
plot_rating_distribution(ratings)

# Prepare data for visualization
user_indices = test_data['user_id'].values - 1
movie_indices = test_data['movie_id'].map(lambda x: user_movie_matrix.columns.get_loc(x)).values
true_ratings = test_data['rating'].values
predicted_ratings = svd_predictions[user_indices, movie_indices]

plot_actual_vs_predicted(true_ratings, predicted_ratings)
