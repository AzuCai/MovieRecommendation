import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_rating_distribution(ratings):
    plt.figure(figsize=(8, 6))
    sns.histplot(ratings['rating'], bins=5, kde=True)
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()


def plot_actual_vs_predicted(true_ratings, predicted_ratings, sample_size=300):
    indices = np.random.choice(len(true_ratings), sample_size, replace=False)
    true_sample = true_ratings[indices]
    pred_sample = predicted_ratings[indices]

    plt.figure(figsize=(8, 6))
    plt.scatter(true_sample, pred_sample, alpha=0.3, color='blue', label='Predictions')
    plt.plot([1, 5], [1, 5], 'r--', label='Perfect Fit')
    plt.xlabel('True Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Predicted vs Actual Ratings (Sampled)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
