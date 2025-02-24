import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def collaborative_filtering(user_movie_matrix, n_components=150):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_latent = svd.fit_transform(user_movie_matrix)
    movie_latent = svd.components_.T
    predictions = np.dot(user_latent, movie_latent.T)
    predictions = np.clip(predictions, 1, 5)
    return predictions


def content_based_filtering(movie_features, user_prefs):
    movie_matrix = movie_features.drop(columns=['movie_id']).values
    user_matrix = user_prefs.values

    similarity_matrix = cosine_similarity(user_matrix, movie_matrix)
    predictions = np.clip(similarity_matrix, 0, 1) * 4 + 1
    return predictions


def hybrid_recommendation(svd_preds, cb_preds, alpha=0.65):
    hybrid_preds = alpha * svd_preds + (1 - alpha) * cb_preds
    hybrid_preds = np.clip(hybrid_preds, 1, 5)
    return hybrid_preds
