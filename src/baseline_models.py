import pandas as pd
import numpy as np
from surprise import Reader, Dataset, BaselineOnly, KNNBasic, SVD, NMF, accuracy
from surprise.model_selection import train_test_split  # not used here, but available
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import DataLoader
from data_preprocessing import k_core_filter, temporal_split

# --- Load and prepare data ---
loader = DataLoader("dataset/ml-1m")
ratings = loader.load_rating()
movies = loader.load_movies()

filtered = k_core_filter(ratings, k=20)
train_df, test_df = temporal_split(filtered, train_ratio=0.8)

# Phase 3: Baselines with Surprise (Record Everything)

# We train on `train_df` only, evaluate predictions against `test_df`.

# Surprise requires Reader scale 1-5 (ML-1M is already 1-5)
reader = Reader(rating_scale=(1, 5))
train_set = Dataset.load_from_df(train_df[['user_id', 'movie_id', 'rating']], reader).build_full_trainset()
test_set = list(test_df[['user_id', 'movie_id', 'rating']].itertuples(index=False, name=None))


def evaluate_model(algo, name, k=10):
    algo.fit(train_set)
    predictions = algo.test(test_set)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    # For Precision@K we need top-N; Surprise accuracy.precision_recall_at_k requires predictions with est
    # We'll build a helper later; here we just record RMSE/MAE
    print(f"{name:20s} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse, mae, predictions


results = {}

# 1. Global / Popularity approximation: NormalPredictor (random around global mean) is weak;
#    Instead use BaselineOnly for true popularity+user/item bias model
results['BaselineOnly'] = evaluate_model(BaselineOnly(), 'BaselineOnly')

# 2. UserCF (user-based collaborative)
results['UserCF'] = evaluate_model(KNNBasic(sim_options={'name': 'cosine', 'user_based': True}), 'UserCF')

# 3. ItemCF
results['ItemCF'] = evaluate_model(KNNBasic(sim_options={'name': 'cosine', 'user_based': False}), 'ItemCF')

# 4. Explicit Matrix Factorization (SVD)
results['SVD'] = evaluate_model(SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02), 'SVD')

# 5. NMF
results['NMF'] = evaluate_model(NMF(n_factors=50, n_epochs=20), 'NMF')

# 6. Content-Based: We'll implement manually in Phase 3b, but for Surprise-style comparison
#    we will generate predictions externally.

# Store raw predictions for later feature engineering
itemcf_model = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
itemcf_model.fit(train_set)
itemcf_predictions = itemcf_model.test(test_set)

# --- 3b Content-Based Filtering (Classical TF-IDF + Cosine) ---

# Build text: title + genres
movies_train = movies.copy()
movies_train['text'] = movies_train['title'] + ' ' + movies_train['genres'].str.replace('|', ' ')

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_mat = tfidf.fit_transform(movies_train['text'])

# Latent semantic indexing via TruncatedSVD (classical LSI replacement for embeddings)
svd_lsi = TruncatedSVD(n_components=50, random_state=42)
lsi = svd_lsi.fit_transform(tfidf_mat)  # movie x 50
lsi_df = pd.DataFrame(lsi, index=movies_train['movie_id'])

# User profile: mean of LSI vectors for movies they rated in train
user_profiles = train_df.groupby('user_id')['movie_id'].apply(
    lambda x: np.mean([lsi_df.loc[m] for m in x if m in lsi_df.index], axis=0)
)
user_profiles_df = pd.DataFrame(user_profiles.tolist(), index=user_profiles.index)


def cb_predict(user_id, movie_id):
    if user_id not in user_profiles_df.index or movie_id not in lsi_df.index:
        return train_df['rating'].mean()
    user_vec = user_profiles_df.loc[user_id].values.reshape(1, -1)
    item_vec = lsi_df.loc[movie_id].values.reshape(1, -1)
    return cosine_similarity(user_vec, item_vec)[0, 0] * 4 + 1  # scale to 1-5 approx


# Evaluate CBF on full test set
cb_pred_list = []
for u, m, r in test_set:
    est = cb_predict(u, m)
    cb_pred_list.append((u, m, r, est))
# Compute RMSE/MAE manually
cb_est = np.array([x[3] for x in cb_pred_list])
cb_true = np.array([x[2] for x in cb_pred_list])
cb_rmse = np.sqrt(np.mean((cb_est - cb_true) ** 2))
cb_mae = np.mean(np.abs(cb_est - cb_true))
print(f"CBF-TF-IDF-LSI | RMSE: {cb_rmse:.4f} | MAE: {cb_mae:.4f}")

# Paper note: TF-IDF + TruncatedSVD is your explicit replacement for BERT/GloVe — emphasize "pure classical text processing".
