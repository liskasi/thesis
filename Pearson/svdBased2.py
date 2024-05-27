import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr
from multiprocessing import Pool
import time

def load_dataset():
    return pd.read_csv('/home/elizavetasirotina/Documents/sem8/Result_last.csv')

def get_reduced_user_matrix(matrix, k):
    U, s, Vh = svds(matrix, k=k)
    idx = np.argsort(s)[::-1]
    s_sorted = s[idx]
    U_sorted = U[:, idx]
    return U_sorted @ np.diag(s_sorted)

def create_user_item_matrix(df):
    df['customer_email'] = df['customer_email'].astype(str).str.lower()
    user_map = {email: i for i, email in enumerate(df['customer_email'].unique())}
    product_map = {product_id: i for i, product_id in enumerate(df['product_id'].unique())}
    row = df['customer_email'].map(user_map)
    col = df['product_id'].map(product_map)
    data = df['total_purchases']
    return csr_matrix((data, (row, col)), shape=(len(user_map), len(product_map)))

def create_mask_matrix(user_item_matrix):
    mask_matrix = user_item_matrix.copy()
    mask_matrix.data = np.ones_like(mask_matrix.data)
    return csr_matrix(np.ones(mask_matrix.shape) - mask_matrix)

def apply_mask(recommendation_matrix, mask_matrix):
    return recommendation_matrix.multiply(mask_matrix)

def calculate_similarity(args):
    user_i, user_j, user_item_matrix = args
    similarity, _ = pearsonr(user_item_matrix[user_i], user_item_matrix[user_j])
    return user_i, user_j, similarity

def custom_similarity_matrix(user_item_matrix, num_neighbors, num_processes=None):
    num_users = user_item_matrix.shape[0]
    similarity_matrix = np.zeros((num_users, num_users))
    args = [(i, j, user_item_matrix) for i in range(num_users) for j in range(i + 1, num_users)]
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(calculate_similarity, args)

    for user_i, user_j, similarity in results:
        similarity_matrix[user_i, user_j] = similarity
        similarity_matrix[user_j, user_i] = similarity

    result_matrix = np.zeros_like(similarity_matrix)
    for user_i in range(num_users):
        top_n_indices = np.argsort(similarity_matrix[user_i])[-num_neighbors:]
        result_matrix[user_i, top_n_indices] = similarity_matrix[user_i, top_n_indices]
        result_matrix[user_i, user_i] = 0

    return csr_matrix(result_matrix)

def create_recommendation_matrix(user_item_matrix, similarity_matrix):
    recommendation_matrix = similarity_matrix.dot(user_item_matrix)
    sum_of_similarities = np.abs(similarity_matrix).sum(axis=1).A.flatten()
    sum_of_similarities[sum_of_similarities == 0] = 1
    normalization_diagonal = diags(1 / sum_of_similarities)
    recommendation_matrix = normalization_diagonal.dot(recommendation_matrix)
    return recommendation_matrix.tocsr()

def get_top_n_recommendations(masked_recommendation_matrix, N):
    top_n_recommendations = {}
    for i in range(masked_recommendation_matrix.shape[0]):
        row = masked_recommendation_matrix.getrow(i).toarray().ravel()
        top_indices = np.argpartition(-row, N)[:N]
        top_n_recommendations[i] = top_indices.tolist()
    return top_n_recommendations

def remove_product_randomly(user_item_matrix, percentage=0.05):
    removed_products = {}
    eligible_users = [i for i in range(user_item_matrix.shape[0]) if len(user_item_matrix[i].indices) > 5]
    num_selected_users = int(len(eligible_users) * percentage)
    selected_users = np.random.choice(eligible_users, num_selected_users, replace=False)

    for user_index in selected_users:
        purchased_product_indices = user_item_matrix[user_index].indices
        product_to_remove = np.random.choice(purchased_product_indices)
        user_item_matrix[user_index, product_to_remove] = 0
        removed_products[user_index] = product_to_remove

    user_item_matrix.eliminate_zeros()
    return user_item_matrix, removed_products

def calculate_matched_score(removed_products, recommendations):
    match_count = sum(1 for user_index, removed_product in removed_products.items() if user_index in recommendations and removed_product in recommendations[user_index])
    score = match_count / len(removed_products) if removed_products else 0
    append_line_to_file(f"Score: {score} ({match_count} matches out of {len(removed_products)} removed products)")
    # print(f"Score: {score} ({match_count} matches out of {len(removed_products)} removed products)")
    # return score

def append_line_to_file(line):
    with open('pearsonSVD.txt', 'a') as file:  # 'a' opens the file in append mode
        file.write(line + "\n")  # Add newline character for the next line

def main(num_neighbors, num_sing_val, num_recommendations=5):
    start_time = time.time()
    # print("Start", start_time)

    df = load_dataset()
    user_item_matrix_init = create_user_item_matrix(df)
    user_item_matrix, removed_products = remove_product_randomly(user_item_matrix_init)
    user_feature_matrix = get_reduced_user_matrix(user_item_matrix, num_sing_val)
    cosine_sim = custom_similarity_matrix(user_feature_matrix, num_neighbors)

    mask_matrix = create_mask_matrix(user_item_matrix)
    recommendation_matrix = create_recommendation_matrix(user_item_matrix, cosine_sim)
    recommendation_matrix_new = apply_mask(recommendation_matrix, mask_matrix)
    recommendations = get_top_n_recommendations(recommendation_matrix_new, num_recommendations)
    calculate_matched_score(removed_products, recommendations)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

if __name__ == '__main__':
    num_neighbors = [10, 20, 50, 94, 150, 200]
    num_recommendations = [5]
    num_sing_val = [5, 50, 125, 250, 514]

    for i in num_neighbors:
        for j in num_sing_val:
            append_line_to_file(f"Number of similar users = {i}, Number of singular values = {j}")
            main(i, j)
            main(i, j)
            main(i, j)
            main(i, j)
            main(i, j)
