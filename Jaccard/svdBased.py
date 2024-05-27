import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds
from multiprocessing import Pool
import time 

def load_dataset():
    return pd.read_csv('/home/elizavetasirotina/Documents/sem8/Result_last.csv')

def get_reduced_user_matrix(matrix, k):
    U, s, Vh = svds(matrix, k=k)
    
    # Sort the singular values (and vectors) in descending order because svds doesn't sort values
    idx = np.argsort(s)[::-1]  # Get the indices that would sort s in descending order
    s_sorted = s[idx]
    U_sorted = U[:, idx]
    # Vh_sorted = Vh[idx, :]
    
    return np.dot(U_sorted, np.diag(s_sorted))

def create_user_item_matrix(df):
    # Make sure all emails are lower case to avoid dublicates
    df['customer_email'] = df['customer_email'].astype(str).str.lower()

    user_map = {email: i for i, email in enumerate(df['customer_email'].unique())}
    product_map = {product_id: i for i, product_id in enumerate(df['product_id'].unique())}

    # Prepare data for csr_matrix
    row = df['customer_email'].map(user_map)
    col = df['product_id'].map(product_map)
    data = df['total_purchases']

    return csr_matrix((data, (row, col)), shape=(len(user_map), len(product_map)))

# Function to create mask matrix from user-item matrix
def create_mask_matrix(user_item_matrix):
    # Mask non-zero elements in user-item matrix
    mask_matrix = user_item_matrix.copy()
    mask_matrix.data = np.ones_like(mask_matrix.data)
    return csr_matrix(np.ones(mask_matrix.shape) - mask_matrix)

def apply_mask(recommendation_matrix, mask_matrix):
    return recommendation_matrix.multiply(mask_matrix)

# Function to get values from user-item matrix for a given user and indices
def get_values(user_item_matrix, user_id, indices):
    values = np.zeros(len(indices))
    for i in range(len(indices)):
        values[i] = user_item_matrix[user_id, indices[i]]
    return values

# Function to calculate jaccard similarity 
def calculate_jaccard_similarity(args):
    user_i, user_j, user_item_matrix = args
    # Ensure the matrix row slicing matches numpy array handling
    items_i = user_item_matrix[user_i] > 0
    items_j = user_item_matrix[user_j] > 0
    # Calculate the Jaccard similarity
    intersection = np.sum(items_i & items_j)
    union = np.sum(items_i | items_j)
    similarity = intersection / union if union != 0 else 0
    return user_i, user_j, similarity

# Function to compute a custom similarity matrix using Jaccard similarity with parallel processing.
def custom_similarity_matrix(user_item_matrix, num_neighbors, num_processes=None):
    num_users = user_item_matrix.shape[0]
    similarity_matrix = np.zeros((num_users, num_users))

    # Prepare the arguments for parallel processing
    args = [(i, j, user_item_matrix) for i in range(num_users) for j in range(i + 1, num_users)]

    # Calculate Jaccard similarity for each pair of users in parallel using Pool
    from multiprocessing import Pool
    with Pool(processes=num_processes) as pool:
        results = pool.map(calculate_jaccard_similarity, args)

    # Fill the similarity matrix with the results
    for user_i, user_j, similarity in results:
        similarity_matrix[user_i, user_j] = similarity
        similarity_matrix[user_j, user_i] = similarity

    result_matrix = np.zeros_like(similarity_matrix)

    # Retain only the top num_neighbors similarities for each user
    for user_i in range(num_users):
        top_n_indices = np.argsort(similarity_matrix[user_i])[-num_neighbors:]
        result_matrix[user_i, top_n_indices] = similarity_matrix[user_i, top_n_indices]
        result_matrix[user_i, user_i] = 0

    return csr_matrix(result_matrix)

def create_recommendation_matrix(user_item_matrix, similarity_matrix):    
    user_item_matrix_csr = user_item_matrix.tocsr()
    similarity_matrix_csr = similarity_matrix

    recommendation_matrix = similarity_matrix_csr.dot(user_item_matrix_csr)
    
    sum_of_similarities = np.abs(similarity_matrix_csr).sum(axis=1).A.flatten()
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

# Part of testing
def remove_product_randomly(user_item_matrix, percentage=0.05):
    removed_products = {}
    eligible_users = []
    
    for user_index in range(user_item_matrix.shape[0]):
        purchased_product_indices = user_item_matrix[user_index].indices
        if len(purchased_product_indices) > 5:
            eligible_users.append(user_index)
    
    num_selected_users = int(len(eligible_users) * percentage)
    
    selected_users = np.random.choice(eligible_users, num_selected_users, replace=False)
    
    for user_index in selected_users:
        purchased_product_indices = user_item_matrix[user_index].indices
        product_to_remove = np.random.choice(purchased_product_indices)
        user_item_matrix[user_index, product_to_remove] = 0
        removed_products[user_index] = product_to_remove
    
    user_item_matrix.eliminate_zeros()
    
    return user_item_matrix, removed_products

# Part of testing
def calculate_matched_score(removed_products, recommendations):
    match_count = 0
    for user_index, removed_product in removed_products.items():        
        if user_index in recommendations and removed_product in recommendations[user_index]:
            match_count += 1
    
    total_removed_products = len(removed_products)
    score = match_count / total_removed_products if total_removed_products > 0 else 0

    append_line_to_file(f"Score: {score} ({match_count} matches out of {len(removed_products)} removed products)")

    return score

def append_line_to_file(line):
    with open('jaccardSVD.txt', 'a') as file:  # 'a' opens the file in append mode
        file.write(line + "\n")  # Add newline character for the next line

def main(num_neighbors, num_sing_val, num_recommendations = 5):
    start_time = time.time()
    
    df = load_dataset()

    user_item_matrix_init = create_user_item_matrix(df)
    user_item_matrix, removed_products = remove_product_randomly(user_item_matrix_init)

    user_feature_matrix = get_reduced_user_matrix(user_item_matrix, num_sing_val)

    jaccard_sim = custom_similarity_matrix(user_feature_matrix, num_neighbors)

    mask_matrix = create_mask_matrix(user_item_matrix)
    recommendation_matrix = create_recommendation_matrix(user_item_matrix, jaccard_sim)
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