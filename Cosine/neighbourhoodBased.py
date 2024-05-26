import pandas as pd
import numpy as np
import time 
from scipy.sparse import csr_matrix, lil_matrix, diags
from scipy.sparse.linalg import norm as sparse_norm
from multiprocessing import Pool

# Load data from CSV file
def get_dataset(file_path='/home/elizavetasirotina/Documents/sem8/Result_last.csv'):
    return pd.read_csv(file_path)

def create_user_item_matrix(df):
    # Ensure all emails are lowercase to avoid duplicates
    df['customer_email'] = df['customer_email'].astype(str).str.lower()

    # Map each unique customer email and product id to a unique integer ID
    user_map = {email: i for i, email in enumerate(df['customer_email'].unique())}
    product_map = {product_id: i for i, product_id in enumerate(df['product_id'].unique())}

    # Prepare data for csr_matrix
    row = df['customer_email'].map(user_map)
    col = df['product_id'].map(product_map)
    data = df['total_purchases']

    return csr_matrix((data, (row, col)), shape=(len(user_map), len(product_map)))

# Function to create mask matrix from user-item matrix
# All non-zero cells will be replaced with 0 and vice versa
# It is needed to remove already purchased items from recommendations
def create_mask_matrix(user_item_matrix):
    mask_matrix = user_item_matrix.copy()
    mask_matrix.data = np.ones_like(mask_matrix.data) # sets all the data in mask_matrix to ones
    return csr_matrix(np.ones(mask_matrix.shape) - mask_matrix) # replaces ones with zeros and creates a dense function

# Apply mask to recommendation matrix to exclude already purchased items
def apply_mask(recommendation_matrix, mask_matrix):
    return recommendation_matrix.multiply(mask_matrix)

# Function to get values from user-item matrix for a given user and indices
def get_values(user_item_matrix, user_id, indices):
    return user_item_matrix[user_id, indices].toarray().flatten()

# Precomputes norms for rows in the user-item matrix
def precompute_user_norms(user_item_matrix):
     return sparse_norm(user_item_matrix, axis=1)

# Compute similarity for rows in chunks 
def compute_chunk(chunk, user_item_matrix, norms, num_users, num_neighbors):
    similarity_matrix_chunk = lil_matrix((num_users, num_users), dtype=np.float32)
    for i in chunk:
        # Retrieve indices of purchased items for user i
        observed_purchases_idx = user_item_matrix[i].indices
        if len(observed_purchases_idx) == 0:
            continue

        u1_values = get_values(user_item_matrix, i, observed_purchases_idx)
        norm_u1 = norms[i]
        # Boolean array indicating if users have purchased at least one of the same items as user i
        common_items_mask = (user_item_matrix[:, observed_purchases_idx].sum(axis=1) > 0).A1
        # Indices of users who purchased at least one of the same items as user i
        common_users_idx = np.where(common_items_mask)[0]

        # Array containing similarities between users
        user_similarities = []

        for j in common_users_idx:
            if i != j:
                u2_values = get_values(user_item_matrix, j, observed_purchases_idx)
                norm_u2 = np.linalg.norm(u2_values)
                divisor = norm_u1 * norm_u2

                if divisor > 0:
                    similarity = np.dot(u1_values.T, u2_values) / divisor ###
                    # if similarity > 0:
                    user_similarities.append((similarity, j))

        if len(user_similarities) > num_neighbors:
            user_similarities = sorted(user_similarities, reverse=True, key=lambda x: x[0])[:num_neighbors]

        for similarity, j in user_similarities:
            similarity_matrix_chunk[i, j] = similarity

    return similarity_matrix_chunk

# Parallel processing
def custom_similarity_matrix(user_item_matrix, num_neighbors, num_processors = 8):
    num_users = user_item_matrix.shape[0]
    norms = precompute_user_norms(user_item_matrix)

    # Split work into chunks
    chunks = np.array_split(range(num_users), num_processors)

    # Start parallel processing
    with Pool(num_processors) as pool:
        results = pool.starmap(compute_chunk, [(chunk, user_item_matrix, norms, num_users, num_neighbors) for chunk in chunks])

    # Combine results into one matrix
    similarity_matrix = lil_matrix((num_users, num_users), dtype=np.float32)
    for result in results:
        similarity_matrix += result

    return similarity_matrix.tocsr()

def create_recommendation_matrix(user_item_matrix_csr, similarity_matrix_csr):    
    # user_item_matrix_csr = user_item_matrix.tocsr()
    # similarity_matrix_csr = similarity_matrix.tocsr()

    recommendation_matrix = similarity_matrix_csr.dot(user_item_matrix_csr)
    
    # Normalize recommendations by the sum of absolute similarities for each user
    sums_of_similarities_rows = np.abs(similarity_matrix_csr).sum(axis=1).A.flatten()
    # Turn all zeros to ones
    sums_of_similarities_rows[sums_of_similarities_rows == 0] = 1
    normalization_diagonal = diags(1 / sums_of_similarities_rows)

    # Apply normalization
    recommendation_matrix = normalization_diagonal.dot(recommendation_matrix)
    
    return recommendation_matrix.tocsr()

# Retrieve N recommendations for each user
def get_top_n_recommendations(recommendation_matrix, N = 5):
    top_n_recommendations = {}
    for i in range(recommendation_matrix.shape[0]):
        row = recommendation_matrix.getrow(i).toarray().ravel()
        top_indices = np.argpartition(-row, N)[:N]
        top_n_recommendations[i] = top_indices.tolist()
    return top_n_recommendations

# Part of Testing: Randomly removes products for a selected percentage of eligible users
def remove_product_randomly(user_item_matrix, percentage=0.05):
    removed_products = {}
    eligible_users = []
    
    for user_index in range(user_item_matrix.shape[0]):
        purchased_product_indices = user_item_matrix[user_index].indices
        if len(purchased_product_indices) > 5:
            eligible_users.append(user_index)

    # Calculate how many users to select (5% of eligible users)
    num_selected_users = int(len(eligible_users) * percentage)
    
    # Randomly select 5% of these eligible users
    selected_users = np.random.choice(eligible_users, num_selected_users, replace=False)
    
    # Remove one product for each selected user
    for user_index in selected_users:
        purchased_product_indices = user_item_matrix[user_index].indices
        product_to_remove = np.random.choice(purchased_product_indices)
        user_item_matrix[user_index, product_to_remove] = 0
        removed_products[user_index] = product_to_remove
    
    user_item_matrix.eliminate_zeros()
    
    return user_item_matrix, removed_products

# Part of Testing: Calculate Score
def calculate_matched_score(removed_products, recommendations):
    match_count = 0
    # Iterate through each user in the removed products dictionary
    for user_index, removed_product in removed_products.items():        
        # Check if this user's removed product is in their recommendations
        if user_index in recommendations and removed_product in recommendations[user_index]:
            match_count += 1
    
    total_removed_products = len(removed_products)
    score = match_count / total_removed_products if total_removed_products > 0 else 0

    print(f"Score: {score} ({match_count} matches out of {len(removed_products)} removed products)")

def main(num_neighbors):
    start_time = time.time()
    print("Start", start_time)

    df = get_dataset()
    user_item_matrix = create_user_item_matrix(df)

    user_item_matrix, removed_products = remove_product_randomly(user_item_matrix)

    cosine_sim = custom_similarity_matrix(user_item_matrix, num_neighbors)
    mask_matrix = create_mask_matrix(user_item_matrix)
    recommendation_matrix = create_recommendation_matrix(user_item_matrix, cosine_sim)
    recommendation_matrix_new = apply_mask(recommendation_matrix, mask_matrix)
    N = 10
    recommendations = get_top_n_recommendations(recommendation_matrix_new, N)

    calculate_matched_score(removed_products, recommendations)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

if __name__ == '__main__':
    num_sim_users = [10, 20, 50, 94, 150, 200]
    num_recommendations = [5]

    for i in num_sim_users:
        for j in num_recommendations:
            print("Number of similar users =", i, ", Recommendations = ", j)
            main(i, j)
            main(i, j)
            main(i, j)
            main(i, j)
            main(i, j)
