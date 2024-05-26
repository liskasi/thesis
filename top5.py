import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def get_dataset():
    return pd.read_csv('/home/elizavetasirotina/Documents/sem8/Result_last.csv')

def create_user_item_matrix(df):
    df['customer_email'] = df['customer_email'].astype(str).str.lower()
    user_map = {email: i for i, email in enumerate(df['customer_email'].unique())}
    product_map = {product_id: i for i, product_id in enumerate(df['product_id'].unique())}
    row = df['customer_email'].map(user_map)
    col = df['product_id'].map(product_map)
    data = df['total_purchases']
    user_item_matrix = csr_matrix((data, (row, col)), shape=(len(user_map), len(product_map)))
    return user_item_matrix, product_map

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

def calculate_matched_score(removed_products, popular_items):
    match_count = 0
    for user_index, removed_product in removed_products.items():
        if removed_product in popular_items:
            match_count += 1
    total_removed_products = len(removed_products)
    score = match_count / total_removed_products if total_removed_products > 0 else 0
    print(f"Score: {score} ({match_count} matches out of {total_removed_products} removed products)")
    return score

def get_all_product_indices_by_popularity(df, product_map):
    product_popularity = df.groupby('product_id')['total_purchases'].sum().reset_index()
    product_popularity = product_popularity.sort_values(by='total_purchases', ascending=False)
    top_n_popular_items = product_popularity.head(5)
    all_product_ids = top_n_popular_items['product_id'].tolist()
    all_product_indices = [product_map[product_id] for product_id in all_product_ids if product_id in product_map]
    return all_product_indices

def main():
    df = get_dataset()
    user_item_matrix, product_map = create_user_item_matrix(df)
    user_item_matrix, removed_products = remove_product_randomly(user_item_matrix)
    popular_items = get_all_product_indices_by_popularity(df, product_map)
    calculate_matched_score(removed_products, popular_items)

if __name__ == '__main__':
    main()

