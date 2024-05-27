import numpy as np

def calculate_jaccard_similarity(args):
    """Helper function to calculate Jaccard similarity between two users."""
    user_i, user_j, user_item_matrix = args
    # Transform non-binary data to binary by checking if greater than 0
    items_i = user_item_matrix[user_i] > 0
    items_j = user_item_matrix[user_j] > 0
    print(items_i)
    # Calculate the Jaccard similarity
    intersection = np.sum(items_i & items_j)
    union = np.sum(items_i | items_j)
    similarity = intersection / union if union != 0 else 0
    return user_i, user_j, similarity

# Example non-binary user-item matrix
user_item_matrix = np.array([
    [-1, 0, 0, 3, 0],
    [0, 2, 7, 0, 4],
    [5, 3, 0, 3, 6],
    [0, 0, 5, 8, 0]
])

# Calculate Jaccard similarity between user 0 and user 2
args = (0, 2, user_item_matrix)
user_i, user_j, similarity = calculate_jaccard_similarity(args)

print(f"Jaccard similarity between User {user_i} and User {user_j}: {similarity:.2f}")
