#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference Script for GNN-based Recommendation System
Loads a trained model and runs inference on test data
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Check for GPU availability
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# Parse command line arguments
parser = argparse.ArgumentParser(description='GNN Recommendation System Inference')
parser.add_argument('--data_path', type=str, default='../ratings_Electronics (1).csv',
                    help='Path to the CSV dataset')
parser.add_argument('--model_path', type=str, default='Recommendation System/models/final_model.pt',
                    help='Path to the trained model')
parser.add_argument('--output_path', type=str, default='Recommendation System/inference',
                    help='Path to save inference results')
parser.add_argument('--sample_size', type=float, default=0.1, 
                    help='Fraction of data to use for inference (1.0 means all data)')
parser.add_argument('--top_k', type=int, default=10,
                    help='Number of top recommendations to generate for each user')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size for inference')
parser.add_argument('--user_id', type=int, default=None, help='Specific user ID to generate recommendations for')
parser.add_argument('--evaluate_user', type=int, default=None, help='Specific user ID to evaluate the model on')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_path, exist_ok=True)

# Timer for performance tracking
class Timer:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        print(f"{self.name} took {self.end - self.start:.2f} seconds")

# Define the GNN model (same as in training script)
class GNNRecommender(nn.Module):
    def __init__(self, num_nodes, hidden_channels, embedding_dim=64, model_type='GCN', dropout=0.3, l2_reg=1e-5):
        super(GNNRecommender, self).__init__()
        self.model_type = model_type
        self.dropout = dropout
        self.l2_reg = l2_reg
        
        # Node embedding layer instead of one-hot encoding
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # First layer
        if model_type == 'GCN':
            self.conv1 = GCNConv(embedding_dim, hidden_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(embedding_dim, hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(embedding_dim, hidden_channels)
        
        # Second layer
        if model_type == 'GCN':
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv2 = GATConv(hidden_channels, hidden_channels)
        
        # Third layer (adding more depth with residual connection)
        if model_type == 'GCN':
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv3 = GATConv(hidden_channels, hidden_channels)
        
        # Output layer for rating prediction
        self.out = nn.Linear(hidden_channels * 2, 1)
    
    def encode(self, x, edge_index):
        # Get node embeddings
        x = self.node_embedding(x)
        
        # First layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second layer
        x_res = x  # Store for residual connection
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third layer with residual connection
        x = self.conv3(x, edge_index) + x_res  # Residual connection
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def decode(self, z, edge_index):
        # Get node features for the source and target nodes of each edge
        src, dst = edge_index
        src_z = z[src]
        dst_z = z[dst]
        
        # Concatenate the node features
        edge_features = torch.cat([src_z, dst_z], dim=1)
        
        # Pass through output layer to get rating prediction
        return self.out(edge_features).squeeze()
    
    def forward(self, x, edge_index):
        # Encode nodes
        z = self.encode(x, edge_index)
        
        # Decode edges to get ratings
        ratings = self.decode(z, edge_index)
        
        # For inference, we don't need L2 regularization
        if self.training:
            l2_loss = 0
            for param in self.parameters():
                l2_loss += torch.norm(param, 2)
            return ratings, l2_loss * self.l2_reg
        else:
            return ratings

def load_and_preprocess_data(file_path, sample_size=0.1):
    """Load and preprocess the dataset for inference."""
    print(f"Loading data from {file_path}...")
    
    with Timer("Data loading"):
        # Use chunking for large files
        if sample_size < 1.0:
            # Read a sample of the data
            df = pd.read_csv(file_path, header=None, names=['userId', 'productId', 'rating', 'timestamp'])
            df = df.sample(frac=sample_size, random_state=42)
        else:
            # Read the entire file
            df = pd.read_csv(file_path, header=None, names=['userId', 'productId', 'rating', 'timestamp'])
    
    print(f"Loaded {len(df)} ratings")
    
    with Timer("Data preprocessing"):
        # Map user and product IDs to consecutive integers
        unique_users = df['userId'].unique()
        unique_products = df['productId'].unique()
        
        print(f"Number of unique users: {len(unique_users)}")
        print(f"Number of unique products: {len(unique_products)}")
        
        user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        product_id_map = {old_id: new_id + len(user_id_map) for new_id, old_id in enumerate(unique_products)}
        
        # Create reverse mappings for later use
        reverse_user_map = {v: k for k, v in user_id_map.items()}
        reverse_product_map = {v: k for k, v in product_id_map.items()}
        
        # Verify mappings
        print(f"User ID map size: {len(user_id_map)}")
        print(f"Product ID map size: {len(product_id_map)}")
        print(f"Reverse user map size: {len(reverse_user_map)}")
        print(f"Reverse product map size: {len(reverse_product_map)}")
        
        # Apply mapping
        df['user_id'] = df['userId'].map(user_id_map)
        df['product_id'] = df['productId'].map(product_id_map)
        
        # Verify all mappings were successful
        if df['user_id'].isna().any() or df['product_id'].isna().any():
            print("WARNING: Some mappings failed!")
            # Remove rows with failed mappings
            df = df.dropna(subset=['user_id', 'product_id'])
            print(f"Remaining rows after dropping NaN values: {len(df)}")
        
        # Split data into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        print(f"Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Get total number of nodes (users + products)
        num_nodes = len(user_id_map) + len(product_id_map)
        print(f"Total number of nodes: {num_nodes}")
        
        # Create edge indices and attributes for PyTorch Geometric
        train_edge_index = torch.tensor(np.array([train_df['user_id'].values, train_df['product_id'].values]), dtype=torch.long)
        train_edge_attr = torch.tensor(train_df['rating'].values, dtype=torch.float)
        
        test_edge_index = torch.tensor(np.array([test_df['user_id'].values, test_df['product_id'].values]), dtype=torch.long)
        test_edge_attr = torch.tensor(test_df['rating'].values, dtype=torch.float)
        
        # Create node indices for all nodes (0 to num_nodes-1)
        node_indices = torch.arange(num_nodes, dtype=torch.long)
        
        # Create PyTorch Geometric Data objects
        train_data = Data(x=node_indices, edge_index=train_edge_index, edge_attr=train_edge_attr)
        test_data = Data(x=node_indices, edge_index=test_edge_index, edge_attr=test_edge_attr)
        
        return train_data, test_data, test_df, num_nodes, reverse_user_map, reverse_product_map

def load_model(model_path, num_nodes):
    """Load the trained model."""
    print(f"Loading model from {model_path}...")
    
    try:
        # Option 1: Less secure but simpler approach - allows all Python objects to be unpickled
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Option 2: More secure approach - only allows specific functions to be unpickled
        # Uncomment the following lines to use this approach instead:
        # import numpy as np
        # from torch.serialization import add_safe_globals
        # add_safe_globals([np.core.multiarray.scalar])
        # checkpoint = torch.load(model_path, map_location=device)
        
        model_args = checkpoint['args']
        
        # Get the original model's embedding size
        original_num_nodes = checkpoint['model_state_dict']['node_embedding.weight'].shape[0]
        print(f"Original model was trained with {original_num_nodes} nodes")
        print(f"Current dataset has {num_nodes} nodes")
        
        # Create model with the ORIGINAL number of nodes to match the saved weights
        model = GNNRecommender(
            num_nodes=original_num_nodes,  # Use original size instead of current num_nodes
            hidden_channels=model_args['hidden_channels'],
            embedding_dim=model_args['embedding_dim'],
            model_type=model_args['model_type'],
            dropout=model_args['dropout'],
            l2_reg=model_args['l2_reg']
        ).to(device)
        
        # Load the model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        model.eval()
        
        print(f"Model loaded successfully")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. If you're using PyTorch 2.6+, try setting weights_only=False (already implemented)")
        print("2. If you're still having issues, try using the Option 2 approach with add_safe_globals")
        print("3. If the model was saved with an older PyTorch version, consider re-saving it with your current version")
        print("4. If you're using a different sample_size than what was used for training, try using the same sample_size")
        raise

def run_inference(model, test_data, test_df, reverse_user_map, reverse_product_map):
    """Run inference on the test data."""
    print("Running inference on test data...")
    
    # Move data to device
    test_data = test_data.to(device)
    
    # Get the model's embedding size
    model_num_nodes = model.node_embedding.weight.shape[0]
    
    # Check if any node indices in test_data are out of bounds for the model
    max_node_idx = test_data.x.max().item()
    if max_node_idx >= model_num_nodes:
        print(f"Warning: Test data contains node indices up to {max_node_idx}, but model only supports up to {model_num_nodes-1}")
        print("Filtering out edges with out-of-bounds node indices...")
        
        # Filter out edges with node indices that are out of bounds for the model
        edge_index = test_data.edge_index
        edge_attr = test_data.edge_attr
        
        # Create mask for valid edges (both source and target nodes are within bounds)
        valid_edges_mask = (edge_index[0] < model_num_nodes) & (edge_index[1] < model_num_nodes)
        
        # Apply mask
        filtered_edge_index = edge_index[:, valid_edges_mask]
        filtered_edge_attr = edge_attr[valid_edges_mask]
        
        print(f"Filtered out {(~valid_edges_mask).sum().item()} edges, {valid_edges_mask.sum().item()} remaining")
        
        # Create new test_data with filtered edges
        test_data = Data(x=test_data.x, edge_index=filtered_edge_index, edge_attr=filtered_edge_attr)
        test_data = test_data.to(device)
    
    # Run inference
    with torch.no_grad():
        with Timer("Inference"):
            predictions = model(test_data.x, test_data.edge_index)
    
    # Convert predictions to numpy
    predictions_np = predictions.cpu().numpy()
    actual_ratings_np = test_data.edge_attr.cpu().numpy()
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'user_id': test_data.edge_index[0].cpu().numpy(),
        'product_id': test_data.edge_index[1].cpu().numpy(),
        'actual_rating': actual_ratings_np,
        'predicted_rating': predictions_np
    })
    
    # Map back to original IDs
    results_df['userId'] = results_df['user_id'].map(reverse_user_map)
    results_df['productId'] = results_df['product_id'].map(reverse_product_map)
    
    # Calculate error
    results_df['error'] = results_df['predicted_rating'] - results_df['actual_rating']
    results_df['abs_error'] = np.abs(results_df['error'])
    
    # Sort by user_id and predicted_rating (descending)
    results_df = results_df.sort_values(['user_id', 'predicted_rating'], ascending=[True, False])
    
    return results_df

def generate_recommendations(model, train_data, num_nodes, top_k, reverse_user_map, reverse_product_map, batch_size=1000):
    """Generate top-k recommendations for each user."""
    print(f"Generating top-{top_k} recommendations for each user...")
    
    # Move data to device
    train_data = train_data.to(device)
    
    # Get the model's embedding size
    model_num_nodes = model.node_embedding.weight.shape[0]
    print(f"Model embedding size: {model_num_nodes}")
    
    # Get all users
    unique_users = torch.unique(train_data.edge_index[0]).cpu().numpy()
    
    # Filter users that are within the model's embedding size
    valid_users = unique_users[unique_users < model_num_nodes]
    if len(valid_users) < len(unique_users):
        print(f"Warning: Filtered out {len(unique_users) - len(valid_users)} users that are out of bounds for the model")
        unique_users = valid_users
    
    # Get all products
    all_products = torch.arange(len(reverse_product_map), dtype=torch.long) + len(reverse_user_map)
    
    # Filter products that are within the model's embedding size
    valid_products = all_products[all_products < model_num_nodes]
    if len(valid_products) < len(all_products):
        print(f"Warning: Filtered out {len(all_products) - len(valid_products)} products that are out of bounds for the model")
        all_products = valid_products
    
    # Verify product indices are within bounds
    max_product_idx = max(reverse_product_map.keys())
    min_product_idx = min(reverse_product_map.keys())
    print(f"Product index range in reverse_product_map: {min_product_idx} to {max_product_idx}")
    print(f"Number of users: {len(reverse_user_map)}")
    
    recommendations = []
    errors = 0
    
    # Process users in batches
    batch_size = min(batch_size, len(unique_users))
    num_batches = (len(unique_users) + batch_size - 1) // batch_size
    
    with Timer("Generating recommendations"):
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(unique_users))
            batch_users = unique_users[start_idx:end_idx]
            
            # For each user in the batch
            for user_id in batch_users:
                try:
                    # Skip if user_id is out of bounds for the model
                    if user_id >= model_num_nodes:
                        continue
                        
                    # Get products this user has already rated
                    user_rated_mask = (train_data.edge_index[0] == user_id)
                    user_rated_products = train_data.edge_index[1][user_rated_mask].cpu()
                    
                    # Get products this user hasn't rated
                    unrated_products_mask = torch.ones(len(all_products), dtype=torch.bool)
                    
                    # Ensure indices are within bounds
                    valid_indices = []
                    for idx in user_rated_products:
                        product_idx = idx.item() - len(reverse_user_map)
                        if 0 <= product_idx < len(reverse_product_map):
                            valid_indices.append(product_idx)
                    
                    if valid_indices:
                        unrated_products_mask[valid_indices] = False
                    
                    unrated_products = all_products[unrated_products_mask]
                    
                    if len(unrated_products) == 0:
                        continue  # Skip if user has rated all products
                    
                    # Create edge index for this user and all unrated products
                    user_nodes = torch.full((len(unrated_products),), user_id, device=device)
                    edge_index = torch.stack([user_nodes, unrated_products.to(device)])
                    
                    # Run inference
                    with torch.no_grad():
                        predictions = model(train_data.x, edge_index)
                    
                    # Get top-k recommendations
                    if len(predictions) > top_k:
                        _, indices = torch.topk(predictions, top_k)
                        top_products = unrated_products[indices.cpu()]
                        top_scores = predictions[indices].cpu().numpy()
                    else:
                        top_products = unrated_products
                        top_scores = predictions.cpu().numpy()
                    
                    # Map back to original IDs
                    original_user_id = reverse_user_map[user_id.item()]
                    
                    # Safely map product IDs
                    original_product_ids = []
                    for p in top_products:
                        product_idx = p.item() - len(reverse_user_map)
                        if product_idx in reverse_product_map:
                            original_product_ids.append(reverse_product_map[product_idx])
                        else:
                            # Skip this product if index is out of bounds
                            continue
                    
                    # Add to recommendations
                    for i, (product_id, score) in enumerate(zip(original_product_ids, top_scores[:len(original_product_ids)])):
                        recommendations.append({
                            'userId': original_user_id,
                            'productId': product_id,
                            'predicted_rating': score,
                            'rank': i + 1
                        })
                except Exception as e:
                    errors += 1
                    if errors <= 5:  # Only print the first 5 errors to avoid flooding the console
                        print(f"Error processing user {user_id}: {e}")
    
    print(f"Completed with {errors} errors")
    
    # Create DataFrame
    recommendations_df = pd.DataFrame(recommendations)
    
    return recommendations_df

def visualize_results(results_df, output_path):
    """Visualize the inference results."""
    print("Visualizing results...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # 1. Scatter plot of actual vs predicted ratings
    plt.figure(figsize=(10, 8))
    plt.scatter(results_df['actual_rating'], results_df['predicted_rating'], alpha=0.3)
    plt.plot([1, 5], [1, 5], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('Actual vs Predicted Ratings')
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'actual_vs_predicted.png'))
    plt.close()
    
    # 2. Histogram of errors
    plt.figure(figsize=(10, 8))
    plt.hist(results_df['error'], bins=50, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'error_distribution.png'))
    plt.close()
    
    # 3. Error by rating value
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='actual_rating', y='error', data=results_df)
    plt.xlabel('Actual Rating')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error by Rating Value')
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'error_by_rating.png'))
    plt.close()
    
    # 4. Calculate and display error metrics
    mse = np.mean(results_df['error'] ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(results_df['abs_error'])
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Mean Error': np.mean(results_df['error']),
        'Median Error': np.median(results_df['error']),
        'Std Dev Error': np.std(results_df['error'])
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_path, 'error_metrics.csv'), index=False)
    
    print("Visualization complete. Results saved to", output_path)
    return metrics

def save_mappings(user_map, product_map, output_path):
    """Save the mapping dictionaries for future reference."""
    print("Saving ID mappings...")
    
    # Convert dictionaries to DataFrames
    user_map_df = pd.DataFrame(list(user_map.items()), columns=['original_id', 'internal_id'])
    product_map_df = pd.DataFrame(list(product_map.items()), columns=['original_id', 'internal_id'])
    
    # Save to CSV
    user_map_df.to_csv(os.path.join(output_path, 'user_id_mapping.csv'), index=False)
    product_map_df.to_csv(os.path.join(output_path, 'product_id_mapping.csv'), index=False)
    
    print(f"Mappings saved to {output_path}")

def load_mappings(mapping_dir):
    """Load the mapping dictionaries from saved CSV files."""
    print("Loading ID mappings...")
    
    user_map_path = os.path.join(mapping_dir, 'user_id_mapping.csv')
    product_map_path = os.path.join(mapping_dir, 'product_id_mapping.csv')
    
    if not os.path.exists(user_map_path) or not os.path.exists(product_map_path):
        print("Warning: Mapping files not found. Please run inference first to generate mappings.")
        return None, None, None, None
    
    # Load DataFrames
    user_map_df = pd.read_csv(user_map_path)
    product_map_df = pd.read_csv(product_map_path)
    
    # Convert to dictionaries
    user_map = dict(zip(user_map_df['original_id'], user_map_df['internal_id']))
    product_map = dict(zip(product_map_df['original_id'], product_map_df['internal_id']))
    
    # Create reverse mappings
    reverse_user_map = {v: k for k, v in user_map.items()}
    reverse_product_map = {v: k for k, v in product_map.items()}
    
    print(f"Loaded mappings: {len(user_map)} users and {len(product_map)} products")
    
    return user_map, product_map, reverse_user_map, reverse_product_map

def recommend_for_user(user_id, model_path, mapping_dir, top_k=10, data_path=None):
    """Generate recommendations for a specific user using the saved model and mappings.
    
    Args:
        user_id: Original user ID to generate recommendations for
        model_path: Path to the saved model
        mapping_dir: Directory containing the saved mappings
        top_k: Number of recommendations to generate
        data_path: Path to the original data (to identify already rated products)
        
    Returns:
        DataFrame containing the recommendations for the user
    """
    print(f"Generating recommendations for user {user_id}")
    
    # Load mappings
    user_map, product_map, reverse_user_map, reverse_product_map = load_mappings(mapping_dir)
    
    if user_id not in user_map:
        print(f"Error: User ID {user_id} not found in the mappings.")
        return None
    
    # Get internal user ID
    internal_user_id = user_map[user_id]
    print(f"Internal user ID: {internal_user_id}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = max(max(reverse_user_map.keys()), max(reverse_product_map.keys())) + 1
    model = load_model(model_path, num_nodes)
    
    # Get the model's embedding size
    model_num_nodes = model.node_embedding.weight.shape[0]
    
    # Check if user ID is within the model's embedding size
    if internal_user_id >= model_num_nodes:
        print(f"Error: User ID {user_id} (internal ID: {internal_user_id}) is out of bounds for the model (max: {model_num_nodes-1}).")
        print("This can happen when using a different sample_size than what was used for training.")
        print("Try using the same sample_size that was used for training, or retrain the model with the current sample_size.")
        return None
    
    # Get already rated products if data_path is provided
    rated_products = set()
    if data_path:
        df = pd.read_csv(data_path)
        user_ratings = df[df['user_id'] == user_id]
        for _, row in user_ratings.iterrows():
            if row['product_id'] in product_map:
                rated_products.add(product_map[row['product_id']])
        print(f"User has already rated {len(rated_products)} products")
    
    # Get all product IDs
    all_product_ids = set(product_map.values())
    
    # Filter products that are within the model's embedding size
    valid_product_ids = {pid for pid in all_product_ids if pid < model_num_nodes}
    if len(valid_product_ids) < len(all_product_ids):
        print(f"Warning: Filtered out {len(all_product_ids) - len(valid_product_ids)} products that are out of bounds for the model")
    
    # Get unrated products
    unrated_products = list(valid_product_ids - rated_products)
    print(f"Found {len(unrated_products)} unrated products")
    
    # Prepare data for prediction
    model.eval()
    predictions = []
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    num_batches = (len(unrated_products) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(unrated_products))
            batch_products = unrated_products[start_idx:end_idx]
            
            # Create edges for this batch
            batch_edges = []
            for product_id in batch_products:
                if product_id < model_num_nodes:  # Ensure product_id is valid
                    batch_edges.append([internal_user_id, product_id])
            
            if not batch_edges:
                continue
                
            edge_index = torch.tensor(batch_edges, dtype=torch.long).t().to(device)
            
            # Create node indices for all nodes (0 to model_num_nodes-1)
            node_indices = torch.arange(model_num_nodes, dtype=torch.long).to(device)
            
            # No edge attributes for prediction
            edge_attr = torch.ones(edge_index.size(1), 1).to(device)
            
            # Run prediction
            pred = model(node_indices, edge_index).squeeze().cpu().numpy()
            
            # Store predictions
            for idx, product_id in enumerate(batch_products):
                if idx < len(pred) and product_id < model_num_nodes:  # Ensure indices are valid
                    try:
                        original_product_id = reverse_product_map[product_id]
                        predictions.append((original_product_id, float(pred[idx])))
                    except KeyError:
                        # Skip if product_id is not in reverse_product_map
                        continue
    
    # Sort predictions by predicted rating (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top-k recommendations
    top_recommendations = predictions[:top_k]
    
    # Create DataFrame
    recommendations_df = pd.DataFrame(top_recommendations, columns=['product_id', 'predicted_rating'])
    recommendations_df['user_id'] = user_id
    
    # Reorder columns
    recommendations_df = recommendations_df[['user_id', 'product_id', 'predicted_rating']]
    
    return recommendations_df

def evaluate_for_user(user_id, model_path, mapping_dir, data_path):
    """Evaluate the model on a specific user's ratings.
    
    Args:
        user_id: Original user ID to evaluate
        model_path: Path to the saved model
        mapping_dir: Directory containing the saved mappings
        data_path: Path to the original data
        
    Returns:
        DataFrame containing the evaluation results for the user
    """
    print(f"Evaluating model for user {user_id}")
    
    # Load mappings
    user_map, product_map, reverse_user_map, reverse_product_map = load_mappings(mapping_dir)
    
    if user_id not in user_map:
        print(f"Error: User ID {user_id} not found in the mappings.")
        return None, None
    
    # Get internal user ID
    internal_user_id = user_map[user_id]
    print(f"Internal user ID: {internal_user_id}")
    
    # Load data
    df = pd.read_csv(data_path)
    user_ratings = df[df['user_id'] == user_id]
    
    if user_ratings.empty:
        print(f"Error: No ratings found for user {user_id}.")
        return None, None
    
    print(f"Found {len(user_ratings)} ratings for user {user_id}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = max(max(reverse_user_map.keys()), max(reverse_product_map.keys())) + 1
    model = load_model(model_path, num_nodes)
    
    # Get the model's embedding size
    model_num_nodes = model.node_embedding.weight.shape[0]
    
    # Check if user ID is within the model's embedding size
    if internal_user_id >= model_num_nodes:
        print(f"Error: User ID {user_id} (internal ID: {internal_user_id}) is out of bounds for the model (max: {model_num_nodes-1}).")
        print("This can happen when using a different sample_size than what was used for training.")
        print("Try using the same sample_size that was used for training, or retrain the model with the current sample_size.")
        return None, None
    
    # Prepare data for prediction
    model.eval()
    results = []
    
    # Process in batches to avoid memory issues
    batch_size = 100
    num_batches = (len(user_ratings) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(user_ratings))
            batch_ratings = user_ratings.iloc[start_idx:end_idx]
            
            # Create edges for this batch
            batch_edges = []
            batch_ratings_list = []
            batch_product_ids = []
            
            for _, row in batch_ratings.iterrows():
                product_id = row['product_id']
                rating = row['rating']
                
                if product_id in product_map:
                    internal_product_id = product_map[product_id]
                    if internal_product_id < model_num_nodes:  # Ensure product_id is valid
                        batch_edges.append([internal_user_id, internal_product_id])
                        batch_ratings_list.append(rating)
                        batch_product_ids.append(product_id)
            
            if not batch_edges:
                continue
                
            edge_index = torch.tensor(batch_edges, dtype=torch.long).t().to(device)
            
            # Create node indices for all nodes (0 to model_num_nodes-1)
            node_indices = torch.arange(model_num_nodes, dtype=torch.long).to(device)
            
            # No edge attributes for prediction
            edge_attr = torch.ones(edge_index.size(1), 1).to(device)
            
            # Run prediction
            pred = model(node_indices, edge_index).squeeze().cpu().numpy()
            
            # Store results
            for idx, (product_id, actual_rating) in enumerate(zip(batch_product_ids, batch_ratings_list)):
                if idx < len(pred):
                    predicted_rating = float(pred[idx])
                    error = predicted_rating - actual_rating
                    results.append({
                        'user_id': user_id,
                        'product_id': product_id,
                        'actual_rating': actual_rating,
                        'predicted_rating': predicted_rating,
                        'error': error,
                        'abs_error': abs(error)
                    })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    if not results_df.empty:
        mse = mean_squared_error(results_df['actual_rating'], results_df['predicted_rating'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(results_df['actual_rating'], results_df['predicted_rating'])
        
        print(f"\nEvaluation metrics for user {user_id}:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # Add metrics to the results
        metrics = {
            'user_id': user_id,
            'num_ratings': len(results_df),
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame([metrics])
        
        return results_df, metrics_df
    
    return None, None

def main():
    """Main function to run the inference pipeline."""
    print("Starting GNN Recommendation System Inference")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load and preprocess data
    train_data, test_data, test_df, num_nodes, reverse_user_map, reverse_product_map = load_and_preprocess_data(
        args.data_path, sample_size=args.sample_size
    )
    
    # Save mappings for future reference
    user_map = {v: k for k, v in reverse_user_map.items()}
    product_map = {v: k for k, v in reverse_product_map.items()}
    save_mappings(user_map, product_map, args.output_path)
    
    # Load model
    model = load_model(args.model_path, num_nodes)
    
    # Get the model's embedding size
    model_num_nodes = model.node_embedding.weight.shape[0]
    
    # Create node indices for all nodes (0 to model_num_nodes-1)
    node_indices = torch.arange(model_num_nodes, dtype=torch.long).to(device)
    
    # Update train_data and test_data to use the correct node indices
    train_data.x = node_indices
    test_data.x = node_indices
    
    # Run inference on test data
    results_df = run_inference(model, test_data, test_df, reverse_user_map, reverse_product_map)
    
    # Save results
    results_df.to_csv(os.path.join(args.output_path, 'inference_results.csv'), index=False)
    print(f"Inference results saved to {os.path.join(args.output_path, 'inference_results.csv')}")
    
    # Display sample of results
    print("\nSample of inference results:")
    print(results_df.head(10))
    
    # Visualize results
    metrics = visualize_results(results_df, args.output_path)
    print("\nError metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate recommendations
    recommendations_df = generate_recommendations(
        model, train_data, num_nodes, args.top_k, reverse_user_map, reverse_product_map, batch_size=args.batch_size
    )
    
    # Save recommendations
    recommendations_df.to_csv(os.path.join(args.output_path, 'recommendations.csv'), index=False)
    print(f"Recommendations saved to {os.path.join(args.output_path, 'recommendations.csv')}")
    
    # Display sample of recommendations
    print("\nSample of recommendations:")
    print(recommendations_df.head(10))
    
    print("\nInference complete!")

if __name__ == "__main__":
    # Print help message
    print("""
GNN Recommendation System Inference Script
-----------------------------------------
This script provides several functionalities:
1. Full inference pipeline: Loads the model, runs inference on test data, and generates recommendations
2. Single user recommendations: Generates recommendations for a specific user
3. Single user evaluation: Evaluates the model on a specific user's ratings

Usage examples:
- Full pipeline: python rs_gnn_inference.py --model_path "models/final_model.pt" --data_path "data/ratings.csv" --output_path "inference"
- Single user recommendations: python rs_gnn_inference.py --model_path "models/final_model.pt" --data_path "data/ratings.csv" --output_path "inference" --user_id 12345
- Single user evaluation: python rs_gnn_inference.py --model_path "models/final_model.pt" --data_path "data/ratings.csv" --output_path "inference" --evaluate_user 12345
""")

    # parser = argparse.ArgumentParser(description='GNN Recommendation System Inference')
    # parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    # parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    # parser.add_argument('--output_path', type=str, default='inference', help='Directory to save results')
    # parser.add_argument('--sample_size', type=float, default=1.0, help='Fraction of data to use for testing')
    # parser.add_argument('--top_k', type=int, default=10, help='Number of recommendations to generate per user')
    # parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing users')
    # parser.add_argument('--user_id', type=int, default=None, help='Specific user ID to generate recommendations for')
    # parser.add_argument('--evaluate_user', type=int, default=None, help='Specific user ID to evaluate the model on')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # If evaluate_user is provided, evaluate the model on that user only
    if args.evaluate_user is not None:
        results_df, metrics_df = evaluate_for_user(
            args.evaluate_user, args.model_path, args.output_path, args.data_path
        )
        if results_df is not None:
            output_file = os.path.join(args.output_path, f'evaluation_user_{args.evaluate_user}.csv')
            results_df.to_csv(output_file, index=False)
            print(f"Evaluation results for user {args.evaluate_user} saved to {output_file}")
            
            metrics_file = os.path.join(args.output_path, f'metrics_user_{args.evaluate_user}.csv')
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Evaluation metrics for user {args.evaluate_user} saved to {metrics_file}")
            
            # Create visualizations for this user
            plt.figure(figsize=(12, 8))
            
            # Scatter plot of actual vs predicted ratings
            plt.subplot(2, 2, 1)
            plt.scatter(results_df['actual_rating'], results_df['predicted_rating'], alpha=0.5)
            plt.plot([1, 5], [1, 5], 'r--')  # Diagonal line for perfect predictions
            plt.xlabel('Actual Rating')
            plt.ylabel('Predicted Rating')
            plt.title('Actual vs Predicted Ratings')
            plt.grid(True)
            
            # Histogram of errors
            plt.subplot(2, 2, 2)
            plt.hist(results_df['error'], bins=20, alpha=0.7)
            plt.xlabel('Error (Predicted - Actual)')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.grid(True)
            
            # Box plot of absolute errors
            plt.subplot(2, 2, 3)
            plt.boxplot(results_df['abs_error'])
            plt.ylabel('Absolute Error')
            plt.title('Absolute Error Distribution')
            plt.grid(True)
            
            # Error vs actual rating
            plt.subplot(2, 2, 4)
            plt.scatter(results_df['actual_rating'], results_df['error'], alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Actual Rating')
            plt.ylabel('Error (Predicted - Actual)')
            plt.title('Error vs Actual Rating')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_path, f'evaluation_plots_user_{args.evaluate_user}.png'))
            print(f"Evaluation plots saved to {os.path.join(args.output_path, f'evaluation_plots_user_{args.evaluate_user}.png')}")
    elif args.user_id is not None:
        recommendations_df = recommend_for_user(
            args.user_id, args.model_path, args.output_path, args.top_k, args.data_path
        )
        if recommendations_df is not None:
            output_file = os.path.join(args.output_path, f'recommendations_user_{args.user_id}.csv')
            recommendations_df.to_csv(output_file, index=False)
            print(f"Recommendations for user {args.user_id} saved to {output_file}")
            print("\nTop recommendations:")
            print(recommendations_df)
    else:
        # Run the full inference pipeline
        main()
