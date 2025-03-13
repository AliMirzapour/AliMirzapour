#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized GNN-based Recommendation System Training Script
Designed for large datasets with GPU acceleration
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import train_test_split_edges
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output

# Check for GPU availability
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train GNN Recommendation System')
parser.add_argument('--data_path', type=str, default='../ratings_Electronics (1).csv',
                    help='Path to the CSV dataset')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=64, help='Hidden channels in GNN')
parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of node embeddings')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2 regularization strength')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN', 'SAGE', 'GAT'],
                    help='Type of GNN model to use')
parser.add_argument('--sample_size', type=float, default=1.0, 
                    help='Fraction of data to use (1.0 means all data)')
parser.add_argument('--save_path', type=str, default='Recommendation System/models',
                    help='Path to save the trained model')
parser.add_argument('--cross_validation', action='store_true',
                    help='Whether to use k-fold cross-validation')
parser.add_argument('--k_folds', type=int, default=5,
                    help='Number of folds for cross-validation')
args = parser.parse_args()

# Create save directory if it doesn't exist
os.makedirs(args.save_path, exist_ok=True)

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

# Define the GNN model
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
        
        # Apply L2 regularization
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2)
        
        return ratings, l2_loss * self.l2_reg

class LossTracker:
    """Class to track and visualize training and validation loss."""
    def __init__(self, save_path):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.save_path = save_path
        
        # Set up the plot
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.train_line, = self.ax.plot([], [], 'b-', label='Training Loss')
        self.val_line, = self.ax.plot([], [], 'r-', label='Validation Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training and Validation Loss')
        self.ax.legend()
        self.ax.grid(True)
        
    def update(self, epoch, train_loss, val_loss):
        """Update the loss tracker with new values."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Update the plot
        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)
        
        # Adjust the plot limits
        self.ax.set_xlim(0, max(self.epochs) + 1)
        self.ax.set_ylim(0, max(max(self.train_losses), max(self.val_losses)) * 1.1)
        
        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Save the plot
        plt.savefig(os.path.join(self.save_path, 'loss_plot.png'))
        
    def save_final_plot(self):
        """Save the final loss plot."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, 'final_loss_plot.png'))
        plt.close()
        
        # Also save the loss values to a CSV file
        loss_df = pd.DataFrame({
            'Epoch': self.epochs,
            'Train_Loss': self.train_losses,
            'Val_Loss': self.val_losses
        })
        loss_df.to_csv(os.path.join(self.save_path, 'loss_history.csv'), index=False)
        
        print(f"Loss plot and history saved to {self.save_path}")

def load_and_preprocess_data(file_path, sample_size=1.0):
    """Load and preprocess the dataset efficiently."""
    print(f"Loading data from {file_path}...")
    
    with Timer("Data loading"):
        # Use chunking for large files
        if sample_size < 1.0:
            # Read a sample of the data
            df = pd.read_csv(file_path, header = None, names=['userId', 'productId', 'rating', 'timestamp'])
            df = df.sample(frac=sample_size, random_state=42)
        else:
            # Read the entire file
            df = pd.read_csv(file_path, header = None, names=['userId', 'productId', 'rating', 'timestamp'])
    
    print(f"Loaded {len(df)} ratings")
    
    with Timer("Data preprocessing"):
        # Map user and product IDs to consecutive integers
        unique_users = df['userId'].unique()
        unique_products = df['productId'].unique()
        
        user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        product_id_map = {old_id: new_id + len(user_id_map) for new_id, old_id in enumerate(unique_products)}
        
        # Apply mapping
        df['user_id'] = df['userId'].map(user_id_map)
        df['product_id'] = df['productId'].map(product_id_map)
        
        # Split data into train, validation, and test sets
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        # Get total number of nodes (users + products)
        num_nodes = len(user_id_map) + len(product_id_map)

        # Create edge indices and attributes for PyTorch Geometric
        train_edge_index = torch.tensor(np.array([train_df['user_id'].values, train_df['product_id'].values]), dtype=torch.long)
        train_edge_attr = torch.tensor(train_df['rating'].values, dtype=torch.float)

        val_edge_index = torch.tensor(np.array([val_df['user_id'].values, val_df['product_id'].values]), dtype=torch.long)
        val_edge_attr = torch.tensor(val_df['rating'].values, dtype=torch.float)

        test_edge_index = torch.tensor(np.array([test_df['user_id'].values, test_df['product_id'].values]), dtype=torch.long)
        test_edge_attr = torch.tensor(test_df['rating'].values, dtype=torch.float)
        
        # Create node indices for all nodes (0 to num_nodes-1)
        # This is much more memory efficient than creating a full identity matrix
        node_indices = torch.arange(num_nodes, dtype=torch.long)
       
        # Create PyTorch Geometric Data objects
        train_data = Data(x=node_indices, edge_index=train_edge_index, edge_attr=train_edge_attr)
        val_data = Data(x=node_indices, edge_index=val_edge_index, edge_attr=val_edge_attr)
        test_data = Data(x=node_indices, edge_index=test_edge_index, edge_attr=test_edge_attr)
        print(f"train_data, val_data, test_data is {train_data}, {val_data}, {test_data}")
        return train_data, val_data, test_data, num_nodes

def train_model(model, train_data, val_data, optimizer, epochs, batch_size):
    """Train the GNN model with early stopping."""
    model.train()
    
    # Move data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience = args.patience
    patience_counter = 0
    
    # Initialize loss tracker
    loss_tracker = LossTracker(args.save_path)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    for epoch in range(epochs):
        with Timer(f"Epoch {epoch+1}/{epochs}"):
            # Training
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            train_out, l2_loss = model(train_data.x, train_data.edge_index)
            
            # Calculate loss with L2 regularization
            train_loss = F.mse_loss(train_out, train_data.edge_attr) + l2_loss
            
            # Backward pass and optimization
            train_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out, _ = model(val_data.x, val_data.edge_index)
                val_loss = F.mse_loss(val_out, val_data.edge_attr)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Update loss tracker
            loss_tracker.update(epoch + 1, train_loss.item() - l2_loss.item(), val_loss.item())
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item() - l2_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, L2 Loss: {l2_loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pt'))
                print(f"New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    # Save the final loss plot
    loss_tracker.save_final_plot()
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'best_model.pt')))
    return model

def evaluate_model(model, data):
    """Evaluate the model on the given data."""
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        
        # Calculate metrics
        mse = F.mse_loss(out, data.edge_attr).item()
        rmse = np.sqrt(mse)
        mae = F.l1_loss(out, data.edge_attr).item()
        
        # Calculate additional metrics
        out_np = out.cpu().numpy()
        edge_attr_np = data.edge_attr.cpu().numpy()
        
        # Calculate R-squared
        ss_tot = np.sum((edge_attr_np - np.mean(edge_attr_np)) ** 2)
        ss_res = np.sum((edge_attr_np - out_np) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Visualize predictions vs actual ratings
        visualize_predictions(edge_attr_np, out_np, os.path.join(args.save_path, 'predictions.png'))
        
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def visualize_predictions(actual, predicted, save_path):
    """Visualize the model's predictions compared to actual ratings."""
    plt.figure(figsize=(12, 8))
    
    # Scatter plot of actual vs predicted
    plt.scatter(actual, predicted, alpha=0.3, color='blue')
    
    # Perfect prediction line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels and title
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Actual vs Predicted Ratings')
    plt.grid(True)
    
    # Add a histogram of errors
    errors = predicted - actual
    plt.figure(figsize=(12, 8))
    plt.hist(errors, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    plt.savefig(save_path.replace('.png', '_error_hist.png'))
    
    # Create a 2D histogram (heatmap) of actual vs predicted
    plt.figure(figsize=(12, 8))
    plt.hist2d(actual, predicted, bins=20, cmap='viridis')
    plt.colorbar(label='Count')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Heatmap of Actual vs Predicted Ratings')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.grid(True)
    plt.savefig(save_path.replace('.png', '_heatmap.png'))
    
    # Save the scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(actual, predicted, alpha=0.3, color='blue')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Actual vs Predicted Ratings')
    plt.grid(True)
    plt.savefig(save_path)
    
    # Close all figures
    plt.close('all')
    
    print(f"Prediction visualizations saved to {os.path.dirname(save_path)}")
    
    # Also save the predictions to a CSV file
    predictions_df = pd.DataFrame({
        'Actual': actual,
        'Predicted': predicted,
        'Error': errors
    })
    predictions_df.to_csv(save_path.replace('.png', '.csv'), index=False)
    
    # Calculate and print error statistics
    print(f"Error statistics:")
    print(f"  Mean error: {np.mean(errors):.4f}")
    print(f"  Median error: {np.median(errors):.4f}")
    print(f"  Std dev of error: {np.std(errors):.4f}")
    print(f"  Min error: {np.min(errors):.4f}")
    print(f"  Max error: {np.max(errors):.4f}")
    print(f"  25th percentile: {np.percentile(errors, 25):.4f}")
    print(f"  75th percentile: {np.percentile(errors, 75):.4f}")

def visualize_model_architecture(model, save_path):
    """Visualize the model architecture using torchviz if available."""
    try:
        from torchviz import make_dot
        
        # Create dummy data for visualization
        num_nodes = model.node_embedding.num_embeddings
        dummy_x = torch.arange(10, dtype=torch.long)  # Just use 10 nodes for visualization
        dummy_edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=torch.long)
        
        # Forward pass to get the output
        dummy_output, _ = model(dummy_x, dummy_edge_index)
        
        # Visualize the model architecture
        dot = make_dot(dummy_output, params=dict(model.named_parameters()))
        dot.format = 'png'
        dot.render(os.path.join(save_path, 'model_architecture'))
        
        print(f"Model architecture visualization saved to {save_path}")
    except ImportError:
        print("torchviz not available. Install it with 'pip install torchviz' to visualize the model architecture.")
    except Exception as e:
        print(f"Error visualizing model architecture: {e}")

def cross_validate(df, num_folds=5):
    """Perform k-fold cross-validation."""
    print(f"Performing {num_folds}-fold cross-validation...")
    
    # Create folds
    fold_size = len(df) // num_folds
    metrics = []
    
    for fold in range(num_folds):
        print(f"\n--- Fold {fold+1}/{num_folds} ---")
        
        # Create train and validation sets
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < num_folds - 1 else len(df)
        
        val_df = df.iloc[val_start:val_end].copy()
        train_df = pd.concat([df.iloc[:val_start], df.iloc[val_end:]]).copy()
        
        # Create test set (10% of training data)
        train_df, test_df = train_test_split(train_df, test_size=0.1, random_state=42)
        
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        # Map user and product IDs to consecutive integers
        unique_users = df['userId'].unique()
        unique_products = df['productId'].unique()
        
        user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        product_id_map = {old_id: new_id + len(user_id_map) for new_id, old_id in enumerate(unique_products)}
        
        # Apply mapping
        train_df['user_id'] = train_df['userId'].map(user_id_map)
        train_df['product_id'] = train_df['productId'].map(product_id_map)
        val_df['user_id'] = val_df['userId'].map(user_id_map)
        val_df['product_id'] = val_df['productId'].map(product_id_map)
        test_df['user_id'] = test_df['userId'].map(user_id_map)
        test_df['product_id'] = test_df['productId'].map(product_id_map)
        
        # Get total number of nodes (users + products)
        num_nodes = len(user_id_map) + len(product_id_map)
        
        # Create edge indices and attributes for PyTorch Geometric
        train_edge_index = torch.tensor(np.array([train_df['user_id'].values, train_df['product_id'].values]), dtype=torch.long)
        train_edge_attr = torch.tensor(train_df['rating'].values, dtype=torch.float)
        
        val_edge_index = torch.tensor(np.array([val_df['user_id'].values, val_df['product_id'].values]), dtype=torch.long)
        val_edge_attr = torch.tensor(val_df['rating'].values, dtype=torch.float)
        
        test_edge_index = torch.tensor(np.array([test_df['user_id'].values, test_df['product_id'].values]), dtype=torch.long)
        test_edge_attr = torch.tensor(test_df['rating'].values, dtype=torch.float)
        
        # Create node indices for all nodes (0 to num_nodes-1)
        node_indices = torch.arange(num_nodes, dtype=torch.long)
        
        # Create PyTorch Geometric Data objects
        train_data = Data(x=node_indices, edge_index=train_edge_index, edge_attr=train_edge_attr)
        val_data = Data(x=node_indices, edge_index=val_edge_index, edge_attr=val_edge_attr)
        test_data = Data(x=node_indices, edge_index=test_edge_index, edge_attr=test_edge_attr)
        
        # Create model
        fold_save_path = os.path.join(args.save_path, f"fold_{fold+1}")
        os.makedirs(fold_save_path, exist_ok=True)
        
        model = GNNRecommender(
            num_nodes=num_nodes,
            hidden_channels=args.hidden_channels,
            embedding_dim=args.embedding_dim,
            model_type=args.model_type,
            dropout=args.dropout,
            l2_reg=args.l2_reg
        ).to(device)
        
        # Create optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Train model
        model = train_model(
            model=model,
            train_data=train_data,
            val_data=val_data,
            optimizer=optimizer,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Evaluate model
        print(f"\nEvaluating model on test set for fold {fold+1}...")
        test_metrics = evaluate_model(model, test_data)
        
        for metric, value in test_metrics.items():
            print(f"Test {metric}: {value:.4f}")
        
        metrics.append(test_metrics)
        
        # Save model for this fold
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_metrics': test_metrics,
            'args': vars(args)
        }, os.path.join(fold_save_path, 'final_model.pt'))
    
    # Calculate average metrics across folds
    avg_metrics = {}
    for metric in metrics[0].keys():
        avg_metrics[metric] = np.mean([m[metric] for m in metrics])
        std_metrics = np.std([m[metric] for m in metrics])
        print(f"Average {metric} across folds: {avg_metrics[metric]:.4f} ± {std_metrics:.4f}")
    
    return avg_metrics

def main():
    """Main function to run the training pipeline."""
    print("Starting GNN Recommendation System Training")
    
    if args.cross_validation:
        # Load data for cross-validation
        print(f"Loading data from {args.data_path}...")
        with Timer("Data loading"):
            if args.sample_size < 1.0:
                df = pd.read_csv(args.data_path, header=None, names=['userId', 'productId', 'rating', 'timestamp'])
                df = df.sample(frac=args.sample_size, random_state=42)
            else:
                df = pd.read_csv(args.data_path, header=None, names=['userId', 'productId', 'rating', 'timestamp'])
        
        print(f"Loaded {len(df)} ratings")
        
        # Perform cross-validation
        avg_metrics = cross_validate(df, num_folds=args.k_folds)
        
        # Save average metrics
        metrics_df = pd.DataFrame([avg_metrics])
        metrics_df.to_csv(os.path.join(args.save_path, 'cross_validation_metrics.csv'), index=False)
        
        print(f"Cross-validation complete. Results saved to {args.save_path}")
    else:
        # Regular training
        train_data, val_data, test_data, num_nodes = load_and_preprocess_data(
            args.data_path, sample_size=args.sample_size
        )
        print(f"train_data is Done")
        
        # Create model
        model = GNNRecommender(
            num_nodes=num_nodes,
            hidden_channels=args.hidden_channels,
            embedding_dim=args.embedding_dim,
            model_type=args.model_type,
            dropout=args.dropout,
            l2_reg=args.l2_reg
        ).to(device)
        
        print(f"Model: {args.model_type} with {args.hidden_channels} hidden channels and {args.embedding_dim} embedding dimensions")
        print(f"Regularization: dropout={args.dropout}, L2={args.l2_reg}, weight_decay={args.weight_decay}")
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Visualize model architecture
        visualize_model_architecture(model, args.save_path)
        
        # Create optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Train model
        model = train_model(
            model=model,
            train_data=train_data,
            val_data=val_data,
            optimizer=optimizer,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
       
        # Evaluate model
        print("\nEvaluating model on test set...")
        test_metrics = evaluate_model(model, test_data)
        
        for metric, value in test_metrics.items():
            print(f"Test {metric}: {value:.4f}")
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_metrics': test_metrics,
            'args': vars(args)
        }, os.path.join(args.save_path, 'final_model.pt'))
        
        print(f"Model saved to {os.path.join(args.save_path, 'final_model.pt')}")

if __name__ == "__main__":
    main()
