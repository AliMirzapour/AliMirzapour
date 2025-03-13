# GNN-based Recommendation System

This repository contains a Graph Neural Network (GNN) based recommendation system for predicting user-product ratings and generating personalized product recommendations.

## Features

- **Graph-based Approach**: Leverages the power of Graph Neural Networks to capture complex relationships between users and products.
- **Personalized Recommendations**: Generates personalized top-K product recommendations for users.
- **Evaluation Metrics**: Provides detailed evaluation metrics including MSE, RMSE, and MAE.
- **Visualization**: Creates visualizations to help understand model performance.
- **Memory Efficiency**: Processes data in batches to handle large datasets efficiently.
- **User-specific Analysis**: Supports evaluation and recommendation generation for specific users.

## Requirements

- Python 3.7+
- PyTorch
- PyTorch Geometric
- pandas
- numpy
- matplotlib
- scikit-learn
- tqdm

## Usage

### Training

To train the GNN recommendation model:

```bash
python rs_gnn_train.py --data_path "data/ratings.csv" --output_path "models" --epochs 100 --batch_size 1024 --hidden_channels 64 --learning_rate 0.01
```

### Inference

The inference script (`rs_gnn_inference.py`) provides several functionalities:

1. **Full Inference Pipeline**: Loads the model, runs inference on test data, and generates recommendations for all users.
2. **Single User Recommendations**: Generates recommendations for a specific user.
3. **Single User Evaluation**: Evaluates the model on a specific user's ratings.

#### Full Pipeline

```bash
python rs_gnn_inference.py --model_path "models/final_model.pt" --data_path "data/ratings.csv" --output_path "inference"
```

#### Single User Recommendations

```bash
python rs_gnn_inference.py --model_path "models/final_model.pt" --data_path "data/ratings.csv" --output_path "inference" --user_id 12345
```

#### Single User Evaluation

```bash
python rs_gnn_inference.py --model_path "models/final_model.pt" --data_path "data/ratings.csv" --output_path "inference" --evaluate_user 12345
```

### Parameters

- `--model_path`: Path to the trained model (required)
- `--data_path`: Path to the dataset (required)
- `--output_path`: Directory to save results (default: "inference")
- `--sample_size`: Fraction of data to use for testing (default: 1.0)
- `--top_k`: Number of recommendations to generate per user (default: 10)
- `--batch_size`: Batch size for processing users (default: 100)
- `--user_id`: Specific user ID to generate recommendations for
- `--evaluate_user`: Specific user ID to evaluate the model on

### Important Note on Sample Size

The `sample_size` parameter determines what fraction of the dataset to use for inference. It's important to note:

1. **Consistency with Training**: Ideally, you should use the same `sample_size` for inference as was used during training. Using a different `sample_size` can lead to mismatches in the node embeddings.

2. **Handling Mismatches**: If you need to use a different `sample_size`, the script includes logic to handle mismatches between the model's embedding size and the current dataset. It will:
   - Filter out nodes that are out of bounds for the model
   - Provide warnings about filtered nodes
   - Ensure that only valid nodes are used for inference

3. **Best Practices**: For the most accurate results:
   - Use the same `sample_size` for both training and inference
   - If you need to use a smaller `sample_size` for memory reasons, consider retraining the model with that sample size

## Output Files

The inference script generates several output files:

- `inference_results.csv`: Contains the inference results including actual and predicted ratings.
- `recommendations.csv`: Contains the top-K recommendations for each user.
- `user_id_mapping.csv` and `product_id_mapping.csv`: Mapping between original and internal IDs.
- Various visualization files (PNG) showing model performance.
- For single user evaluation/recommendation, user-specific files are generated.

## Troubleshooting

### Model Loading Issues

If you encounter a `pickle` error when loading the model, it may be due to changes in PyTorch 2.6 regarding the `weights_only` parameter. The script includes error handling to address this issue.

### Node Embedding Size Mismatch

If you see errors about node embedding size mismatches (e.g., "size mismatch for node_embedding.weight"), this is likely due to using a different `sample_size` than what was used during training. Solutions:

1. Use the same `sample_size` that was used during training
2. The script will automatically handle mismatches by filtering out nodes that are out of bounds
3. For best results, consider retraining the model with your desired `sample_size`

## Model Architecture

The GNN model uses a Graph Convolutional Network (GCN) architecture to learn embeddings for users and products. The model consists of:

1. GCN layers to process the user-product interaction graph
2. Linear layers to predict ratings based on the learned embeddings
3. Regularization techniques to prevent overfitting

## Performance

The model's performance is evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

Visualizations are generated to help understand the model's performance, including:

- Scatter plots of actual vs. predicted ratings
- Histograms of prediction errors
- Box plots of absolute errors

## Future Improvements

- Incorporate product features and user demographics
- Implement more advanced GNN architectures
- Add support for implicit feedback
- Implement time-aware recommendations
- Add support for cold-start users and products 