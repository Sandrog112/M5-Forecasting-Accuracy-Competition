import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import gc
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Setup logging
def setup_logging():
    """
    Sets up logging configuration to save logs to log folder.
    Creates log folder if it doesn't exist.
    """
    log_folder = "logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    log_file = os.path.join(log_folder, "inference.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_models(models_dir="models"):
    """
    Load trained LightGBM models from directory.
    
    Args:
        models_dir: Directory containing saved models
        
    Returns:
        list: List of loaded LightGBM models
    """
    models = []
    model_files = [f for f in os.listdir(models_dir) if f.startswith("lgb_model_fold_") and f.endswith(".txt")]
    model_files.sort()  # Sort to ensure fold order
    
    logging.info(f"Loading {len(model_files)} models from {models_dir}...")
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model = lgb.Booster(model_file=model_path)
        models.append(model)
        logging.info(f"Loaded model from {model_path}")
        
    return models

def prepare_test_data(data_path="data/processed_data.pkl"):
    """
    Prepare test data for model evaluation.
    
    Args:
        data_path: Path to processed data
        
    Returns:
        tuple: X_test features and y_test values
    """
    logging.info("Loading processed data for testing...")
    data = pd.read_pickle(data_path)
    
    # Features used during training
    features = [
        # main features
        "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI", "sell_price",
        # demand features
        "shift_t28", "shift_t29", "shift_t30",
        # std
        "rolling_std_t7", "rolling_std_t30", "rolling_std_t60", "rolling_std_t90", "rolling_std_t180",
        # mean
        "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60", "rolling_mean_t90", "rolling_mean_t180",
        # min
        "rolling_min_t7", "rolling_min_t30", "rolling_min_t60",
        # max
        "rolling_max_t7", "rolling_max_t30", "rolling_max_t60",
        # dist features
        "rolling_skew_t30", "rolling_kurt_t30",
        # price features
        "price_change_t1", "price_change_t365",
        "rolling_price_std_t7", "rolling_price_std_t30",
        # time features
        "year", "quarter", "month", "week", "day", "dayofweek", "is_weekend",
    ]
    
    # Use validation data (from 1914 to 1941, which is 28 days)
    is_val = (data["d"] >= 1914) & (data["d"] < 1942) & (data["part"] == "validation")
    
    X_test = data[is_val][features].reset_index(drop=True)
    y_test = data[is_val]["demand"].reset_index(drop=True)
    
    logging.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    return X_test, y_test

def evaluate_models(models, X_test, y_test):
    """
    Evaluate each model and calculate metrics.
    
    Args:
        models: List of trained models
        X_test: Test features
        y_test: True values
        
    Returns:
        tuple: Predictions for each model and evaluation metrics
    """
    all_preds = []
    all_metrics = []
    
    # Log statistics about the target variable
    logging.info(f"Target variable stats: min={y_test.min():.2f}, max={y_test.max():.2f}, "
                f"mean={y_test.mean():.2f}, std={y_test.std():.2f}")
    
    for i, model in enumerate(models):
        logging.info(f"Evaluating model for fold {i+1}...")
        
        # Make predictions
        preds = model.predict(X_test)
        all_preds.append(preds)
        
        # Log prediction statistics
        logging.info(f"Predictions stats: min={preds.min():.2f}, max={preds.max():.2f}, "
                    f"mean={preds.mean():.2f}, std={preds.std():.2f}")
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        metrics = {
            'Fold': i+1,
            'MAE': mae,
            'RMSE': rmse
        }
        all_metrics.append(metrics)
        
        logging.info(f"Fold {i+1} metrics: MAE={mae:.4f}, RMSE={rmse:.4f}")
        
    # Calculate average metrics
    metrics_df = pd.DataFrame(all_metrics)
    avg_metrics = metrics_df[['MAE', 'RMSE']].mean().to_dict()
    avg_metrics['Fold'] = 'Average'
    all_metrics.append(avg_metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Calculate ensemble (average) predictions
    ensemble_preds = np.mean(all_preds, axis=0)
    
    logging.info(f"Ensemble metrics: MAE={mean_absolute_error(y_test, ensemble_preds):.4f}, "
                f"RMSE={np.sqrt(mean_squared_error(y_test, ensemble_preds)):.4f}")
    
    return metrics_df

def plot_metrics_table(metrics_df):
    """
    Create a table plot of evaluation metrics.
    
    Args:
        metrics_df: DataFrame containing metrics for each fold
    """
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    plt.figure(figsize=(8, 4))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    
    # Format the data for the table - handle strings properly
    table_data = metrics_df.copy()
    for col in table_data.columns:
        if col != 'Fold':  # Skip the Fold column
            # Format numeric columns to 4 decimal places
            table_data[col] = table_data[col].apply(lambda x: f"{x:.4f}")
    
    # Create table with formatted values
    table = plt.table(
        cellText=table_data.values,
        colLabels=metrics_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0.2, 0.2, 0.6, 0.6]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title('M5 Forecasting Model Evaluation Metrics', fontsize=16, pad=20)
    
    # Save the plot
    metrics_table_path = os.path.join(outputs_dir, 'metrics_table.png')
    plt.savefig(metrics_table_path, bbox_inches='tight', dpi=200)
    logging.info(f"Metrics table saved to {metrics_table_path}")
    plt.close()

def evaluate_and_visualize():
    """
    Main function to evaluate models and create visualizations.
    """
    # Setup logging
    setup_logging()
    logging.info("Starting model evaluation...")
    
    # Load models
    models = load_models()
    
    # Prepare test data
    X_test, y_test = prepare_test_data()
    
    # Evaluate models
    metrics_df = evaluate_models(models, X_test, y_test)
    
    # Plot metrics table
    plot_metrics_table(metrics_df)
    
    logging.info("Model evaluation complete!")
    
    return metrics_df

if __name__ == "__main__":
    evaluate_and_visualize()