import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import gc

DAYS_PRED = 28  # Prediction window for M5 competition

# Setup logging
def setup_logging():
    """
    Sets up logging configuration to save logs to log folder.
    Creates log folder if it doesn't exist.
    """
    log_folder = "logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    log_file = os.path.join(log_folder, "train.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# Custom time series splitter
class CustomTimeSeriesSplitter:
    """
    Custom cross-validation splitter that respects time series structure.
    Splits time series data into training and validation sets based on time.
    """
    def __init__(self, n_splits=5, train_days=80, test_days=20, day_col="d"):
        self.n_splits = n_splits
        self.train_days = train_days
        self.test_days = test_days
        self.day_col = day_col

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and validation set.
        
        Args:
            X: Features dataframe
            y: Target variable (not used)
            groups: Groups (not used)
            
        Yields:
            tuple: (train_index, val_index)
        """
        SEC_IN_DAY = 3600 * 24
        sec = (X[self.day_col].astype(np.int64) - X[self.day_col].iloc[0].astype(np.int64)) * SEC_IN_DAY
        duration = sec.max()

        train_sec = self.train_days * SEC_IN_DAY
        test_sec = self.test_days * SEC_IN_DAY
        total_sec = test_sec + train_sec

        if self.n_splits == 1:
            train_start = duration - total_sec
            train_end = train_start + train_sec

            train_mask = (sec >= train_start) & (sec < train_end)
            test_mask = sec >= train_end

            yield sec[train_mask].index.values, sec[test_mask].index.values

        else:
            step = DAYS_PRED * SEC_IN_DAY

            for idx in range(self.n_splits):
                shift = (self.n_splits - (idx + 1)) * step
                train_start = duration - total_sec - shift
                train_end = train_start + train_sec
                test_end = train_end + test_sec

                train_mask = (sec > train_start) & (sec <= train_end)

                if idx == self.n_splits - 1:
                    test_mask = sec > train_end
                else:
                    test_mask = (sec > train_end) & (sec <= test_end)

                yield sec[train_mask].index.values, sec[test_mask].index.values

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def show_cv_days(cv, X, dt_col, day_col):
    """
    Displays information about each CV fold's date range and days.
    
    Args:
        cv: Cross-validation splitter
        X: Features dataframe
        dt_col: Date column name
        day_col: Day number column name
    """
    for ii, (tr, tt) in enumerate(cv.split(X)):
        logging.info(f"----- Fold: ({ii + 1} / {cv.n_splits}) -----")
        tr_start = X.iloc[tr][dt_col].min()
        tr_end = X.iloc[tr][dt_col].max()
        tr_days = X.iloc[tr][day_col].max() - X.iloc[tr][day_col].min() + 1

        tt_start = X.iloc[tt][dt_col].min()
        tt_end = X.iloc[tt][dt_col].max()
        tt_days = X.iloc[tt][day_col].max() - X.iloc[tt][day_col].min() + 1

        fold_info = pd.DataFrame(
            {
                "start": [tr_start, tt_start],
                "end": [tr_end, tt_end],
                "days": [tr_days, tt_days],
            },
            index=["train", "test"],
        )
        
        logging.info(f"\n{fold_info}")

def plot_cv_indices(cv, X, dt_col, lw=10):
    """
    Plots the cross-validation folds to visualize the data splits.
    Saves the plot to the outputs folder.
    
    Args:
        cv: Cross-validation splitter
        X: Features dataframe
        dt_col: Date column name
        lw: Line width for plotting
        
    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    # Create outputs directory if it doesn't exist
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        
    n_splits = cv.get_n_splits()
    fig, ax = plt.subplots(figsize=(20, n_splits))

    for ii, (tr, tt) in enumerate(cv.split(X)):
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        ax.scatter(
            X[dt_col],
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=plt.cm.coolwarm,
            vmin=-0.2,
            vmax=1.2,
        )

    MIDDLE = 15
    LARGE = 20
    ax.set_xlabel("Datetime", fontsize=LARGE)
    ax.set_xlim([X[dt_col].min(), X[dt_col].max()])
    ax.set_ylabel("CV iteration", fontsize=LARGE)
    ax.set_yticks(np.arange(n_splits) + 0.5)
    ax.set_yticklabels(list(range(n_splits)))
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=MIDDLE)
    ax.set_title("{}".format(type(cv).__name__), fontsize=LARGE)
    
    # Save the plot
    plot_path = os.path.join(outputs_dir, "cv_splits.png")
    fig.savefig(plot_path, bbox_inches='tight')
    logging.info(f"Cross-validation split plot saved to {plot_path}")
    
    return ax


def train_lgb(bst_params, fit_params, X, y, cv, drop_when_train=None):
    """
    Trains LightGBM models using time-series cross-validation.
    
    Args:
        bst_params: LightGBM model parameters
        fit_params: LightGBM training parameters
        X: Features dataframe
        y: Target series
        cv: Cross-validation splitter
        drop_when_train: Columns to drop during training
        
    Returns:
        list: Trained LightGBM models
    """
    models = []

    if drop_when_train is None:
        drop_when_train = []

    for idx_fold, (idx_trn, idx_val) in enumerate(cv.split(X, y)):
        logging.info(f"\n----- Fold: ({idx_fold + 1} / {cv.get_n_splits()}) -----\n")

        X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
        y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]
        train_set = lgb.Dataset(
            X_trn.drop(drop_when_train, axis=1),
            label=y_trn,
            categorical_feature=["item_id"],
        )
        val_set = lgb.Dataset(
            X_val.drop(drop_when_train, axis=1),
            label=y_val,
            categorical_feature=["item_id"],
        )

        logging.info(f"Training model for fold {idx_fold + 1}...")
        model = lgb.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            num_boost_round=fit_params["num_boost_round"],  
            callbacks=[
                lgb.early_stopping(stopping_rounds=fit_params["early_stopping_rounds"]),
                lgb.log_evaluation(fit_params["verbose_eval"]),
            ],  
        )
        models.append(model)
        logging.info(f"Fold {idx_fold + 1} training completed")

        # Clean up memory
        gc.collect()

    return models

def train_model(data_path="data/processed_data.pkl", save_models=True):
    """
    Main function for training the forecasting model.
    
    Args:
        data_path: Path to the processed data pickle file
        save_models: Whether to save trained models to disk
        
    Returns:
        list: Trained models
    """
    logging.info("Loading processed data...")
    data = pd.read_pickle(data_path)
    
    # Define features
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
    
    # Constants for training
    day_col = "d"
    dt_col = "date"  # Date column name
    
    # Setup CV parameters
    cv_params = {
        "n_splits": 3,
        "train_days": int(365 * 1.5),
        "test_days": DAYS_PRED,
        "day_col": day_col,
    }
    cv = CustomTimeSeriesSplitter(**cv_params)
    
    # Split data
    logging.info("Preparing training and test data...")
    is_train = data["d"] < 1914
    
    # Make sure to include the date column for CV visualization
    X_train = data[is_train][[day_col, dt_col] + features].reset_index(drop=True)
    y_train = data[is_train]["demand"].reset_index(drop=True)
    
    logging.info(f"X_train shape: {X_train.shape}")
    
    # Show CV fold information
    logging.info("Cross-validation fold information:")
    show_cv_days(cv, X_train, dt_col, day_col)
    
    # Visualize CV splits
    plot_cv_indices(cv, X_train, dt_col)
    
    # LightGBM parameters
    bst_params = {
        "boosting_type": "gbdt",
        "metric": "rmse",
        "objective": "regression",
        "n_jobs": -1,
        "seed": 42,
        "learning_rate": 0.1,
        "bagging_fraction": 0.75,
        "bagging_freq": 10,
        "colsample_bytree": 0.75,
    }
    
    fit_params = {
        "num_boost_round": 100_000,
        "early_stopping_rounds": 50,
        "verbose_eval": 100,
    }
    
    # Train models
    logging.info("Starting LightGBM model training...")
    models = train_lgb(
        bst_params, fit_params, X_train, y_train, cv, drop_when_train=[day_col, dt_col]
    )
    
    if save_models:
        # Create models directory if it doesn't exist
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Save models
        logging.info(f"Saving {len(models)} models to {models_dir} directory...")
        for i, model in enumerate(models):
            model_path = os.path.join(models_dir, f"lgb_model_fold_{i}.txt")
            model.save_model(model_path)
            logging.info(f"Model saved to {model_path}")
    
    logging.info("Training complete!")
    return models

if __name__ == "__main__":
    setup_logging()
    
    train_model()