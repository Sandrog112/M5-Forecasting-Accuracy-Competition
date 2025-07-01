import pandas as pd
import numpy as np
import gc
import warnings
import logging
import os
from sklearn.preprocessing import LabelEncoder
from pandas.errors import SettingWithCopyWarning

def setup_logging():
    """
    Sets up logging configuration to save logs to log folder.
    Creates log folder if it doesn't exist.
    """
    log_folder = "logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    log_file = os.path.join(log_folder, "data_preprocessing.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def reduce_mem_usage(df, verbose=False):
    """
    Reduces memory usage of a dataframe by downcasting numeric columns.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        verbose (bool): If True, logs memory reduction information
        
    Returns:
        pandas.DataFrame: Memory-optimized dataframe
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    int_columns = df.select_dtypes(include=["int"]).columns
    float_columns = df.select_dtypes(include=["float"]).columns

    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        logging.info(
            f"Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )
    return df

def encode_categorical(df, cols):
    """
    Encodes categorical columns using LabelEncoder.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        cols (list): List of column names to encode
        
    Returns:
        pandas.DataFrame: Dataframe with encoded columns
    """
    for col in cols:
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)

    return df

def extract_num(ser):
    """
    Extracts numeric part from strings.
    
    Args:
        ser (pandas.Series): Series containing strings with numeric parts
        
    Returns:
        pandas.Series: Series with extracted numbers as integers
    """
    return ser.str.extract(r"(\d+)").astype(np.int16)

def reshape_sales(sales, submission, d_thresh=0, verbose=False):
    """
    Reshapes sales data from wide to long format and combines with submission data.
    
    Args:
        sales (pandas.DataFrame): Sales data in wide format
        submission (pandas.DataFrame): Submission template
        d_thresh (int): Day threshold for filtering
        verbose (bool): If True, logs information about the process
        
    Returns:
        pandas.DataFrame: Reshaped and combined data
    """
    logging.info("Reshaping sales data...")
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    DAYS_PRED = submission.shape[1] - 1

    product = sales[id_columns]

    sales = sales.melt(id_vars=id_columns, var_name="d", value_name="demand")
    sales = reduce_mem_usage(sales)

    vals = submission[submission["id"].str.endswith("validation")]
    evals = submission[submission["id"].str.endswith("evaluation")]

    vals.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + DAYS_PRED)]
    evals.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + DAYS_PRED)]

    evals["id"] = evals["id"].str.replace("_evaluation", "_validation")
    vals = vals.merge(product, how="left", on="id")
    evals = evals.merge(product, how="left", on="id")
    evals["id"] = evals["id"].str.replace("_validation", "_evaluation")

    if verbose:
        logging.info("Validation and evaluation data samples prepared")

    vals = vals.melt(id_vars=id_columns, var_name="d", value_name="demand")
    evals = evals.melt(id_vars=id_columns, var_name="d", value_name="demand")

    sales["part"] = "train"
    vals["part"] = "validation"
    evals["part"] = "evaluation"

    data = pd.concat([sales, vals, evals], axis=0)

    del sales, vals, evals

    data["d"] = extract_num(data["d"])
    data = data[data["d"] >= d_thresh]

    data = data[data["part"] != "evaluation"]

    gc.collect()

    logging.info(f"Reshaped data shape: {data.shape}")
    return data

def merge_calendar(data, calendar):
    """
    Merges reshaped data with calendar information.
    
    Args:
        data (pandas.DataFrame): Reshaped sales data
        calendar (pandas.DataFrame): Calendar data with dates and events
        
    Returns:
        pandas.DataFrame: Merged data
    """
    logging.info("Merging with calendar data...")
    calendar = calendar.drop(["weekday", "wday", "month", "year"], axis=1)
    return data.merge(calendar, how="left", on="d")

def merge_prices(data, prices):
    """
    Merges data with pricing information.
    
    Args:
        data (pandas.DataFrame): Data from previous steps
        prices (pandas.DataFrame): Pricing data
        
    Returns:
        pandas.DataFrame: Merged data
    """
    logging.info("Merging with pricing data...")
    return data.merge(prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])

def add_demand_features(df):
    """
    Adds demand-related features like shifts and rolling statistics.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with additional features
    """
    logging.info("Adding demand features...")
    DAYS_PRED = 28  
    
    for diff in [0, 1, 2]:
        shift = DAYS_PRED + diff
        df[f"shift_t{shift}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(shift)
        )

    for window in [7, 30, 60, 90, 180]:
        df[f"rolling_std_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).std()
        )

    for window in [7, 30, 60, 90, 180]:
        df[f"rolling_mean_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).mean()
        )

    for window in [7, 30, 60]:
        df[f"rolling_min_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).min()
        )

    for window in [7, 30, 60]:
        df[f"rolling_max_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).max()
        )

    df["rolling_skew_t30"] = df.groupby(["id"])["demand"].transform(
        lambda x: x.shift(DAYS_PRED).rolling(30).skew()
    )
    df["rolling_kurt_t30"] = df.groupby(["id"])["demand"].transform(
        lambda x: x.shift(DAYS_PRED).rolling(30).kurt()
    )
    return df

def add_price_features(df):
    """
    Adds price-related features like price changes and rolling statistics.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with additional features
    """
    logging.info("Adding price features...")
    df["shift_price_t1"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1)
    )
    df["price_change_t1"] = (df["shift_price_t1"] - df["sell_price"]) / (
        df["shift_price_t1"]
    )
    df["rolling_price_max_t365"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1).rolling(365).max()
    )
    df["price_change_t365"] = (df["rolling_price_max_t365"] - df["sell_price"]) / (
        df["rolling_price_max_t365"]
    )

    df["rolling_price_std_t7"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.rolling(7).std()
    )
    df["rolling_price_std_t30"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.rolling(30).std()
    )
    return df.drop(["rolling_price_max_t365", "shift_price_t1"], axis=1)

def add_time_features(df, dt_col):
    """
    Adds time-related features from a date column.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        dt_col (str): Name of the date column
        
    Returns:
        pandas.DataFrame: Dataframe with additional time features
    """
    logging.info("Adding time features...")
    df[dt_col] = pd.to_datetime(df[dt_col])
    attrs = [
        "year",
        "quarter",
        "month",
        "day",
        "dayofweek",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        df[attr] = getattr(df[dt_col].dt, attr).astype(dtype)

    df["week"] = df[dt_col].dt.isocalendar().week.astype(np.int8)
    
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    return df

def preprocess_data(save=True):
    """
    Main function that orchestrates the entire preprocessing pipeline.
    
    Args:
        save (bool): If True, saves the processed data to disk
        
    Returns:
        pandas.DataFrame: Fully processed dataframe
    """
    setup_logging()
    logging.info("Starting data preprocessing...")
    
    logging.info("Loading data files...")
    sales = pd.read_csv("data/sales_train_validation.csv")
    calendar = pd.read_csv("data/calendar.csv")
    prices = pd.read_csv("data/sell_prices.csv")
    submission = pd.read_csv("data/sample_submission.csv")
    
    NUM_ITEMS = sales.shape[0]
    DAYS_PRED = submission.shape[1] - 1
    
    logging.info(f"Sales shape: {sales.shape}")
    logging.info(f"Calendar shape: {calendar.shape}")
    logging.info(f"Prices shape: {prices.shape}")
    logging.info(f"Submission shape: {submission.shape}")
    
    sales = reduce_mem_usage(sales, verbose=True)
    calendar = reduce_mem_usage(calendar, verbose=True)
    prices = reduce_mem_usage(prices, verbose=True)
    submission = reduce_mem_usage(submission, verbose=True)
    
    logging.info("Encoding categorical variables...")
    calendar = encode_categorical(
        calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    ).pipe(reduce_mem_usage)

    sales = encode_categorical(
        sales, ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
    ).pipe(reduce_mem_usage)

    prices = encode_categorical(prices, ["item_id", "store_id"]).pipe(reduce_mem_usage)
    
    warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
    
    d_thresh = 1941 - int(365 * 2)
    data = reshape_sales(sales, submission, d_thresh=d_thresh)
    del sales
    gc.collect()
    
    calendar["d"] = extract_num(calendar["d"])
    data = merge_calendar(data, calendar)
    del calendar
    gc.collect()
    
    data = merge_prices(data, prices)
    del prices
    gc.collect()
    
    data = reduce_mem_usage(data, verbose=True)
    
    data = add_demand_features(data).pipe(reduce_mem_usage)
    data = add_price_features(data).pipe(reduce_mem_usage)
    data = add_time_features(data, "date").pipe(reduce_mem_usage)
    
    data = data.sort_values("date")
    
    logging.info(f"Start date: {data['date'].min()}")
    logging.info(f"End date: {data['date'].max()}")
    logging.info(f"Final data shape: {data.shape}")
    
    if save:
        output_path = "data/processed_data.pkl"
        logging.info(f"Saving processed data to {output_path}")
        data.to_pickle(output_path)
        
    logging.info("Data preprocessing complete!")
    return data

if __name__ == "__main__":
    preprocess_data(save=True)