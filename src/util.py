import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data_and_split(file_path, test_size=0.2, random_state=42, delimiter=None):
    
    """
    Load data from a specified file and split it into training and testing datasets.

    This function is designed to work with a variety of file formats including CSV, TXT, Excel (XLS/XLSX), and JSON.
    It reads the data into a pandas DataFrame and splits it into two separate DataFrames for training and testing purposes.
    
    The function assumes that the input file is structured with features and an optional target variable. If a delimiter
    is not specified for CSV/TXT files, the function defaults to using a comma (','). For Excel and JSON files, the delimiter
    parameter is ignored.

    Parameters:
    ----------
    file_path : str
        The path to the data file.
    
    test_size : float, optional (default=0.2)
        The proportion of the dataset to include in the test split. Should be between 0.0 and 1.0.
    
    random_state : int, optional (default=42)
        Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    
    delimiter : str, optional (default=None)
        The delimiter used in the file if it's a CSV or TXT. If None, defaults to ',' for CSV.

    Returns:
    -------
    train_df : DataFrame
        The training dataset, containing a subset of rows from the original data file.
    
    test_df : DataFrame
        The testing dataset, containing the remaining rows not included in the training dataset.

    Raises:
    ------
    ValueError
        If the file type is not supported (not in CSV, TXT, Excel, or JSON formats).

    Examples:
    --------
    To split a CSV file with a comma as a delimiter:

    >>> train_df, test_df = load_data_and_split('data/bank-full.csv', test_size=0.25, random_state=123)

    To split a CSV file with a semi-colon as a delimiter:

    >>> train_df, test_df = load_data_and_split('data/bank-full.csv', test_size=0.25, random_state=123, delimiter=';')

    Note:
    -----
    The file path provided to the function must be accessible from the current working directory.
    """
    
    # Determine the file type
    _, file_extension = os.path.splitext(file_path)
    
    # Load the data according to the file type
    if file_extension in ['.csv', '.txt']:
        data = pd.read_csv(file_path, delimiter=delimiter)
    elif file_extension in ['.xls', '.xlsx']:
        data = pd.read_excel(file_path)
    elif file_extension == '.json':
        data = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type!")
    
    # Split the data into train and test sets
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    
    return train_df, test_df
