import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_dataset(filepath, delimiter=";", test_size=0.25, random_state=None):
    """
    Load a dataset from a CSV file into a pandas DataFrame and then split it into 
    training and test sets.
    
    Parameters:
    - filepath: str, path to the CSV file.
    - delimiter: str, the delimiter used in the CSV file. Default is ';'.
    - test_size: float, the proportion of the dataset to include in the test split. Default is 0.25.
    - random_state: int, controls the shuffling applied to the data before applying the split. Default is None.
    
    Returns:
    - train_df: pandas.DataFrame, the training set.
    - test_df: pandas.DataFrame, the test set.
    
    Raises:
    - FileNotFoundError: If the file does not exist at the specified path.
    - Exception: For issues that arise during the loading of the file.
    """
    try:
        dataframe = pd.read_csv(filepath, delimiter=delimiter)
        train_df, test_df = train_test_split(dataframe, test_size=test_size, random_state=random_state)
        return train_df, test_df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file at {filepath} was not found.") from e
    except Exception as e:
        raise Exception(f"An error occurred while loading the file: {e}") from e