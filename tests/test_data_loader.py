import pytest
import pandas as pd
from data_loader import load_and_split_dataset

def test_load_and_split_dataset(mocker):
    # Create a mock DataFrame to be returned by pd.read_csv
    mock_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['a', 'b', 'c']
    })

    # Mock pd.read_csv to return the mock DataFrame
    mocker.patch('pandas.read_csv', return_value=mock_data)

    # Call the function with the mock file path
    train_df, test_df = load_and_split_dataset('mock_file.csv', delimiter=';', test_size=0.25, random_state=42)

    # Assert the mock was called
    assert pd.read_csv.called

    # Assert the DataFrame is split into two parts with correct proportions
    assert len(train_df) == int(0.5 * len(mock_data))
    assert len(test_df) == len(mock_data) - len(train_df)

def test_load_and_split_dataset_file_not_found(mocker):
    # Mock pd.read_csv to raise FileNotFoundError for a non-existent file
    mocker.patch('pandas.read_csv', side_effect=FileNotFoundError)

    # Test to ensure FileNotFoundError is raised for non-existent files
    with pytest.raises(FileNotFoundError):
        load_and_split_dataset('non_existent_file.csv')

def test_load_and_split_dataset_invalid_data(mocker):
    # Mock pd.read_csv to raise a ValueError for invalid data
    mocker.patch('pandas.read_csv', side_effect=ValueError("Invalid data"))

    # Assert that the function raises an exception on invalid data
    with pytest.raises(ValueError):
        load_and_split_dataset('mock_file.csv')
