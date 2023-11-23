# Create a test dataframe by using a small sample data.
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from io import StringIO

# Sample data to simulate a CSV file for testing.
data_csv = """feature1,feature2,label
1,2,0
3,4,1
5,6,0
7,8,1
9,10,0
"""

# Write sample data to a CSV file for testing
csv_file_path = 'data/sample_data.csv'
with open(csv_file_path, 'w') as file:
    file.write(data_csv)

# Convert the CSV data into a DataFrame
df = pd.read_csv(StringIO(data_csv))

# Define the Excel file path
excel_file_path = 'data/sample_data.xlsx'

# Write the DataFrame to an Excel file for testing
df.to_excel(excel_file_path, index=False)

import pytest
from load_data_and_split import load_data_and_split

# Test to ensure the split ratio is correct for csv file
def test_split_ratio():
    train_df, test_df = load_data_and_split(csv_file_path, test_size=0.2)
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01)

# Test to ensure the function raises a ValueError for unsupported file types
def test_unsupported_file_type():
    with pytest.raises(ValueError):
        _, _ = load_data_and_split('/path/to/data.unsupported')

# Test to ensure the function correctly uses the delimiter
def test_delimiter_handling():
    # This should pass as we are using the default delimiter which is a comma
    train_df, test_df = load_data_and_split(csv_file_path, delimiter=',')
    assert not train_df.empty and not test_df.empty

# Test to ensure the split ratio is correct for excel file
def test_excel_data_loading():
    train_df, test_df = load_data_and_split(excel_file_path, test_size=0.2)
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01)
