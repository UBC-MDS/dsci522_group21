# Create a test dataframe by using a small sample data.
import pandas as pd
import os
import sys
from io import StringIO
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.util import load_data_and_split

# for load_data_and_split
data_csv = """feature1,feature2,label
1,2,0
3,4,1
5,6,0
7,8,1
9,10,0
1,2,0
3,4,1
5,6,0
7,8,1
9,10,0
"""

csv_file_path = 'data/sample_data.csv'
with open(csv_file_path, 'w') as file:
    file.write(data_csv)

txt_file_path = 'data/sample_data.txt'
with open(txt_file_path, 'w') as file:
    file.write(data_csv)

df = pd.read_csv(StringIO(data_csv))

excel_file_path = 'data/sample_data.xlsx'
df.to_excel(excel_file_path, index=False)

csv_semicolon_path = 'data/sample_data_semicolon.csv'
df.to_csv(csv_semicolon_path, index=False, sep=";")

csv_tab_path = 'data/sample_data_tab.csv'
df.to_csv(csv_tab_path, index=False, sep="\t")

json_path = 'data/sample_data.json'
df.to_json(json_path, index=False)

# Test to ensure the split ratio is correct for csv file
def test_load_split_ratio_csv():
    train_df, test_df = load_data_and_split(csv_file_path, test_size=0.2)
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01)

# Test to ensure the function raises a ValueError for unsupported file types
def test_load_unsupported_file_type():
    with pytest.raises(ValueError):
        _, _ = load_data_and_split('/path/to/data.unsupported')

# Test to ensure the function correctly uses the delimiter
def test_load_delimiter_handling():
    # This should pass as we are using the default delimiter which is a comma
    train_df, test_df = load_data_and_split(csv_file_path, delimiter=',')
    assert not train_df.empty and not test_df.empty

# Test to ensure the split ratio is correct for excel file
def test_load_excel_data_loading():
    train_df, test_df = load_data_and_split(excel_file_path, test_size=0.2)
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01)

# Test to ensure the function works for other common delimiters
def test_load_another_delim():
    train_df, test_df = load_data_and_split(csv_semicolon_path, delimiter=';')
    train_df_2, test_df_2 = load_data_and_split(csv_tab_path, delimiter='\t')
    assert not train_df.empty and not test_df.empty
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01)
    assert not train_df_2.empty and not test_df_2.empty
    assert len(test_df_2) / (len(test_df_2) + len(train_df_2)) == pytest.approx(0.2, 0.01)

# Test to ensure the function works for text files
def test_load_txt():
    train_df, test_df = load_data_and_split(txt_file_path)
    assert not train_df.empty and not test_df.empty
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01)

# Test to ensure the function works for json files
def test_load_json():
    train_df, test_df = load_data_and_split(json_path)
    assert not train_df.empty and not test_df.empty
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01)

# Test to ensure the function works with default and non-default split size
def test_load_split_ratio():
    train_df, test_df = load_data_and_split(csv_file_path)
    train_df_2, test_df_2 = load_data_and_split(csv_file_path, test_size=0.3)
    assert len(test_df) / (len(test_df) + len(train_df)) == pytest.approx(0.2, 0.01)
    assert len(test_df_2) / (len(test_df_2) + len(train_df_2)) == pytest.approx(0.3, 0.01)