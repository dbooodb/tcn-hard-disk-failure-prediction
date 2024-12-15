!pip install wget
import os
import shutil
import wget
import zipfile
import pandas as pd
import datetime
import numpy as np
import IPython
from glob import glob

# Config
# Base URL: The BackBlaze dataset URL
base_url = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/"

# Base Path: Directory to hold the dataset
# -- old
#notebook_path = IPython.get_ipython().starting_dir
#base_path = os.path.abspath(os.path.join(notebook_path, '..', 'HDD_dataset'))
#os.makedirs(base_path, exist_ok=True)
base_path = '/kaggle/working/HDD_dataset'
os.makedirs(base_path, exist_ok=True)
base_path

# Output Path: Directory to output the database
# -- old
#output_dir = os.path.abspath(os.path.join(notebook_path, '..', 'output'))
#os.makedirs(output_dir, exist_ok=True)
output_dir = '/kaggle/working/output'
os.makedirs(output_dir, exist_ok=True)
output_dir

# Years: Years of data to download and analyze (From 2013 to 2019)
# -- old
#years = [str(year) for year in range(2013, 2020)]
years = ['2013']

# Model: The specific HDD model to analyze
model = "ST3000DM001"

# Find Failed: if True, keep only failed HDDs, otherwise keep all HDDs
find_failed = False
suffix = 'failed' if find_failed else 'all'

# Define directories for each year
year_dirs = {year: os.path.join(base_path, year) for year in years}
years_list = "_" + "_".join(years)

# Directory mapping for zip files
suffixes = {
    "data_2013.zip": '2013',
    "data_2014.zip": '2014',
    "data_2015.zip": '2015',
    "data_Q1_2016.zip": None,
    "data_Q2_2016.zip": None,
    "data_Q3_2016.zip": None,
    "data_Q4_2016.zip": None,
    "data_Q1_2017.zip": None,
    "data_Q2_2017.zip": None,
    "data_Q3_2017.zip": None,
    "data_Q4_2017.zip": None,
    "data_Q1_2018.zip": None,
    "data_Q2_2018.zip": None,
    "data_Q3_2018.zip": None,
    "data_Q4_2018.zip": None,
    "data_Q1_2019.zip": None,
    "data_Q2_2019.zip": None,
    "data_Q3_2019.zip": None,
}

# Download and extract dataset
years = [str(_) for _ in years]
for y in years:
    print("Year:", y)
    year_path = os.path.join(base_path, y)
    os.makedirs(year_path, exist_ok=True)
    for zip_name, unzip_dir in suffixes.items():
        if y in zip_name:
            url = base_url + zip_name
            zip_path = os.path.join(base_path, zip_name)
            if not os.path.exists(zip_path):
                print("Downloading:", url)
                wget.download(url, out=base_path)
            print("\nUnzipping:", zip_path)
            dest_path = year_path if unzip_dir is None else base_path
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(dest_path)

            if unzip_dir is not None and unzip_dir != y:
                unzip_path = os.path.join(dest_path, unzip_dir)
                for f in os.listdir(unzip_path):
                    shutil.move(os.path.join(unzip_path, f),
                            os.path.join(year_path, f))
                os.rmdir(unzip_path)

# Collect serial numbers
list_failed = []

for year in years:
    year_dir = year_dirs[year]
    files = glob(os.path.join(year_dir, '*.csv'))

    for file_path in sorted(files):
        try:
            file_r = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: The file {file_path} does not exist.")
            continue

        model_chosen = file_r[file_r['model'] == model]

        if model_chosen.empty:
            continue

        if find_failed:
            model_chosen = model_chosen[model_chosen['failure'] == 1]

        list_failed.extend(model_chosen['serial_number'].values)

np.save(os.path.join(output_dir, f'HDD{years_list}_{suffix}_{model}.npy'), list_failed)

# Create DataFrame
failed = set(np.load(os.path.join(output_dir, f'HDD{years_list}_{suffix}_{model}.npy')))
database = pd.DataFrame()

for year in years:
    year_path = year_dirs[year]
    files = sorted([f for f in os.listdir(year_path) if f.endswith('.csv')])

    for file in files:
        file_path = os.path.join(year_path, file)
        file_date = datetime.datetime.strptime(file.split('.')[0], '%Y-%m-%d')
        old_time = datetime.datetime.strptime(f'{year}-01-01', '%Y-%m-%d')
        
        if file_date >= old_time:
            df = pd.read_csv(file_path)
            model_chosen = df[df['model'] == model]
            relevant_rows = model_chosen[model_chosen['serial_number'].isin(failed)]

            drop_columns = [col for col in relevant_rows if 'smart_' in col and int(col.split('_')[1]) in {22, 220, 222, 224, 226}]
            relevant_rows.drop(columns=drop_columns, errors='ignore', inplace=True)

            database = pd.concat([database, relevant_rows], ignore_index=True)
            print('adding day ' + str(model_chosen['date'].values))

database.to_pickle(os.path.join(output_dir, f'HDD{years_list}_{suffix}_{model}_appended.pkl'))

# -- old
#most_common_models = df.groupby(['model'], as_index=True)['model', 'date'].size()
#most_common_models = most_common_models.sort_values(ascending=False) 

# Check the most common models
most_common_models = database.groupby('model').size()
most_common_models = most_common_models.sort_values(ascending=False)
print("Most common models:")
print(most_common_models)


# 2013的数据格式
#date,serial_number,model,capacity_bytes,failure,smart_1_normalized,smart_1_raw,smart_2_normalized,smart_2_raw,smart_3_normalized,smart_3_raw,smart_4_normalized,smart_4_raw,smart_5_normalized,smart_5_raw,smart_7_normalized,smart_7_raw,smart_8_normalized,smart_8_raw,smart_9_normalized,smart_9_raw,smart_10_normalized,smart_10_raw,smart_11_normalized,smart_11_raw,smart_12_normalized,smart_12_raw,smart_13_normalized,smart_13_raw,smart_15_normalized,smart_15_raw,smart_183_normalized,smart_183_raw,smart_184_normalized,smart_184_raw,smart_187_normalized,smart_187_raw,smart_188_normalized,smart_188_raw,smart_189_normalized,smart_189_raw,smart_190_normalized,smart_190_raw,smart_191_normalized,smart_191_raw,smart_192_normalized,smart_192_raw,smart_193_normalized,smart_193_raw,smart_194_normalized,smart_194_raw,smart_195_normalized,smart_195_raw,smart_196_normalized,smart_196_raw,smart_197_normalized,smart_197_raw,smart_198_normalized,smart_198_raw,smart_199_normalized,smart_199_raw,smart_200_normalized,smart_200_raw,smart_201_normalized,smart_201_raw,smart_223_normalized,smart_223_raw,smart_225_normalized,smart_225_raw,smart_240_normalized,smart_240_raw,smart_241_normalized,smart_241_raw,smart_242_normalized,smart_242_raw,smart_250_normalized,smart_250_raw,smart_251_normalized,smart_251_raw,smart_252_normalized,smart_252_raw,smart_254_normalized,smart_254_raw,smart_255_normalized,smart_255_raw
#2013-04-10,MJ0351YNG9Z0XA,Hitachi HDS5C3030ALA630,3000592982016,0,,0,,,,,,,,0,,,,,,4031,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,26,,,,,,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,
