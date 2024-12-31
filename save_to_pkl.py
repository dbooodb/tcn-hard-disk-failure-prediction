import os
import numpy as np
import pandas as pd
import datetime
import argparse


def save_to_pkl(model, years, failed, base_path, output_dir):
    try:
        print(f"Starting save_to_pkl with model={model}, years={years}")
        years_list = "_" + "_".join(years)
        models = [m.strip() for m in model.split(',')]
        model_str = "_".join(models)
        suffix = 'failed' if failed else 'all'

        hdd_model_file_path = os.path.join(output_dir, f'HDD{years_list}_{suffix}_{model_str}.npy')
        print(f"Looking for .npy file at: {hdd_model_file_path}")
        
        if not os.path.exists(hdd_model_file_path):
            raise FileNotFoundError(f"Required .npy file not found: {hdd_model_file_path}")
            
        # Load the hard drives data
        try:
            hdd_model_data = set(np.load(hdd_model_file_path))
            print(f"Loaded {len(hdd_model_data)} serial numbers from .npy file")
        except Exception as e:
            print(f"Error loading .npy file: {str(e)}")
            raise
        
        hdd_model_pkl_file_path = os.path.join(output_dir, f'HDD{years_list}_{suffix}_{model_str}_appended.pkl')

        database = pd.DataFrame()

        # Define the directories for each year
        year_dirs = {year: os.path.join(base_path, year) for year in years}
        print(f"Processing directories: {year_dirs}")

        # Iterate over each year
        for year in years:
            try:
                year_path = year_dirs[year]
                if not os.path.exists(year_path):
                    print(f"Warning: Year directory not found: {year_path}")
                    continue
                    
                files = sorted([f for f in os.listdir(year_path) if f.endswith('.csv')])
                print(f"Found {len(files)} CSV files in {year_path}")

                # Iterate over each file in the directory
                for file in files:
                    try:
                        file_path = os.path.join(year_path, file)
                        file_date = datetime.datetime.strptime(file.split('.')[0], '%Y-%m-%d')
                        old_time = datetime.datetime.strptime(f'{year}-01-01', '%Y-%m-%d')
                        
                        if file_date >= old_time:
                            print(f"Processing file: {file}")
                            try:
                                df = pd.read_csv(file_path)
                                print(f"CSV loaded successfully. Shape: {df.shape}")
                            except Exception as e:
                                print(f"Error reading CSV file {file}: {str(e)}")
                                continue

                            for model in models:
                                try:
                                    print(f"Processing model {model}")
                                    model_chosen = df[df['model'] == model]
                                    print(f"Found {len(model_chosen)} rows for model {model}")
                                    
                                    relevant_rows = model_chosen[model_chosen['serial_number'].isin(hdd_model_data)]
                                    print(f"Found {len(relevant_rows)} relevant rows for model {model} in {file}")

                                    if len(relevant_rows) > 0:
                                        print(f"Column names before drop: {relevant_rows.columns.tolist()}")
                                        # Drop unnecessary columns since the following columns are not standard for all models
                                        drop_columns = [col for col in relevant_rows if 'smart_' in col and int(col.split('_')[1]) in {22, 220, 222, 224, 226}]
                                        relevant_rows.drop(columns=drop_columns, errors='ignore', inplace=True)
                                        print(f"Column names after drop: {relevant_rows.columns.tolist()}")

                                        # Append the row to the database
                                        try:
                                            database = pd.concat([database, relevant_rows], ignore_index=True)
                                            print(f"Database size after concat: {len(database)}")
                                        except Exception as e:
                                            print(f"Error during concatenation: {str(e)}")
                                            print(f"Database dtypes: {database.dtypes}")
                                            print(f"Relevant rows dtypes: {relevant_rows.dtypes}")
                                            raise
                                except Exception as e:
                                    print(f"Error processing model {model} in file {file}: {str(e)}")
                                    continue
                    except Exception as e:
                        print(f"Error processing file {file}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error processing year {year}: {str(e)}")
                continue

        print(f"Total rows in final database: {len(database)}")
        # Save the database to a pickle file
        try:
            database.to_pickle(hdd_model_pkl_file_path)
            return f'Data saved to {hdd_model_pkl_file_path}'
        except Exception as e:
            print(f"Error saving pickle file: {str(e)}")
            raise
    except Exception as e:
        print(f"Fatal error in save_to_pkl: {str(e)}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save HDD data to pickle')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., ST4000DM000)')
    parser.add_argument('--years', type=str, required=True, help='Years to process (comma-separated, e.g., 2016,2017)')
    parser.add_argument('--base_path', type=str, default='./HDD_dataset', help='Base path for data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--failed', action='store_true', help='Generate only failed models')

    args = parser.parse_args()
    
    # Convert years string to list
    years = [year.strip() for year in args.years.split(',')]
    
    try:
        result = save_to_pkl(args.model, years, args.failed, args.base_path, args.output_dir)
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")