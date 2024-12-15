#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#pip install pandas numpy matplotlib scipy dask seaborn scikit-learn tqdm

"""
Prognostika - Hard Drive Failure Analysis
This script performs comprehensive analysis of hard drive failure data using machine learning techniques.
"""

import pandas as pd
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import dask.dataframe as dd
from collections import Counter
import os
import glob
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.colors
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings('ignore')

# 设置pandas显示选项
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def load_data(years=['2013'], quarters=['Q1']):
    """
    加载并处理硬盘数据
    """
    required_features = ['serial_number', 'model', 'failure', 'date']
    all_data = []
    
    # Kaggle工作目录
    script_dir = '/kaggle/working'
    
    # 构建pickle文件名
    quarter_part = '_'.join(quarters) if quarters else 'full_year'
    pickle_file = os.path.join(script_dir, f"{'_'.join(years)}_{quarter_part}.pkl")
    
    # 季度月份范围
    quarter_ranges = {
        'Q1': (1, 3),   # 1月到3月
        'Q2': (4, 6),   # 4月到6月
        'Q3': (7, 9),   # 7月到9月
        'Q4': (10, 12)  # 10月到12月
    }
    
    # 检查pickle文件是否存在
    if os.path.exists(pickle_file):
        print(f"Loading data from {pickle_file}...")
        with open(pickle_file, 'rb') as f:
            df = pickle.load(f)
        print("Data loaded successfully from pickle file.")
        return df
    
    # 处理每一年的数据
    for y in tqdm(years, desc="Analyzing years"):
        # 获取目录中的所有CSV文件
        data_dir = os.path.join(script_dir, 'HDD_dataset', y)
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if not files:
            print(f"No files found in {data_dir}")
            continue
            
        print(f"Found {len(files)} files in {data_dir}")
        
        # 根据季度筛选文件
        filtered_files = []
        for q in quarters:
            start_month, end_month = quarter_ranges[q]
            
            for f in files:
                try:
                    # 解析文件名中的月份
                    file_date = f.split('.')[0]  # 去掉.csv后缀
                    file_month = int(file_date.split('-')[1])  # 获取月份并转换为整数
                    
                    # 检查月份是否在季度范围内
                    if start_month <= file_month <= end_month:
                        filtered_files.append(os.path.join(data_dir, f))
                except Exception as e:
                    print(f"Error parsing file name {f}: {str(e)}")
        
        print(f"Filtered {len(filtered_files)} files for {quarters}")
        
        if not filtered_files:
            print(f"Warning: No files found for {y} {quarters}")
            continue
        
        # 处理每个文件
        for f in tqdm(filtered_files, desc=f"Analyzing files in {y} {', '.join(quarters)}"):
            try:
                # 读取第一行来确定列
                temp_df = pd.read_csv(f, nrows=1)
                all_columns = temp_df.columns
                raw_columns = [col for col in all_columns if 'raw' in col]
                features = required_features + raw_columns
                
                # 读取完整数据
                data = pd.read_csv(f, usecols=features, parse_dates=['date'], low_memory=False)
                data['failure'] = data['failure'].astype('int')
                all_data.append(data)
                
                print(f"Successfully processed {f}, shape: {data.shape}")
                
            except Exception as e:
                print(f"Error reading file {f}: {str(e)}")
    
    # 合并所有数据
    if all_data:
        print(f"Concatenating {len(all_data)} dataframes...")
        df = pd.concat(all_data, ignore_index=True)
        df.set_index(['serial_number', 'date'], inplace=True)
        df.sort_index(inplace=True)
        
        print(f"Final dataframe shape: {df.shape}")
        
        # 保存到pickle文件
        print(f"Saving to {pickle_file}...")
        with open(pickle_file, 'wb') as f:
            pickle.dump(df, f)
        print(f"Data concatenated and saved to {pickle_file} successfully.")
    else:
        print("No data to concatenate.")
        return None
        
    return df

def analyze_manufacturers(df):
    """
    分析各制造商的数据分布
    
    Args:
        df (pd.DataFrame): 输入数据框
    """
    total_records = df.shape[0]
    
    # 按制造商分割数据
    manufacturers = {
        'Seagate': 'S',
        'HGST': 'HG',
        'Toshiba': 'T',
        'WDC': 'W',
        'Hitachi': 'Hi'
    }
    
    manufacturer_stats = {}
    for name, prefix in manufacturers.items():
        mfr_df = df[df["model"].str.startswith(prefix)]
        percentage = (mfr_df.shape[0] / total_records) * 100
        manufacturer_stats[name] = {
            'records': mfr_df.shape[0],
            'percentage': percentage
        }
        print(f"{name} records: {mfr_df.shape}, {percentage:.2f}% of total")
    
    # 绘制饼图
    plt.figure(figsize=(10, 10))
    plt.pie([stats['percentage'] for stats in manufacturer_stats.values()],
            labels=manufacturer_stats.keys(),
            autopct='%1.1f%%',
            startangle=90)
    plt.axis('equal')
    plt.title('Distribution of Hard Drive Manufacturers')
    plt.show()

def main():
    """主函数"""
    # 设置更详细的日志
    print("Starting analysis...")
    print(f"Current working directory: {os.getcwd()}")
    
    # 加载2013年第二季度的数据（4-6月）
    df = load_data(years=['2013'], quarters=['Q2'])
    if df is None:
        print("Failed to load data")
        return
    
    print(f"Loaded dataframe with shape: {df.shape}")
    
    # 备份原始数据
    backup_df = df.copy()
    
    # 分析制造商分布
    print("Analyzing manufacturer distribution...")
    analyze_manufacturers(df)
    
    print("Analysis completed")

if __name__ == "__main__":
    main() 