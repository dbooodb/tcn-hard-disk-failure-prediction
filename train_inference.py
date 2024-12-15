import os
import shutil
import wget
import zipfile
import pandas as pd
import datetime
import numpy as np
from glob import glob
import sys
import torch
import torch.optim as optim
from pathlib import Path

sys.path.append("..")
from algorithms.Networks_pytorch import *
from algorithms.Dataset_manipulation import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

def randomForestClassification(X_train, Y_train, X_test, Y_test, metric, **args):
    Y_test_real = []
    prediction = []
    # Train and validate the network using RandomForest
    X_train, Y_train = shuffle(X_train, Y_train)
    model = RandomForestClassifier(n_estimators=30, min_samples_split=10, random_state=3)
    model.fit(X_train[:, :], Y_train)
    prediction = model.predict(X_test)
    Y_test_real = Y_test
    report_metrics(Y_test_real, prediction, metric)

def TCNClassification(X_train, Y_train, X_test, Y_test, metric, **args):
    # Train and validate the network using TCN
    net_train_validate_tcn(args['net'], args['optimizer'], X_train, Y_train, X_test, Y_test, 
                          args['epochs'], args['batch_size'], args['lr'])

def LSTMClassification(X_train, Y_train, X_test, Y_test, metric, **args):
    # Train and validate the network using LSTM
    train_dataset = FPLSTMDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], 
                                             shuffle=True, collate_fn=FPLSTM_collate)
    test_dataset = FPLSTMDataset(X_test, Y_test.values)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['batch_size'], 
                                            shuffle=True, collate_fn=FPLSTM_collate)
    net_train_validate_LSTM(args['net'], args['optimizer'], train_loader, test_loader, 
                           args['epochs'], X_test.shape[0], X_train.shape[0], args['lr'])

def classification(X_train, Y_train, X_test, Y_test, classifier, metric, **args):
    """
    Perform classification using the specified classifier.
    
    Parameters:
    - X_train (array-like): Training data features.
    - Y_train (array-like): Training data labels. 
    - X_test (array-like): Test data features.
    - Y_test (array-like): Test data labels.
    - classifier (str): The classifier to use. Options: 'RandomForest', 'TCN', 'LSTM'.
    - metric (str): The metric to evaluate the classification performance.
    - **args: Additional arguments specific to each classifier.
    """
    print('Classification using {} is starting'.format(classifier))
    if classifier == 'RandomForest':
        randomForestClassification(X_train, Y_train, X_test, Y_test, metric, **args)
    elif classifier == 'TCN':
        TCNClassification(X_train, Y_train, X_test, Y_test, metric, **args)
    elif classifier == 'LSTM':
        LSTMClassification(X_train, Y_train, X_test, Y_test, metric, **args)

def load_backblaze_data(data_dir):
    """
    加载Backblaze数据集
    
    Parameters:
    - data_dir: 数据目录路径
    
    Returns:
    - 合并后的DataFrame
    """
    print(f"Loading data from {data_dir}")
    all_files = glob.glob(f"{data_dir}/*.csv")
    print(f"Found {len(all_files)} files")
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid CSV files found")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(combined_df)}")
    return combined_df

def main():
    # 配置Kaggle数据路径
    DATA_DIR = "/kaggle/working/HDD_dataset/2013"
    
    # 基本配置
    model = 'ST3000DM001'
    years = ['2013']  # 修改为只使用2013年数据
    windowing = 1
    min_days_HDD = 115
    days_considered_as_failure = 7
    test_train_perc = 0.3
    oversample_undersample = 2
    balancing_normal_failed = 20
    history_signal = 32
    classifier = 'LSTM'
    features_extraction_method = False
    CUDA_DEV = "0"
    ranking = 'Ok'
    num_features = 18
    overlap = 1

    # Features definition
    features = {
        'total_features': [
            'date', 'serial_number', 'model', 'failure',
            'smart_1_normalized', 'smart_5_normalized', 'smart_5_raw',
            'smart_7_normalized', 'smart_9_raw', 'smart_12_raw',
            'smart_183_raw', 'smart_184_normalized', 'smart_184_raw',
            'smart_187_normalized', 'smart_187_raw', 'smart_189_normalized',
            'smart_193_normalized', 'smart_193_raw', 'smart_197_normalized',
            'smart_197_raw', 'smart_198_normalized', 'smart_198_raw',
            'smart_199_raw'
        ],
        'iSTEP': [
            'date', 'serial_number', 'model', 'failure',
            'smart_5_raw', 'smart_3_raw', 'smart_10_raw',
            'smart_12_raw', 'smart_4_raw', 'smart_194_raw',
            'smart_1_raw', 'smart_9_raw', 'smart_192_raw',
            'smart_193_raw', 'smart_197_raw', 'smart_198_raw',
            'smart_199_raw'
        ]
    }

    # 加载数据
    try:
        cache_path = Path(f'/kaggle/working/cache_{model}_Dataset.pkl')
        if cache_path.exists():
            print("Loading cached dataset...")
            df = pd.read_pickle(cache_path)
        else:
            print("Loading and processing raw data...")
            df = load_backblaze_data(DATA_DIR)
            # 只保留指定型号的数据
            df = df[df['model'] == model].copy()
            
            # 数据预处理
            bad_missing_hds, bad_power_hds, df = filter_HDs_out(df, 
                                                              min_days=min_days_HDD,
                                                              time_window='30D', 
                                                              tolerance=30)
            
            df['predict_val'], df['validate_val'] = generate_failure_predictions(df,
                                                    days=days_considered_as_failure,
                                                    window=history_signal)
            
            if ranking != 'None':
                df = feature_selection(df, num_features)
                
            # 缓存处理后的数据
            df.to_pickle(cache_path)
            print(f"Dataset cached to {cache_path}")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Dataset shape:", df.shape)
    print("Number of failures:", df['failure'].sum())

    # Dataset partitioning
    Xtrain, Xtest, ytrain, ytest = dataset_partitioning(
        df, model, overlap=overlap, rank=ranking,
        num_features=num_features, technique='random',
        test_train_perc=test_train_perc, windowing=windowing,
        window_dim=history_signal, resampler_balancing=balancing_normal_failed,
        oversample_undersample=oversample_undersample
    )

    # Classifier specific setup
    if classifier == 'LSTM':
        lr = 0.001
        batch_size = 256
        epochs = 300
        dropout = 0.1
        lstm_hidden_s = 64
        fc1_hidden_s = 16
        num_inputs = Xtrain.shape[1]
        
        # 确保CUDA可用
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        net = FPLSTM(lstm_hidden_s, fc1_hidden_s, num_inputs, 2, dropout)
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        
        try:
            classification(
                X_train=Xtrain, Y_train=ytrain,
                X_test=Xtest, Y_test=ytest,
                classifier=classifier,
                metric=['RMSE', 'MAE', 'FDR', 'FAR', 'F1', 'recall', 'precision'],
                net=net, optimizer=optimizer,
                epochs=epochs, batch_size=batch_size, lr=lr
            )
        except Exception as e:
            print(f"Error during training: {e}")

if __name__ == "__main__":
    main() 