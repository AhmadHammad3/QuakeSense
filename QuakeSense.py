#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import importlib
import torch
import numpy as np
import os
import csv
import torch.nn as nn
import torch.optim as optim
import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
import obspy


# In[ ]:


def show_versions():
    packages = {
        'numpy': 'np',
        'pandas': 'pd',
        'sklearn': 'sklearn',
        'matplotlib': 'plt',
        'obspy': 'obspy',
        'pywt': 'pywt',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'torchaudio': 'torchaudio',
    }

    for pkg_name, alias in packages.items():
        try:
            module = importlib.import_module(pkg_name)
            print(f"{pkg_name} version: {module.__version__}")
        except ImportError:
            print(f"{pkg_name} is NOT installed. Installing it now...")
            get_ipython().system('pip install {pkg_name}')
            print(f"{pkg_name} has been installed.")
        except AttributeError:
            print(f"Could not find the version for {pkg_name}")

show_versions()


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[ ]:


mseed_dir_lunar_train = "./NasaProject/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA"
mseed_dir_lunar_test = "./NasaProject/space_apps_2024_seismic_detection/data/lunar/test/data/S12_GradeB"
mseed_dir_mars_train = "./NasaProject/space_apps_2024_seismic_detection/data/mars/training/data"
mseed_dir_mars_test = "./NasaProject/space_apps_2024_seismic_detection/data/mars/test/data"

csv_path_train = "train_list.csv"
csv_path_test = "test_list.csv"


# In[ ]:


def append_to_csv(csv_path, dir_paths):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['fname'])
        
        for dir_path in dir_paths:
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    if filename.endswith('.mseed'):
                        file_path = os.path.join(root, filename)
                        writer.writerow([file_path])

append_to_csv(csv_path_train, [mseed_dir_lunar_train, mseed_dir_mars_train])
print(f"Training CSV file has been created at: {csv_path_train}")

append_to_csv(csv_path_test, [mseed_dir_lunar_test, mseed_dir_mars_test])
print(f"Testing CSV file has been created at: {csv_path_test}")


# In[ ]:


def normalize_data(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data.reshape(-1, 1))
    return normalized_data, scaler


# In[ ]:


class CNN1DAutoencoder(nn.Module):
    def __init__(self):
        super(CNN1DAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2),  
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=3, stride=1, padding=1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)        
        return x


# In[ ]:


def train_cnn1d_autoencoder(train_data, input_size, epochs=30, learning_rate=0.001, batch_size=1024, save_path="cnn1d_autoencoder.pth"):
    autoencoder = CNN1DAutoencoder().to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        autoencoder = torch.nn.DataParallel(autoencoder)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    print(f"Loaded entire dataset into GPU memory: {train_data.shape}")

    torch.backends.cudnn.benchmark = True

    num_batches = train_data.shape[0] // batch_size
    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0.0
        autoencoder.train()

        for batch_idx in range(num_batches):
            batch_data = train_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            optimizer.zero_grad()

            outputs = autoencoder(batch_data)
            loss = criterion(outputs, batch_data)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress = (batch_idx + 1) / num_batches
            progress_percentage = progress * 100
            progress_bar = f"[{'#' * int(progress * 40)}{'.' * (40 - int(progress * 40))}]"
            sys.stdout.write(f"\rEpoch [{epoch+1}/{epochs}] Progress: {progress_bar} {progress_percentage:.2f}%")
            sys.stdout.flush()

        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - start_time
        print(f"\nEpoch [{epoch+1}/{epochs}] completed, Loss: {avg_loss:.4f}, Time per epoch: {epoch_time:.2f} seconds")

        model_save_path = f"cnn1d_autoencoder_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth"
        torch.save(autoencoder.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    return autoencoder


# In[ ]:


def detect_anomalies_with_scoring(autoencoder, test_data, threshold=0.5, z_thresh=5.85):
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    autoencoder.eval()

    with torch.no_grad():
        reconstructions = autoencoder(test_data)
        reconstruction_error = np.mean(np.abs(test_data.cpu().numpy() - reconstructions.cpu().numpy()), axis=1)

    weighted_error = reconstruction_error * np.where(reconstruction_error > threshold, 2, 1)

    scaler = MinMaxScaler()
    weighted_error_normalized = scaler.fit_transform(weighted_error.reshape(-1, 1)).flatten()

    z_scores = zscore(weighted_error_normalized)

    outliers = np.where(np.abs(z_scores) > z_thresh, -1, 1)

    anomaly_scores = weighted_error_normalized

    return anomaly_scores, outliers


# In[ ]:


def plot_results_with_anomaly_scores(original_data, anomaly_scores, outliers, base_path="C:/Users/jacks/Downloads/NasaProject"):
    plots_dir = os.path.join(base_path, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    time_array = np.arange(len(original_data))

    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_array, original_data, label="Original Signal", alpha=0.7)
    plt.title("Original Signal")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_array, anomaly_scores, label="Anomaly Scores", color="red", alpha=0.7)
    plt.title("Anomaly Scores over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(time_array, original_data, label="Original Signal", alpha=0.7)

    anomaly_times = time_array[outliers == -1]
    anomaly_severity = anomaly_scores[outliers == -1]

    plt.scatter(anomaly_times, original_data[anomaly_times], 
                c=anomaly_severity, cmap='hot', label="Anomalies", marker='x')

    plt.title("Anomalies Detected on Original Signal")
    plt.colorbar(label='Anomaly Severity')
    plt.legend()

    existing_files = os.listdir(plots_dir)
    file_index = len(existing_files) + 1
    save_path = os.path.join(plots_dir, f"anomalies_detected_{file_index}.png")

    plt.savefig(save_path)
    plt.show()


# In[ ]:


def load_and_prepare_training_data(file_list, num_files_to_load=None):
    combined_data = []

    if num_files_to_load:
        file_list = file_list[:num_files_to_load]

    for file_path in file_list:
        print(f'Loading {file_path}')
        stream = obspy.read(file_path)
        trace = stream[0]
        data = trace.data

        normalized_data, _ = normalize_data(data)

        combined_data.append(normalized_data)

    train_data = np.vstack(combined_data)
    train_data = np.expand_dims(train_data, axis=1)

    return train_data


# In[ ]:


def process_mseed_file(file_path, autoencoder):
    stream = obspy.read(file_path)
    trace = stream[0]
    data = trace.data

    normalized_data, scaler = normalize_data(data)

    normalized_data = np.expand_dims(normalized_data, axis=1)
    normalized_data_tensor = torch.tensor(normalized_data, dtype=torch.float32).to(device)

    anomaly_scores, outliers = detect_anomalies_with_scoring(autoencoder, normalized_data_tensor)

    anomaly_scores_denormalized = scaler.inverse_transform(anomaly_scores.reshape(-1, 1))

    min_len = min(len(data), len(anomaly_scores_denormalized))
    data = data[:min_len]
    anomaly_scores_denormalized = anomaly_scores_denormalized[:min_len]
    outliers = outliers[:min_len]

    plot_results_with_anomaly_scores(data, anomaly_scores_denormalized, outliers)

    return anomaly_scores_denormalized, outliers


# In[ ]:


def calculate_anomaly_percentage(outliers):
    windows = np.array_split(outliers, 10000)
    anomalous_windows = 0

    for window in windows:
        anomalies_in_window = np.sum(window == -1)
        if anomalies_in_window > 1:
            anomalous_windows += 1

    anomaly_percentage = (anomalous_windows / 1000) * 100
    return anomaly_percentage


# In[ ]:


csv_file = r"train_list.csv"
file_list = pd.read_csv(csv_file)['fname']

train_data = load_and_prepare_training_data(file_list, num_files_to_load=77)
print(f'Total number of data points in the training data: {train_data.shape[0]}')


# In[ ]:


autoencoder = train_cnn1d_autoencoder(train_data, input_size=1, epochs=2, batch_size=65536)


# In[ ]:


torch.cuda.empty_cache()

csv_file = "test_list.csv"
file_list = pd.read_csv(csv_file)['fname']

test_data = load_and_prepare_training_data(file_list, num_files_to_load=77)
print(f'Total number of data points in the test data: {test_data.shape[0]}')

autoencoder.eval()


# In[ ]:


total_anomaly_percentage = 0
file_count = len(file_list)

for file_path in file_list:
    torch.cuda.empty_cache()
    print(f'Processing {file_path}')
    
    reconstruction_error, outliers = process_mseed_file(file_path, autoencoder)
    
    anomaly_percentage = calculate_anomaly_percentage(outliers)
    total_anomaly_percentage += anomaly_percentage
    
    print(f'Outliers: {outliers}')
    print(f'Reconstruction Error: {reconstruction_error}')
    print(f'Anomaly Percentage: {anomaly_percentage:.2f}%')

average_anomaly_percentage = total_anomaly_percentage / file_count
print(f'Overall Average Anomaly Percentage: {average_anomaly_percentage:.2f}%')

