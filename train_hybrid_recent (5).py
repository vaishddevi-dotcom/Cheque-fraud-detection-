# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 21:37:03 2025

@author: Z0142995
"""
import os 
import sys
import subprocess
import zipfile
import urllib.request
import shutil
WHEEL_BASE_DIR = "/tmp/wheels"
WHEEL_ZIP_PATH = "/tmp/wheels.zip"

WHEELS_ZIP_URL = os.environ.get("WHEELS_ZIP_URL")

if not WHEELS_ZIP_URL:
    raise RuntimeError("WHEELS_ZIP_URL environment variable is not set")

os.makedirs(WHEEL_BASE_DIR, exist_ok=True)
print("Downloading wheels.zip from Blob Storage...")
urllib.request.urlretrieve(WHEELS_ZIP_URL, WHEEL_ZIP_PATH)
print("Extracting wheels.zip...")
with zipfile.ZipFile(WHEEL_ZIP_PATH, "r") as z:
    z.extractall(WHEEL_BASE_DIR)
    
subprocess.check_call(["ls", "-R", "/tmp/wheels"])

print("Installing NON-TensorFlow packages (offline)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--no-index",
    "--find-links", "/tmp/wheels/wheels",
    "numpy==1.26.1",
    "pandas",
    "scipy",
    "scikit-learn",
    "requests",
    "azure-storage-blob",
    "kiwisolver",
    "contourpy",
    "Pillow",
    "pyparsing",
    "fonttools",
    "cycler",
    "absl-py",
    "protobuf==3.20.3",
    "googleapis-common-protos==1.63.2",
    "wrapt",
    "opt_einsum==3.3.0",
    "gast==0.4.0",
    "azure-core",
    "typing-extensions",
    "isodate",
    "cryptography",
    "cffi",
    "pycparser",
    "six",
    "certifi",
    "charset-normalizer",
    "idna",
    "urllib3"

])


subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--no-index",
    "--find-links", "/tmp/wheels/wheels",
    "matplotlib",
    "--no-deps"
])


print("Installing TensorFlow dependencies (offline)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--no-index",
    "--find-links", "/tmp/wheels/wheels",

    # TensorFlow runtime deps
    "astunparse",
    "flatbuffers",
    "grpcio",
    "h5py",
    "keras==2.15.0",
    "tf-keras==2.15.1",
    "libclang",
    "ml-dtypes",
    "termcolor",
    "tensorflow-io-gcs-filesystem",
    #"tensorflow-cpu==2.15.1",
    "tensorflow-estimator==2.15.0",
    "tensorboard==2.15.2",
    "tensorboard-data-server==0.7.2",
    "protobuf==3.20.3",
    # shared deps TF expects
    "six",
    "typing-extensions",
    "wrapt"
])



print("Installing TensorFlow (offline, no deps)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--no-index",
    "--find-links", "/tmp/wheels/wheels",
    "tensorflow-cpu==2.15.1",
    "--no-deps"
])


print("Offline installation completed successfully.")

import requests
from requests.auth import HTTPBasicAuth
import base64
import json
import logging
import xml.etree.ElementTree as ET
import io
from io import BytesIO

import pandas as pd
from azure.storage.blob import BlobServiceClient
import os
import threading
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle, zipfile
from pathlib import Path
from io import StringIO
import argparse
import base64
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_zip_csv(zip_file):
    import zipfile, io
    with zipfile.ZipFile(zip_file) as z:
        file_list = z.namelist()
        df_list = [pd.read_csv(z.open(file)) for file in file_list if file.endswith('.csv')]
        return pd.concat(df_list, ignore_index=True)
    
def load_zip_csv1(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        dfs = [pd.read_csv(zf.open(name)) for name in zf.namelist()]
    return pd.concat(dfs, ignore_index=True)
        

def rc_model(t, R, C, T0, Tinf):
    return T0 + (Tinf - T0) * (1 - np.exp(-t / (R * C)))
    
def read_txt_file(file_path):
    """
    Reads the contents of a .txt file and returns it as a string.
    
    :param file_path: Path to the .txt file
    :return: Contents of the file as a string
    """
    #file_path='SAS_token.txt'
    with open(file_path, 'r') as file:
        sas_token = file.read()
    return sas_token

def remove_trailing_number_pattern(df):
    """
    Removes trailing pattern like '_1_', '_99_', etc. from ALL column names in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: Modified DataFrame with cleaned column names.
    """
    df.columns = df.columns.str.replace(r'__signal_\d+_$', '', regex=True)
    return df

def list_blobs(account_name, container_name, sas_token, blob_type='.csv'):
    base_url = f"https://{account_name}.blob.core.windows.net/{container_name}"
    params = {
        'restype': 'container',
        'comp': 'list',
    }
    headers = {
        'x-ms-version': '2020-10-02'
    }

    blob_names = []
    next_marker = ''

    while True:
        # If there's a continuation token, include it in the parameters
        if next_marker:
            params['marker'] = next_marker

        # Make the request
        response = requests.get(f"{base_url}?{sas_token}", headers=headers, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        # Parse XML
        root = ET.fromstring(response.content)

        for blob in root.findall('.//Blob'):
            blob_name = blob.find('Name').text
            if blob_name.endswith(blob_type):
                blob_names.append(blob_name)

        # Get the next marker, if any
        next_marker_elem = root.find('.//NextMarker')
        if next_marker_elem is not None and next_marker_elem.text:
            next_marker = next_marker_elem.text
        else:
            break

    return blob_names


def download_blob_to_dataframe(blob_url, sas_token, container_name, blob_name):
    """
    Download a blob from Azure Storage (CSV or Parquet) and return it as a Pandas DataFrame.
    """
    # Create BlobServiceClient
    blob_service_client = BlobServiceClient(account_url=blob_url, credential=sas_token)

    # Get the blob client
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Download the blob content as a stream
    blob_stream = blob_client.download_blob().readall()
    buffer = io.BytesIO(blob_stream)

    # Detect file type
    file_ext = blob_name.lower().split('.')[-1]

    if file_ext == "csv":
        df = pd.read_csv(buffer)
    elif file_ext == "parquet":
        df = pd.read_parquet(buffer)
    else:
        raise ValueError(f"Unsupported file type: .{file_ext}")

    return df


def downloadcsv(project, test_list, testrun_id, sas_token):
    account_name = "azp229sa"
    container_name = "azp229sacontainer"
    blob_type = '.csv'
    blob_url = "https://azp229sa.blob.core.windows.net"

    # List all csv blobs in the container
    csv_blobs = list_blobs(account_name, container_name, sas_token, blob_type)

    # Build filter pattern based on inputs
    filtered_files = []
    for file in csv_blobs:
        if (
            project in file
            and any(test in file for test in test_list)
            and str(testrun_id) in file
        ):
            if not (file.endswith("mSimulationRawData_max.csv") or file.endswith("mSimulationDamageData.csv")):
                filtered_files.append(file)

    print("Filtered files:", filtered_files)

    combined_csv_data = None

    for file in filtered_files:
        file_path = f"{container_name}/{file}"
        container_path = file_path.rsplit('/', 1)[0]
        blob_name = os.path.basename(file)
        folder_two_up = file_path.rsplit('/', 2)[0]
        foldername = os.path.basename(folder_two_up)

        csv_data = download_blob_to_dataframe(blob_url, sas_token, container_path, blob_name)
        csv_data = pd.DataFrame(csv_data)

        time_col = next((col for col in csv_data.columns if col.lower() == 'time'), None)
        if time_col and time_col != 'Time':
            csv_data = csv_data.rename(columns={time_col: 'Time'})
        if 'Time' not in csv_data.columns:
            print(f"⚠️ Skipping file (no Time column): {file}")
            continue

        # Rename other columns to ensure uniqueness if needed
        csv_data = csv_data.rename(columns=lambda x: f"{x}" if x != 'Time' else x)

        # Merge with combined DataFrame
        if combined_csv_data is None:
            combined_csv_data = csv_data
        else:
            try:
                combined_csv_data = pd.merge(combined_csv_data, csv_data, on='Time', how='outer')
            except Exception as e:
                print(f"⚠️ Skipping merge for {file}: {e}")
                continue
        print(f'Processed: {foldername}_{blob_name}')

    print("Preparation complete!")
    if combined_csv_data is None:
        raise Exception("❌ No valid CSV files with Time column found")

    return combined_csv_data.to_csv(index=False)

   
def create_windowed_input(X, window_size):
    X_seq = []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size].flatten())
    return np.array(X_seq)

def create_windowed_data(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size].flatten())  # shape: [window_size * features]
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)

def create_sparse_windowed_data(X, y, window_size=5, step=100):
    X_seq, y_seq = [], []
    total_window = step * (window_size - 1)

    for i in range(total_window, len(X)):
        window = [X[i - j * step] for j in reversed(range(window_size))]
        X_seq.append(np.concatenate(window))
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)

def create_sparse_windowed_input(X, window_size=5, step=100):
    X_seq = []
    total_window = step * (window_size - 1)

    for i in range(total_window, len(X)):
        window = [X[i - j * step] for j in reversed(range(window_size))]
        X_seq.append(np.concatenate(window))
    
    return np.array(X_seq)

def apply_feature_engineering(train_df, test_df):
    """
    Applies feature engineering on training and testing dataframes.
    Adds derivatives, rolling means, interaction terms, and encodes categoricals.
    Ensures both dataframes stay aligned.
    """
    import pandas as pd
    import numpy as np

    def add_common_features(df):
        # Derivative features (first-order)
        df['dIdt'] = df['inBat_potDc'].diff().fillna(0) / df['Time'].diff().fillna(1)
        df['dTamb'] = df['envAmb_tmp'].diff().fillna(0) / df['Time'].diff().fillna(1)
        df['dTcool'] = df['inClnt_tmpF'].diff().fillna(0) / df['Time'].diff().fillna(1)

        # Rolling mean (smoothing context)
        df['inBat_potDc_mean5'] = df['inBat_potDc'].rolling(window=5, min_periods=1).mean()
        df['envAmb_tmp_mean5'] = df['envAmb_tmp'].rolling(window=5, min_periods=1).mean()
        df['inClnt_tmpF_mean5'] = df['inClnt_tmpF'].rolling(window=5, min_periods=1).mean()

        # Interaction terms
        df['power_loss'] = df['inBat_potDc'] ** 2
        df['I_Tcool'] = df['inBat_potDc'] * df['inClnt_tmpF']
        df['I_AmbDelta'] = df['inBat_potDc'] * (df['inClnt_tmpF'] - df['envAmb_tmp'])

        return df

    # Apply to both datasets
    train_df = add_common_features(train_df.copy())
    test_df = add_common_features(test_df.copy())

    # Handle categorical one-hot encoding (example: styInv_mod)
    cat_col = 'styInv_mod'
    if cat_col in train_df.columns:
        train_df = pd.get_dummies(train_df, columns=[cat_col], prefix=cat_col, drop_first=True)
        test_df = pd.get_dummies(test_df, columns=[cat_col], prefix=cat_col, drop_first=True)

        # Align columns
        train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

    return train_df, test_df


def upload_folder_to_blob_storage(sas_token, account_url, container_name, folder_path, destination_folder):

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found!")
        return

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:

            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, folder_path).replace("\\", "/")

            blob_name = f"{destination_folder}/{relative_path}"

            blob_url = f"{account_url}/{container_name}/{blob_name}?{sas_token}"

            with open(file_path, "rb") as data:
                file_data = data.read()

            headers = {
                "x-ms-blob-type": "BlockBlob",
                "Content-Length": str(len(file_data))
            }

            response = requests.put(blob_url, headers=headers, data=file_data)

            if response.status_code == 201:
                print(f"Uploaded {blob_name}")
            else:
                print(f"Failed uploading {blob_name}: {response.text}")

account_url = "https://azp229sa.blob.core.windows.net"
container_name = "azp229sacontainer"
# NEW folder for ML models
destination_folder = "MLModels/HybridModel"
folder_path = "exported_model"
# Read SAS token
#with open("SAS_token.txt", "r") as f:
    #sas_token = f.read().strip()

sas_token = os.environ.get("SAS_TOKEN")

if not sas_token:
    raise Exception("SAS_TOKEN not found in environment")

# Clean token if needed
if sas_token.startswith("?"):
    sas_token = sas_token[1:]

print("SAS TOKEN (first 20 chars):", sas_token[:20])


def train_hybrid_model_sequential(selected_project, selected_test, selected_testrun_id, test_zip_path,
                                  sas_token, model_config, epochs=5, batch_size=256, window_size=20,
                                  time_gap_sec=10, use_manual_rc=False):
                                                                      
    #config_text = os.environ.get("AZUREML_INPUT_model_config")
    


    #if config_text and config_text.strip():


    if model_config:
        config = model_config

    else: 
        config_path = Path("exported_model/pipeline_config.json")
        if not config_path.exists():
            raise RuntimeError(

            "No model_config provided and no saved pipeline_config.json found"
        )
        with open(config_path) as f:
            config = json.load(f)

    dc_col = config["dc_col"]
    amb_col = config["amb_col"]
    cool_col = config["cool_col"]
    time_col = config["time_col"]
    y_cols = config["y_cols"]
    extra_input = config["extra_input"]
    x_cols = [dc_col, amb_col, cool_col] + extra_input

    # --- Setup ---
    #dc_col = 'sty_curRms_acCond'
    #amb_col = 'envAmb_tmp'
    #cool_col = 'inClnt_tmpF'
    #time_col = 'Time'
    #y_cols = ["PCBC41502", "PCBC41503", "PCBC41504", "PCBC41505",
              #"PCBC41506", "PCBC41507", "PCBC41508", "PCBC41509",
              #"PCBC41331", "PCBC41336", "PCBC41226", "PCBC41227",
              #"DCLCap1", "DCLCap2", "DCLCap3", "DCLCap4",
              #"YCap1", "YCap2", "YCap3", "YCap4"]

    #extra_input = ['inBat_potDc', 'styInv_freq_switch', 'inClnt_volFlow', 'styInv_mod', 'styDu_cosphi', 'styDu_rotFrq_elec']
     #= [dc_col, amb_col, cool_col] + extra_input

    step = int(time_gap_sec / 0.1)
    total_window = step * (window_size - 1)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    model = None
    rc_params = {}

    print(f"\nStarting training over {epochs} epochs across {len(selected_test)} test files")

    local_data_dir = Path("local_data")
    local_data_dir.mkdir(exist_ok=True)

    # --- Pre-download and store all required test CSVs ---
    print("\nDownloading and caching test data locally...")

    test_data_paths = []
    for test_name, test_run_id in zip(selected_test, selected_testrun_id):
        local_csv_path = local_data_dir / f"{test_name}.csv"
        test_data_paths.append(local_csv_path)

        if not local_csv_path.exists():
            try:
                csv_str = downloadcsv(selected_project, test_name, test_run_id, sas_token)
                with open(local_csv_path, "w", encoding="utf-8") as f:
                    f.write(csv_str)
                print(f"Downloaded and saved: {test_name}")
            except Exception as e:
                raise Exception(f"Download failed for {test_name}: {e}")
        else:
            print(f"Found cached file: {test_name}")

    print(f"\nStarting training over {epochs} epochs")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for file_idx, (test_name, csv_path) in enumerate(zip(selected_test, test_data_paths)):
            try:
                df = pd.read_csv(csv_path)
                df = remove_trailing_number_pattern(df)
            except Exception as e:
                print(f"❌ Failed to load cached CSV {csv_path.name}: {e}")
                continue

            numeric_cols = df.select_dtypes(include='number').columns.tolist()

            # RC Model Fit
            T_RC_train = []
            for col in y_cols:
                y = df[col].values
                t = df[time_col].values
                T0, Tinf = y[0], y[-1]

                if use_manual_rc:
                    R = float(input(f"Enter R for {col}: "))
                    C = float(input(f"Enter C for {col}: "))
                    T0_fit, Tinf_fit = T0, Tinf
                else:
                    try:
                        popt, _ = curve_fit(rc_model, t, y, p0=[1.0, 1.0, T0, Tinf],
                                            bounds=([0, 0, T0 - 10, T0], [np.inf, np.inf, T0 + 10, Tinf + 20]), maxfev=10000)
                        R, C, T0_fit, Tinf_fit = popt
                        print(f"[{test_name}] {col} RC: R={R:.4f}, C={C:.4f}")
                    except Exception:
                        R, C, T0_fit, Tinf_fit = 1.0, 1.0, T0, Tinf
                        print(f"[{test_name}] {col} RC fit failed. Using default.")

                rc_params[col] = (R, C, T0_fit, Tinf_fit)
                T_RC_train.append(rc_model(t, R, C, T0_fit, Tinf_fit))

            # Compute Residuals
            T_RC_train = np.vstack(T_RC_train).T
            y_actual = df[y_cols].values
            residuals = y_actual - T_RC_train
            smoothed_residuals = gaussian_filter1d(residuals, sigma=2, axis=0)

            # Prepare Input/Output Sequences
            X_raw = df[x_cols].values
            X_seq, y_seq = create_sparse_windowed_data(X_raw, smoothed_residuals, window_size, step)

            if epoch == 0 and file_idx == 0:
                X_scaled = x_scaler.fit_transform(X_seq)
                y_scaled = y_scaler.fit_transform(y_seq)

                input_dim = X_scaled.shape[1]
                model = Sequential([
                    Dense(256, input_dim=input_dim, activation='relu'),
                    Dropout(0.3),
                    Dense(128, activation='relu'),
                    Dense(len(y_cols))
                ])
                model.compile(optimizer='adam', loss='mse')
                print("Model and scalers initialized.")
            else:
                X_scaled = x_scaler.transform(X_seq)
                y_scaled = y_scaler.transform(y_seq)

            print(f"Training on file: {test_name}")
            model.fit(X_scaled, y_scaled, epochs=1, batch_size=batch_size, verbose=0)

    print("\nNeural network training complete.")

    # --- Save Model and Config ---
    export_dir = Path("exported_model")
    export_dir.mkdir(exist_ok=True)
    model.save(export_dir / "hybrid_model.h5")
    with open(export_dir / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open(export_dir / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)
    with open(export_dir / "config.pkl", "wb") as f:
        pickle.dump({
            "x_cols": x_cols,
            "y_cols": y_cols,
            "rc_params": {k: list(v) for k, v in rc_params.items()},
            "window_size": window_size,
            "step": step,
            "time_col": time_col
        }, f)
    with open(export_dir / "pipeline_config.json", "w") as f:
        json.dump(config, f, indent=4)
    print("\nModel and config exported to 'exported_model/'")



def test_hybrid_model(test_zip_path, sas_token):
    print("\n==============================")
    print("STARTING TESTING PHASE")
    print("==============================")
    sys.stdout.flush()

    # ✅ READ-ONLY model directory
    model_dir = Path("exported_model")

    # ✅ SEPARATE test output directory (NEW)
    test_output_dir = Path("test_outputs")
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Load trained artifacts (READ ONLY)
    model = load_model(model_dir / "hybrid_model.h5")
    with open(model_dir / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(model_dir / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    with open(model_dir / "config.pkl", "rb") as f:
        cfg = pickle.load(f)

    x_cols = cfg["x_cols"]
    y_cols = cfg["y_cols"]
    rc_params = cfg["rc_params"]
    window_size = cfg["window_size"]
    time_col = cfg["time_col"]
    step = cfg["step"]

    # ✅ TEMP test data folder
    test_dir = Path("test_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(test_zip_path, "r") as z:
        z.extractall(test_dir)

    csv_files = list(test_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError("❌ No CSV files found in test ZIP")

    rmse_all, mae_all = [], []
    metrics = {}

    # ✅ predictions go OUTSIDE exported_model
    pred_dir = test_output_dir / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_files:
        print(f"\nTesting on: {csv_path.name}")
        sys.stdout.flush()

        df = pd.read_csv(csv_path)
        df = remove_trailing_number_pattern(df)

        missing = (set(x_cols) | set(y_cols) | {time_col}) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

        t = df[time_col].values

        # --- RC prediction ---
        T_RC = []
        for col in y_cols:
            R, C, T0, Tinf = rc_params[col]
            T_RC.append(rc_model(t, R, C, T0, Tinf))
        T_RC = np.vstack(T_RC).T

        # --- NN residual prediction ---
        X_raw = df[x_cols].values
        X_seq, _ = create_sparse_windowed_data(
            X_raw,
            np.zeros((len(X_raw), len(y_cols))),
            window_size,
            step
        )

        X_scaled = x_scaler.transform(X_seq)
        res_scaled = model.predict(X_scaled, verbose=0)
        residual_pred = y_scaler.inverse_transform(res_scaled)

        # --- Final output ---
        y_true = df[y_cols].values[-len(residual_pred):]
        y_pred = T_RC[-len(residual_pred):] + residual_pred

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        metrics[csv_path.name] = {"rmse": rmse, "mae": mae}
        rmse_all.append(rmse)
        mae_all.append(mae)

        out = pd.DataFrame({time_col: t[-len(residual_pred):]})
        for i, c in enumerate(y_cols):
            out[f"actual_{c}"] = y_true[:, i]
            out[f"predicted_{c}"] = y_pred[:, i]

        out.to_csv(pred_dir / f"{csv_path.stem}_predictions.csv", index=False)

    metrics["average"] = {
        "rmse": float(np.mean(rmse_all)),
        "mae": float(np.mean(mae_all))
    }

    # ✅ metrics saved OUTSIDE exported_model
    with open(test_output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(test_output_dir / "test_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "rmse", "mae"])
        for k, v in metrics.items():
            if k != "average":
                w.writerow([k, v["rmse"], v["mae"]])

    print("Testing complete. Results saved to test_outputs/")
    sys.stdout.flush()



account_name = "azp229sa"
container_name = "azp229sacontainer"
blob_type = '.csv'
#sas_token = read_txt_file('SAS_token.txt')
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--test_zip", required=True)

parser.add_argument(
    "--model_config",
    type=str,
    required=True,
    help="Base64-encoded model configuration JSON"
)


args = parser.parse_args()
decoded = base64.b64decode(args.data).decode()
data = json.loads(decoded)
decoded_config = base64.b64decode(args.model_config).decode()
model_config = json.loads(decoded_config)
config_df = pd.DataFrame(data)

# Extract lists from the DataFrame
selected_test = config_df['selected_test'].tolist()
selected_testrun_id = config_df['selected_testrun_id'].astype(str).tolist()

# Static project and test zip path (can also be parameterized)
selected_project = "TwinOps"
test_zip_path = args.test_zip

# Call your training function
train_hybrid_model_sequential(
    selected_project=selected_project,
    selected_test=selected_test,
    selected_testrun_id=selected_testrun_id,
    test_zip_path=test_zip_path,
    sas_token=sas_token,
    model_config=model_config,
    epochs=2,
    batch_size=256,
    window_size=20,
    time_gap_sec=10,
    use_manual_rc=False
)


# ----- TESTING -----
test_hybrid_model(
    test_zip_path=test_zip_path,
    sas_token=sas_token
)

print("Uploading exported_model folder to Blob Storage...")

upload_folder_to_blob_storage(
    sas_token,
    account_url,
    container_name,
    folder_path,
    destination_folder
)

