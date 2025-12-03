import pandas as pd
import numpy as np
import os
import zipfile
import gdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- KONFIGURASI ---
# Ganti dengan ID File Google Drive kamu (ambil ID-nya saja)
GDRIVE_FILE_ID = '1THQQqK-0iZ9rLsSzbktliPy7FhUtyRH-' 

DATASET_ZIP = 'data_telco.zip'
CSV_FILENAME = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
OUTPUT_DIR = 'data_processed'

def main():
    print("🚀 Memulai Proses Otomatisasi Preprocessing...")

    # 1. LOAD DATA (Download otomatis jika belum ada)
    # Cek apakah file CSV sudah ada
    if not os.path.exists(CSV_FILENAME):
        print(f" File {CSV_FILENAME} tidak ditemukan.")
        
        # Cek apakah file ZIP ada, jika tidak, DOWNLOAD pakai gdown
        if not os.path.exists(DATASET_ZIP):
            print(f" Mendownload {DATASET_ZIP} dari Google Drive...")
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            gdown.download(url, DATASET_ZIP, quiet=False)
        
        # Ekstrak ZIP
        if os.path.exists(DATASET_ZIP):
            print(f" Mengekstrak {DATASET_ZIP}...")
            with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
                zip_ref.extractall('.')
        else:
            print(" Gagal mendownload dataset. Periksa ID Google Drive!")
            return

    print(f" Membaca dataset: {CSV_FILENAME}")
    try:
        df = pd.read_csv(CSV_FILENAME)
    except FileNotFoundError:
        print(f" Error: Gagal membaca {CSV_FILENAME}. Cek hasil ekstraksi!")
        return

    # 2. CLEANING DATA
    print(" Membersihkan data...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    print(f"   - Menghapus {initial_rows - df.shape[0]} baris error (NaN).")

    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # 3. ENCODING
    print(" Melakukan Encoding...")
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])
    df_clean = pd.get_dummies(df, drop_first=True)
    
    # 4. SPLITTING
    print(" Membagi Data (Train 80% - Test 20%)...")
    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. SCALING
    print(" Melakukan Scaling...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    y_train_final = pd.DataFrame(y_train, columns=['Churn']).reset_index(drop=True)
    y_test_final = pd.DataFrame(y_test, columns=['Churn']).reset_index(drop=True)

    train_data = pd.concat([X_train_final, y_train_final], axis=1)
    test_data = pd.concat([X_test_final, y_test_final], axis=1)

    # 6. SAVE DATA
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    train_path = os.path.join(OUTPUT_DIR, 'train_clean.csv')
    test_path = os.path.join(OUTPUT_DIR, 'test_clean.csv')

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"\n SUKSES! Data bersih tersimpan di:")
    print(f"   - {train_path}")
    print(f"   - {test_path}")

if __name__ == "__main__":
    main()