import pandas as pd
import os
import shutil
import mlflow
import mlflow.sklearn
import dagshub 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. KONFIGURASI DAGSHUB (FIX) ---


os.environ["DAGSHUB_USER_TOKEN"] = "ce3238b3a7c35717e39d5ea8b431f6ddebfc92c6"

# Inisialisasi Project
dagshub.init(repo_owner='Naufal22', repo_name='Eksperimen_SML_MuhammadNaufalAqil', mlflow=True)

def main():
    print("Memulai Training Model...")

    # 2. LOAD DATA BERSIH
    train_path = '../preprocessing/data_processed/train_clean.csv'
    test_path = '../preprocessing/data_processed/test_clean.csv'

    if not os.path.exists(train_path):
        print("Error: File data tidak ditemukan. Jalankan preprocessing dulu!")
        return

    print("Membaca data...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop('Churn', axis=1)
    y_train = train_data['Churn']
    X_test = test_data.drop('Churn', axis=1)
    y_test = test_data['Churn']

    # 3. SETTING EKSPERIMEN MLFLOW
    mlflow.set_experiment("Eksperimen_Telco_Churn")

    with mlflow.start_run():
        print("Sedang melatih model...")
        
        n_estimators = 100
        max_depth = 10
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {acc:.4f}")

        # 4. LOGGING KE DAGSHUB
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)


        print("Mengupload Model ke DagsHub (Metode Artifact)...")
        
        local_model_path = "model_rf_local"
        
        # Bersihkan jika ada sisa folder lama
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)

        # 1. Simpan struktur MLflow di lokal
        mlflow.sklearn.save_model(model, local_model_path)

        # 2. Upload folder tersebut sebagai Artifact ke DagsHub

        mlflow.log_artifacts(local_model_path, artifact_path="model_rf")

        # 3. Hapus folder lokal
        shutil.rmtree(local_model_path)

        # 5. BUAT & LOG ARTEFAK GAMBAR
        print("Membuat Confusion Matrix...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Acc: {acc:.2f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.savefig("confusion_matrix.png")
        plt.close()

        mlflow.log_artifact("confusion_matrix.png")
        os.remove("confusion_matrix.png")

        print("Selesai! Cek dashboard DagsHub")

if __name__ == "__main__":
    main()