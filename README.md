# Eksperimen MLOps: Prediksi Churn Pelanggan Telco

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![DagsHub](https://img.shields.io/badge/DagsHub-Integration-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 👤 Penulis
**Muhammad Naufal Aqil**
*Dicoding Submission: Membangun Sistem Machine Learning (MLOps)*

## 📖 Ringkasan Proyek
Repositori ini mencakup tahap **Eksperimentasi dan Pemrosesan Data** dalam siklus hidup MLOps untuk kasus prediksi churn pelanggan telekomunikasi. Tujuan utama proyek ini adalah membangun model Machine Learning yang dapat mengidentifikasi pelanggan yang berisiko berhenti berlangganan, sehingga tim bisnis dapat melakukan retensi proaktif.

Fokus utama pada repositori ini:
- **Otomatisasi Preprocessing** menggunakan GitHub Actions.
- **Pelacakan Eksperimen (Experiment Tracking)** menggunakan MLflow & DagsHub.
- **Pengembangan Model** menggunakan Random Forest Classifier.

## 🚀 Fitur Utama
* **Otomatisasi Preprocessing**
  Data mentah diproses secara otomatis (download, cleaning, encoding, splitting) melalui workflow GitHub Actions setiap kali ada pembaruan kode.
* **Experiment Tracking**
  Seluruh metrik pelatihan (Akurasi, Presisi), parameter model, dan artefak disimpan secara terpusat di DagsHub menggunakan MLflow. Memungkinkan perbandingan performa antar versi model dengan mudah.

## 📊 Tentang Dataset
Dataset yang digunakan adalah **Telco Customer Churn** (sumber: IBM/Kaggle). Dataset ini mencakup informasi:
- **Churn**: Pelanggan yang berhenti dalam bulan terakhir.
- **Layanan**: Telepon, internet, keamanan online, dll.
- **Info Akun**: Tenor, kontrak, metode pembayaran, tagihan bulanan.

## 🛠️ Struktur Proyek
```text
├── .github/workflows/      # CI Pipeline untuk Otomatisasi Preprocessing
├── preprocessing/
│   ├── automate_preprocessing.py  # Script pembersihan & rekayasa fitur
│   └── data_processed/            # Folder output data bersih (Train/Test)
├── Membangun_model/
│   ├── modelling.py        # Script training yang terhubung ke DagsHub
│   └── requirements.txt    # Dependensi library
└── Monitoring/             # Script inferensi lokal & konfigurasi monitoring
````

## 📈 Hasil Eksperimen

  * **Algoritma:** Random Forest Classifier
  * **Metrik:** Dicatat dan dilacak di DagsHub.
  * **Artefak:** Confusion Matrix & Model Pickle tersimpan di cloud storage DagsHub.


> Proyek ini merupakan bagian dari modul "Membangun Sistem Machine Learning" oleh Dicoding Indonesia.

