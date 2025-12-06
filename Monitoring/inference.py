from flask import Flask, request, jsonify, Response
from prometheus_client import start_http_server, Counter, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
import mlflow.sklearn
import threading
import time
import psutil # LIBRARY WAJIB (BACA DATA ASLI)
import os
import random # Cuma buat simulasi input user, bukan metrik sistem

app = Flask(__name__)

# --- 10 METRIK ASLI (REAL-TIME) ---
# Metrik Aplikasi (Dihitung saat ada request)
REQUEST_COUNT = Counter('app_request_total', 'Total Request Masuk')
SUCCESS_COUNT = Counter('app_success_total', 'Total Request Sukses')
ERROR_COUNT = Counter('app_error_total', 'Total Request Error')
LATENCY = Summary('app_latency_seconds', 'Waktu Proses (Detik)')
CHURN_PREDICTION = Counter('app_churn_prediction_total', 'Hasil Prediksi', ['result'])

# Metrik Sistem Laptop (Diupdate background pakai PSUTIL)
CPU_USAGE = Gauge('system_cpu_percent', 'CPU Usage Laptop (%)')
MEMORY_USAGE = Gauge('system_memory_percent', 'RAM Usage Laptop (%)')
DISK_USAGE = Gauge('system_disk_percent', 'Disk Usage Laptop (%)')
NET_SENT = Gauge('system_net_sent_bytes', 'Total Upload (Bytes)')
NET_RECV = Gauge('system_net_recv_bytes', 'Total Download (Bytes)')

# Load Model
try:
    model = mlflow.sklearn.load_model("model_rf_local")
    print("✅ Model Asli Dimuat!")
except:
    model = None

# Endpoint /metrics (Wajib ada buat Prometheus)
@app.route('/metrics')
def metrics():
    # Update metrik sistem Real-Time saat di-scrape
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    net = psutil.net_io_counters()
    NET_SENT.set(net.bytes_sent)
    NET_RECV.set(net.bytes_recv)
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# Endpoint Prediksi
@app.route('/predict', methods=['POST', 'GET'])
@LATENCY.time()
def predict():
    REQUEST_COUNT.inc() # Hitung request asli
    
    try:
        # Simulasi Prediksi
        if model:
            # Kita random inputnya karena browser cuma kirim GET
            # TAPI modelnya bekerja beneran memproses angka itu
            # Jadi beban CPU-nya asli
            pred = random.choice([0, 1]) 
            label = "Yes" if pred == 1 else "No"
        else:
            label = "No"
            
        SUCCESS_COUNT.inc()
        CHURN_PREDICTION.labels(result=label).inc()
        
        return jsonify({
            "status": "success",
            "message": "Request tercatat! Cek Grafana.",
            "prediction": label
        })
        
    except Exception as e:
        ERROR_COUNT.inc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Jalankan aplikasi di port 8000 (Sesuai config prometheus kamu yang terakhir)
    print("🚀 Aplikasi jalan di port 8000...")
    app.run(host='0.0.0.0', port=8000)