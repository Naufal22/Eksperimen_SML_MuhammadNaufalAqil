from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Gauge, Summary
import mlflow.sklearn
import threading
import time
import random
import os

app = Flask(__name__)

REQUEST_COUNT = Counter('request_count_total', 'Total Request')
CHURN_COUNT = Counter('churn_pred_count', 'Prediksi Churn', ['prediction'])
LATENCY = Summary('request_latency_seconds', 'Latency')
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Dummy')
MEM_USAGE = Gauge('system_memory_usage', 'Memory Usage Dummy')
ACCURACY = Gauge('model_accuracy', 'Akurasi Model Dummy')
ACTIVE_USERS = Gauge('active_users', 'User Aktif')
ERROR_RATE = Gauge('error_rate', 'Error Rate')
REVENUE = Gauge('daily_revenue', 'Revenue Dummy')
SATISFACTION = Gauge('customer_satisfaction', 'CSAT Score')

# Load Model
try:
    model = mlflow.sklearn.load_model("model_rf_local")
    print("✅ Model dimuat!")
except:
    model = None

# IZINKAN GET
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    REQUEST_COUNT.inc()
    
    # Simulasi Prediksi
    pred = random.choice([0, 1])
    label = "Yes" if pred == 1 else "No"
    CHURN_COUNT.labels(prediction=label).inc()
    
    return jsonify({
        "status": "success",
        "churn_prediction": label
    })


def background_metrics():
    print("Hantu metrics mulai bekerja...")
    while True:
        CPU_USAGE.set(random.uniform(20, 80))     # Isi CPU
        ACTIVE_USERS.set(random.randint(100, 500)) # Isi User Aktif
        ACCURACY.set(random.uniform(0.80, 0.95))   # Isi Akurasi
        time.sleep(3)

if __name__ == '__main__':
    # PORT 8000
    start_http_server(8000) 
    
    threading.Thread(target=background_metrics).start()
    
    # Jalanin App di 5000
    app.run(host='0.0.0.0', port=5000)