import socket
import joblib
import json
import numpy as np
import sys
import os
from colorama import init, Fore, Style
from datetime import datetime

# 1. SETUP VISUALS
init(autoreset=True)  # Makes colors work on Windows

# 2. CONFIGURATION
HOST = '127.0.0.1'
PORT = 9999
MODEL_FILE = 'ids_model.joblib'

# 3. LOAD BRAIN
print(f"{Fore.CYAN}[INIT] Loading AI Model from {MODEL_FILE}...")
if not os.path.exists(MODEL_FILE):
    print(f"{Fore.RED}[ERROR] Model file not found! Run '2_train_ml_models.py' first.")
    sys.exit(1)

model = joblib.load(MODEL_FILE)
print(f"{Fore.GREEN}[SUCCESS] AI Model Loaded. System Ready.")

# 4. START SERVER
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# This line fixes the "Port Busy" error by forcing the OS to release the port
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"{Fore.CYAN}[NET] Visual Defense Center listening on {HOST}:{PORT}")
    print("-" * 60)
except OSError:
    print(f"{Fore.RED}[ERROR] Port {PORT} is busy! Please close other terminal windows and try again.")
    sys.exit(1)

# 5. MAIN LOOP
while True:
    try:
        client_socket, addr = server.accept()
        print(f"{Fore.YELLOW}[CONN] New Connection from {addr}")

        while True:
            # Receive data (up to 4KB)
            data = client_socket.recv(4096)
            if not data:
                break  # Client disconnected

            try:
                # Decode data
                json_data = data.decode('utf-8')
                features_list = json.loads(json_data)

                # Convert to numpy array for the AI
                features = np.array(features_list).reshape(1, -1)

                # AI PREDICTION
                prediction = model.predict(features)[0]
                timestamp = datetime.now().strftime("%H:%M:%S")

                if prediction == 1:  # ATTACK
                    print(f"{Style.BRIGHT}{Fore.RED}[{timestamp}] 🚨 ALERT: MALICIOUS TRAFFIC DETECTED!")
                    response = "ALERT_BLOCK"
                else:  # NORMAL
                    print(f"{Fore.GREEN}[{timestamp}] ✅ Normal Traffic")
                    response = "SAFE_PASS"

                # Send reply to client
                client_socket.send(response.encode('utf-8'))

            except Exception as e:
                print(f"{Fore.RED}[ERR] Bad Data Packet: {e}")
                break

        client_socket.close()
        print(f"{Fore.YELLOW}[CONN] Connection Closed.")

    except KeyboardInterrupt:
        print(f"\n{Fore.CYAN}[STOP] Server shutting down...")
        break