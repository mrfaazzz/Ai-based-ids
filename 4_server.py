
import socket
import joblib
import json
import numpy as np
import sys
import os
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

HOST = '127.0.0.1'
PORT = 9999
MODEL_FILE = 'ids_model.joblib'


def start_server():
    print(f"{Style.BRIGHT}[INIT] Starting AI-Based IDS Server...")

    if not os.path.exists(MODEL_FILE):
        print(f"{Fore.RED}[ERROR] Model '{MODEL_FILE}' not found. Run '2_train_ml.py' first.")
        sys.exit(1)

    print(f" Loading Model: {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)
    print(f" Model loaded successfully.")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)

    print("-" * 60)
    print(f" Server is online at {HOST}:{PORT}")
    print(f" Awaiting network traffic...")
    print("-" * 60)

    while True:
        try:
            client_conn, client_addr = server_socket.accept()
            print(f"{Fore.YELLOW}[CONNECTION]{Style.RESET_ALL} Connected to  {client_addr}")

            while True:
                data = client_conn.recv(4096).decode('utf-8')
                if not data:
                    break

                try:
                    features_list = json.loads(data)
                    features = np.array(features_list).reshape(1, -1)

                    prediction = model.predict(features)[0]
                    timestamp = datetime.now().strftime("%H:%M:%S")

                    if prediction == 1:
                        print(
                            f"{Style.BRIGHT}[{timestamp}] {Fore.RED}ALERT: MALICIOUS TRAFFIC DETECTED {Style.RESET_ALL}(Source: {client_addr[0]})")
                        response = "ALERT_BLOCK"
                    else:
                        print(f"[{timestamp}] {Fore.GREEN}MONITOR: Normal Traffic Flow")
                        response = "SAFE_PASS"

                    client_conn.send(response.encode('utf-8'))

                except json.JSONDecodeError:
                    print(f"{Fore.RED}[ERROR] Received malformed data packet.")
                    break

            client_conn.close()
            print(f"{Fore.YELLOW}[DISCONNECT] Connection closed.")
            print("-" * 60)

        except KeyboardInterrupt:
            print(f"\n{Fore.CYAN}[SHUTDOWN] Stopping server...")
            break
        except Exception as e:
            print(f"{Fore.RED}[ERROR] {e}")


if __name__ == "__main__":
    start_server()