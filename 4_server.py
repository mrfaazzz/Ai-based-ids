import socket
import joblib
import numpy as np
import json

print("Loading IDS Model...")
model = joblib.load('ids_model.joblib')
print("Model loaded.")

HOST = '127.0.0.1'  # Localhost
PORT = 9999  # Port to listen on


def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"IDS Server listening on {HOST}:{PORT}...")
    print("Waiting for network traffic...")

    while True:
        try:
            client_conn, client_addr = server_socket.accept()

            data = client_conn.recv(4096).decode('utf-8')

            if not data:
                break

            try:
                features_list = [float(x) for x in data.split(',')]
                features = np.array(features_list).reshape(1, -1)

                prediction = model.predict(features)[0]
                result = "ATTACK DETECTED!" if prediction == 1 else "Normal Traffic"

                response = f"Analysis: {result}"
                client_conn.send(response.encode('utf-8'))

                print(f"Received traffic from {client_addr} -> Verdict: {result}")

            except ValueError:
                print("Error: Received malformed data")
                client_conn.send(b"Error: Invalid data format")

            client_conn.close()

        except KeyboardInterrupt:
            print("\nStopping server...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    start_server()