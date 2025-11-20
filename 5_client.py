import socket
import pandas as pd
import time
import random

SERVER_IP = '127.0.0.1'
SERVER_PORT = 9999
DATA_FILE = 'clean_test.csv'


def simulate_traffic():
    print(f"Loading traffic data from {DATA_FILE}...")
    df = pd.read_csv()


    labels = df['label']
    features = df.drop('label', axis=1)

    print("Starting traffic simulation... (Press Ctrl+C to stop)")

    # Iterate through the rows
    for i in range(len(features)):
        try:
            # Get a single row of data
            row_data = features.iloc[i].values
            true_label = "Attack" if labels.iloc[i] == 1 else "Normal"

            # Convert row to comma-separated string
            message = ",".join(map(str, row_data))

            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((SERVER_IP, SERVER_PORT))

            client_socket.send(message.encode('utf-8'))

            response = client_socket.recv(1024).decode('utf-8')
            print(f"Packet #{i + 1} [Actual: {true_label}] -> Server says: {response}")

            client_socket.close()

            time.sleep(random.uniform(0.1, 0.5))

        except KeyboardInterrupt:
            print("\nSimulation stopped.")
            break
        except ConnectionRefusedError:
            print("Error: Could not connect to server. Is 4_server.py running?")
            break


if __name__ == "__main__":
    simulate_traffic()