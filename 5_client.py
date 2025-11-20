import socket
import pandas as pd
import json
import time
import random
import sys
from colorama import init, Fore

init(autoreset=True)

# 1. CONFIGURATION
HOST = '127.0.0.1'
PORT = 9999
DATA_FILE = 'clean_test.csv'

print(f"{Fore.CYAN}[INIT] Loading Test Data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)

    if 'label' in df.columns:
        df = df.drop('label', axis=1)
    print(f"{Fore.GREEN}[SUCCESS] Loaded {len(df)} traffic samples.")
except FileNotFoundError:
    print(f"{Fore.RED}[ERROR] {DATA_FILE} not found! Run '1_analysis.py' first.")
    sys.exit(1)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    print(f"{Fore.CYAN}[NET] Connecting to Visual Defense Center at {HOST}:{PORT}...")
    client.connect((HOST, PORT))
    print(f"{Fore.GREEN}[SUCCESS] Connected!")
except ConnectionRefusedError:
    print(f"{Fore.RED}[ERROR] Server is offline. Run '4_server.py' first!")
    sys.exit(1)

try:
    while True:
        random_index = random.randint(0, len(df) - 1)
        packet_data = df.iloc[random_index].values.tolist()

        json_data = json.dumps(packet_data)
        client.send(json_data.encode('utf-8'))

        print(f"{Fore.YELLOW}[OUT] Sending Packet #{random_index}...", end=" ")

        response = client.recv(1024).decode('utf-8')

        if "ALERT" in response:
            print(f"{Fore.RED}SERVER BLOCKED ATTACK!")
        else:
            print(f"{Fore.GREEN}SERVER ALLOWED TRAFFIC.")

        time.sleep(2)

except KeyboardInterrupt:
    print("\n[STOP] Client stopping...")
    client.close()