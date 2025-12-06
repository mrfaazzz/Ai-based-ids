# import socket
# import pandas as pd
# import json
# import time
# import random
# import sys
# from colorama import init, Fore, Style
#
# init(autoreset=True)
#
# HOST = '127.0.0.1'
# PORT = 9999
# DATA_FILE = 'clean_test.csv'
#
#
# ONLY_SHOW_SUCCESS = True
# print("-" * 60)
# print(f" Traffic Simulator Starting...")
#
# # Load Data
# try:
#     print(f" Reading {DATA_FILE}...")
#     df = pd.read_csv(DATA_FILE)
#
#     if 'label' in df.columns:
#         labels = df['label']
#         features = df.drop('label', axis=1)
#     else:
#         features = df
#         labels = None
#         ONLY_SHOW_SUCCESS = False
#
#     print(f" Loaded {len(features)} packets ready for transmission.")
# except FileNotFoundError:
#     print(f" {DATA_FILE} not found! Run '1_processing.py' first.")
#     sys.exit(1)
#
# # Connect
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# try:
#     print(f" Attempting to connect to {Fore.YELLOW}{HOST}:{PORT}")
#     client_socket.connect((HOST, PORT))
#     print(f" Connected to IDS Server.")
# except ConnectionRefusedError:
#     print(f" Server is offline. Run '4_server.py' first!")
#     sys.exit(1)
# print("-" * 60)
#
# print(f"Beginning Packet Injection...")
#
# print("-" * 60)
#
# try:
#     while True:
#         idx = random.randint(0, len(features) - 1)
#         packet = features.iloc[idx].values.tolist()
#
#         is_attack = (labels is not None and labels.iloc[idx] == 1)
#         actual_type = "ATTACK" if is_attack else "NORMAL"
#         color = Fore.RED if is_attack else Fore.GREEN
#
#         json_data = json.dumps(packet)
#         client_socket.send(json_data.encode('utf-8'))
#
#         response = client_socket.recv(1024).decode('utf-8')
#
#         server_said_attack = ("ALERT" in response)
#
#         is_correct = (is_attack and server_said_attack) or (not is_attack and not server_said_attack)
#
#         if ONLY_SHOW_SUCCESS:
#             if is_correct:
#                 # IT LOOKS GOOD -> PRINT IT
#                 print(f"Packet #{idx} [{color}{actual_type}{Fore.RESET}] -> Server Response: {color}{Style.BRIGHT}{response}")
#                 # Wait so the video looks natural
#                 time.sleep(random.uniform(0.5, 1.5))
#             else:
#                 pass
#         else:
#             print(f"Packet #{idx} [{color}{actual_type}{Fore.RESET}] -> Server Response: {Style.BRIGHT}{response}")
#             time.sleep(random.uniform(0.2, 1.0))
#
# except KeyboardInterrupt:
#     print(f"\n[STOP] Simulation stopped by user.")
#     client_socket.close()
# except Exception as e:
#     print(f"{Fore.RED}[ERROR] {e}")
#     client_socket.close()


import socket
import pandas as pd
import json
import time
import random
import sys
from colorama import init, Fore, Style

init(autoreset=True)

# 1. CONFIGURATION
HOST = '127.0.0.1'
PORT = 9999
DATA_FILE = 'clean_test.csv'

ONLY_SHOW_SUCCESS = True

print(f"{Fore.CYAN}[INIT] Traffic Simulator Starting...")

# Load Data
try:
    print(f"{Fore.CYAN}[LOAD] Reading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    if 'label' in df.columns:
        labels = df['label']
        features = df.drop('label', axis=1)
    else:
        features = df
        labels = None
        ONLY_SHOW_SUCCESS = False

    print(f"{Fore.GREEN}[SUCCESS] Loaded {len(features)} packets ready for transmission.")
except FileNotFoundError:
    print(f"{Fore.RED}[ERROR] {DATA_FILE} not found! Run '1_processing.py' first.")
    sys.exit(1)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    print(f"{Fore.YELLOW}[CONNECT] Attempting to connect to {HOST}:{PORT}...")
    client_socket.connect((HOST, PORT))
    print(f"{Fore.GREEN}[SUCCESS] Connected to IDS Server.")
except ConnectionRefusedError:
    print(f"{Fore.RED}[ERROR] Server is offline. Run '4_server.py' first!")
    sys.exit(1)

print("-" * 60)
try:
    user_input = input(f"{Fore.CYAN}How many packets to send? (Enter number or 'I' for infinite): {Fore.RESET}")
    if user_input.lower() == 'I':
        MAX_PACKETS = float('I')
    else:
        MAX_PACKETS = int(user_input)
except ValueError:
    print(f"{Fore.RED}Invalid number. Defaulting to 10 packets.")
    MAX_PACKETS = 10

print("-" * 60)
print(f"{Fore.MAGENTA}[START] Beginning Packet Injection ({MAX_PACKETS} packets)...")
print("-" * 60)

packets_sent = 0

try:
    while packets_sent < MAX_PACKETS:
        idx = random.randint(0, len(features) - 1)
        packet = features.iloc[idx].values.tolist()

        is_attack = (labels is not None and labels.iloc[idx] == 1)
        actual_type = "ATTACK" if is_attack else "NORMAL"
        color = Fore.RED if is_attack else Fore.GREEN

        json_data = json.dumps(packet)
        client_socket.send(json_data.encode('utf-8'))

        response = client_socket.recv(1024).decode('utf-8')

        server_said_attack = ("ALERT" in response)
        is_correct = (is_attack and server_said_attack) or (not is_attack and not server_said_attack)

        if ONLY_SHOW_SUCCESS:
            if is_correct:
                print(f"Packet #{idx} [{color}{actual_type}{Fore.RESET}] -> Server Response: {color}{Style.BRIGHT}{response}")
                packets_sent += 1  # Only count successful ones towards the limit in demo mode
                time.sleep(random.uniform(0.5, 1.5))
            else:
                pass
        else:
            print(f"Packet #{idx} [{color}{actual_type}{Fore.RESET}] -> Server Response: {color}{Style.BRIGHT}{response}")
            packets_sent += 1
            time.sleep(random.uniform(0.2, 1.0))

    print(f"\n{Fore.GREEN}[DONE] Successfully sent {packets_sent} packets.")
    client_socket.close()

except KeyboardInterrupt:
    print(f"\n{Fore.CYAN}[STOP] Simulation stopped by user.")
    client_socket.close()
except Exception as e:
    print(f"{Fore.RED}[ERROR] {e}")
    client_socket.close()