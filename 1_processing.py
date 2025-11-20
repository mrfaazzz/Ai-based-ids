# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# import joblib
#
# columns = [
#     'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
#     'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
#     'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
#     'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
#     'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
#     'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
#     'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
#     'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
#     'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
#     'dst_host_serror_rate', 'dst_host_srv_serror_rate',
#     'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
# ]
#
#
# def process_data():
#     print("Loading datasets...")
#     train = pd.read_csv('KDDTrain.csv', header=None, names=columns)
#     test = pd.read_csv('KDDTest.csv', header=None, names=columns)
#
#     train.drop(['difficulty'], axis=1, inplace=True)
#     test.drop(['difficulty'], axis=1, inplace=True)
#
#     print("Encoding categorical data...")
#     cat_cols = ['protocol_type', 'service', 'flag']
#
#     encoders = {}
#
#     for col in cat_cols:
#         le = LabelEncoder()
#         combined_data = pd.concat([train[col], test[col]], axis=0)
#         le.fit(combined_data)
#
#         train[col] = le.transform(train[col])
#         test[col] = le.transform(test[col])
#
#         encoders[col] = le
#
#     joblib.dump(encoders, 'encoders.joblib')
#     print("Encoders saved as 'encoders.joblib'")
#
#     print("Mapping labels (Normal=0, Attack=1)...")
#     train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
#     test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)
#     train.to_csv('clean_train.csv', index=False)
#     test.to_csv('clean_test.csv', index=False)
#     print("Processing complete. Files saved: clean_train.csv, clean_test.csv")
#
#
# if __name__ == "__main__":
#     process_data()


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib


def process_data():
    print("Loading datasets...")
    # UPDATED: Removed 'header=None' because the files actually have headers
    train = pd.read_csv('KDDTrain.csv')
    test = pd.read_csv('KDDTest.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    cat_cols = ['protocol_type', 'service', 'flag']

    encoders = {}

    print("Encoding categorical data...")
    for col in cat_cols:
        if col in train.columns:
            # Only encode if the column is actually text (object)
            if train[col].dtype == 'object':
                le = LabelEncoder()
                combined_data = pd.concat([train[col], test[col]], axis=0).astype(str)
                le.fit(combined_data)

                train[col] = le.transform(train[col].astype(str))
                test[col] = le.transform(test[col].astype(str))
                encoders[col] = le
            else:
                print(f"Skipping encoding for '{col}' (already numeric).")

    # Save encoders if any were created
    if encoders:
        joblib.dump(encoders, 'encoders.joblib')
        print("Encoders saved as 'encoders.joblib'")
    else:
        # If no encoders needed, save an empty dict or handle gracefully
        joblib.dump({}, 'encoders.joblib')
        print("No text columns found to encode. Saved empty encoders file.")

    print("Mapping labels...")
    # The dataset uses 'label' for the binary/multiclass ID and 'attack_type' for the name
    # We want 'label' to be our target.
    # Check if label is already 0/1 or text
    if train['label'].dtype == 'object':
        # Map 'normal' to 0, everything else to 1
        train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
        test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)
    else:
        # If it's already numeric, ensure it's binary (0 vs 1)
        # In some KDD versions, 'normal' is 0 and attacks are >0
        print("Label column is already numeric. Ensuring binary classification...")
        train['label'] = train['label'].apply(lambda x: 1 if x > 0 else 0)
        test['label'] = test['label'].apply(lambda x: 1 if x > 0 else 0)

    # Remove 'attack_type' if it exists, as it reveals the answer
    if 'attack_type' in train.columns:
        train.drop('attack_type', axis=1, inplace=True)
        test.drop('attack_type', axis=1, inplace=True)

    # Save processed files
    train.to_csv('clean_train.csv', index=False)
    test.to_csv('clean_test.csv', index=False)
    print("Processing complete. Files saved: clean_train.csv, clean_test.csv")


if __name__ == "__main__":
    process_data()