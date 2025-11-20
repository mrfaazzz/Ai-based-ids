import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib


def process_data():
    print("Loading datasets...")
    train = pd.read_csv('KDDTrain.csv')
    test = pd.read_csv('KDDTest.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    cat_cols = ['protocol_type', 'service', 'flag']

    encoders = {}

    print("Encoding categorical data...")
    for col in cat_cols:
        if col in train.columns:
            if train[col].dtype == 'object':
                le = LabelEncoder()
                combined_data = pd.concat([train[col], test[col]], axis=0).astype(str)
                le.fit(combined_data)

                train[col] = le.transform(train[col].astype(str))
                test[col] = le.transform(test[col].astype(str))
                encoders[col] = le
            else:
                print(f"Skipping encoding for '{col}' (already numeric).")

    if encoders:
        joblib.dump(encoders, 'encoders.joblib')
        print("Encoders saved as 'encoders.joblib'")
    else:
        joblib.dump({}, 'encoders.joblib')
        print("No text columns found to encode. Saved empty encoders file.")

    print("Mapping labels...")
    if train['label'].dtype == 'object':
        train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
        test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)
    else:
        print("Label column is already numeric. Ensuring binary classification...")
        train['label'] = train['label'].apply(lambda x: 1 if x > 0 else 0)
        test['label'] = test['label'].apply(lambda x: 1 if x > 0 else 0)

    if 'attack_type' in train.columns:
        train.drop('attack_type', axis=1, inplace=True)
        test.drop('attack_type', axis=1, inplace=True)

    train.to_csv('clean_train.csv', index=False)
    test.to_csv('clean_test.csv', index=False)
    print("Processing complete. Files saved: clean_train.csv, clean_test.csv")


if __name__ == "__main__":
    process_data()