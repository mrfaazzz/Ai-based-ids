# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
#
#
# def train_rf():
#     print("Loading clean training data...")
#     train_df = pd.read_csv('clean_train.csv')
#     test_df = pd.read_csv('clean_test.csv')
#
#     X_train = train_df.drop('label', axis=1)
#     y_train = train_df['label']
#
#     X_test = test_df.drop('label', axis=1)
#     y_test = test_df['label']
#
#     print("Training Random Forest Classifier...")
#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf.fit(X_train, y_train)
#
#     preds = rf.predict(X_test)
#     acc = accuracy_score(y_test, preds)
#
#     print(f"Model Accuracy: {acc * 100:.2f}%")
#     print("\nClassification Report:\n", classification_report(y_test, preds))
#
#     joblib.dump(rf, 'ids_model.joblib')
#     print("Model saved as 'ids_model.joblib'")
#
# if __name__ == "__main__":
#     train_rf()


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def train_rf():
    print("Loading clean training data...")
    train_df = pd.read_csv('clean_train.csv')
    test_df = pd.read_csv('clean_test.csv')

    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']

    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    joblib.dump(rf, 'ids_model.joblib')
    print("Model saved as 'ids_model.joblib'")


if __name__ == "__main__":
    train_rf()