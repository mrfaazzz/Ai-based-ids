import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_models():
    # ==========================================
    # 1. LOAD CLEAN DATA
    # ==========================================
    print("Loading clean data...")
    df_train = pd.read_csv("clean_train.csv")
    df_test = pd.read_csv("clean_test.csv")

    # Prepare features (X) and target (y)
    # We drop 'label' (the answer) and 'attack_type' (text name) if it exists
    cols_to_drop = ['label', 'attack_type']

    # intricate check to only drop columns that actually exist to prevent errors
    train_drop = [c for c in cols_to_drop if c in df_train.columns]
    test_drop = [c for c in cols_to_drop if c in df_test.columns]

    X_train = df_train.drop(train_drop, axis=1)
    y_train = df_train['label']

    X_test = df_test.drop(test_drop, axis=1)
    y_test = df_test['label']

    print(f"Data Loaded. Training with {X_train.shape[1]} features.")

    # ==========================================
    # 2. TRAIN DECISION TREE (Model A)
    # ==========================================
    print("\n--- Training Decision Tree ---")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    dt_pred = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)

    print(f"Decision Tree Accuracy: {dt_acc:.4f}")
    print("Decision Tree Report:")
    print(classification_report(y_test, dt_pred))

    # ==========================================
    # 3. TRAIN RANDOM FOREST (Model B)
    # ==========================================
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print("Random Forest Report:")
    print(classification_report(y_test, rf_pred))

    # ==========================================
    # 4. SAVE RESULTS & MODEL
    # ==========================================
    # Compare and tell user which was better
    if rf_acc > dt_acc:
        print(f"\nResult: Random Forest is better by {(rf_acc - dt_acc) * 100:.2f}%")
        joblib.dump(rf_model, "ids_model.joblib")
    else:
        print(f"\nResult: Decision Tree is better by {(dt_acc - rf_acc) * 100:.2f}%")
        joblib.dump(dt_model, "ids_model.joblib")

    print("[Success] Best model saved as 'ids_model.joblib'")

    # Optional: Create Confusion Matrix Chart for your Report
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix (Random Forest)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("chart_confusion_matrix.png")
    print("[Saved] chart_confusion_matrix.png")


if __name__ == "__main__":
    train_models()