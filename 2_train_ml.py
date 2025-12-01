import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def train_and_evaluate():
    print("Loading clean data...")
    try:
        train_df = pd.read_csv('clean_train.csv')
        test_df = pd.read_csv('clean_test.csv')
    except FileNotFoundError:
        print("Error: clean_train.csv not found. Run 1_processing.py first!")
        return

    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    print("\n--- Training & Evaluation ---")
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        results[name] = f1  # Store F1 for comparison graph

        print(f"Results for {name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print("-" * 30)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    joblib.dump(best_model, 'ids_model.joblib')
    print(f"\n[Saved] Best model based on F1-Score is saved to 'ids_model.joblib'")

    print("\nGenerating Model Comparison Graph...")
    plt.figure(figsize=(7, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="magma")
    plt.ylabel('F1 Score')
    plt.title('Model Performance Comparison (F1 Score)')
    plt.ylim(0, 1.0)
    plt.savefig('model_comparison.png')
    print("Saved 'model_comparison.png'")


if __name__ == "__main__":
    train_and_evaluate()


