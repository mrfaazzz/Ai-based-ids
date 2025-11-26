import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # <--- Needed to display images internally
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11


def process_data():
    print("=" * 60)
    print("       DATA PROCESSING & VISUALIZATION PIPELINE")
    print("=" * 60)

    print("\n Loading Data...")
    train = pd.read_csv('KDDTrain.csv', low_memory=False)
    test = pd.read_csv('KDDTest.csv', low_memory=False)

    if 'difficulty' in train.columns: train.drop('difficulty', axis=1, inplace=True)
    if 'difficulty' in test.columns: test.drop('difficulty', axis=1, inplace=True)

    def clean_label(x):
        if isinstance(x, str):
            return 0 if x == 'normal' else 1
        return 0 if int(x) == 0 else 1

    train['binary_label'] = train['label'].apply(clean_label)
    test['binary_label'] = test['label'].apply(clean_label)

    print("\n Generating EDA Graphs ")

    print(" -> Creating 'distribution.png'...")
    train['temp_label_name'] = train['binary_label'].apply(lambda x: 'Normal' if x == 0 else 'Attack')

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x='temp_label_name', data=train, palette=['#2ecc71', '#e74c3c'])
    plt.title("Class Distribution: Normal vs Attack", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Traffic Type")
    plt.ylabel("Count")

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + 0.4, p.get_height()), ha='center', va='bottom',
                    fontweight='bold')

    plt.savefig('distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    Saved")

    print(" -> Creating 'attack_types.png'...")

    if 'attack_type' in train.columns:
        target_col = 'attack_type'
    else:
        target_col = 'label'

    attacks_only = train[train[target_col] != 'normal']

    if not attacks_only.empty:
        attack_counts = attacks_only[target_col].value_counts(normalize=True) * 100
        top_attacks = attack_counts.head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_attacks.index, y=top_attacks.values, palette="magma")

        plt.title("Top 10 Attack Types (%)", fontsize=14, fontweight='bold')
        plt.xlabel("Attack Name")
        plt.ylabel("Percentage")
        plt.xticks(rotation=45)

        for i, v in enumerate(top_attacks.values):
            plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')

        plt.savefig('attack_types.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    Saved")
    else:
        print("      [WARNING] No attacks found to plot.")

    print(" -> Creating 'correlation.png'...")

    train_enc = train.copy()
    train_enc['label'] = train_enc['binary_label']

    for c in ['protocol_type', 'service', 'flag']:
        if c in train_enc.columns:
            le = LabelEncoder()
            train_enc[c] = le.fit_transform(train_enc[c].astype(str))

    corr = train_enc.corr()

    if 'label' in corr.columns:
        k = 15
        cols = corr.nlargest(k, 'label')['label'].index
        cm = np.corrcoef(train_enc[cols].values.T)

        plt.figure(figsize=(11, 9))
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                    annot_kws={'size': 9}, yticklabels=cols.values, xticklabels=cols.values,
                    cmap='coolwarm', linewidths=1, linecolor='white')

        plt.title("Top 15 Feature Correlations", fontsize=15, fontweight='bold', pad=20)
        plt.savefig('correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    Saved")

    print("=" * 60)
    print("       FINALIZING DATA FOR TRAINING ")
    print("=" * 60)

    train['label'] = train['binary_label']
    test['label'] = test['binary_label']

    cat_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, 'encoders.joblib')

    cols_to_drop = ['temp_label_name', 'binary_label', 'attack_type', 'difficulty']
    for col in cols_to_drop:
        if col in train.columns: train.drop(col, axis=1, inplace=True)
        if col in test.columns: test.drop(col, axis=1, inplace=True)

    scaler = StandardScaler()
    X_train = train.drop('label', axis=1)
    X_test = test.drop('label', axis=1)

    cols = X_train.columns
    train[cols] = scaler.fit_transform(X_train)
    test[cols] = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.joblib')

    train.to_csv('clean_train.csv', index=False)
    test.to_csv('clean_test.csv', index=False)

    print("\n Showing Graphs (Close window to see next one)")

    images_to_show = ['distribution.png', 'attack_types.png', 'correlation.png']

    for img_file in images_to_show:
        if os.path.exists(img_file):
            try:
                img = mpimg.imread(img_file)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Displaying: {img_file}", fontsize=10)
                plt.show()
            except Exception as e:
                print(f"Could not display {img_file}: {e}")

    print("\n[DONE] Processing Complete.")


if __name__ == "__main__":
    process_data()
