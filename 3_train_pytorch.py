import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11


def main():
    print("Loading data for Neural Network...")

    # Check if files exist
    if not os.path.exists("clean_train.csv"):
        print("Error: clean_train.csv not found! Run 1_processing.py first.")
        return

    # Load data
    df_train = pd.read_csv("clean_train.csv")
    df_test = pd.read_csv("clean_test.csv")

    # Columns we don't need for training
    drop_cols = ['label', 'attack_type']

    # Prepare features (X) and target (y)
    # We check if columns exist before dropping to be safe
    train_drop = [c for c in drop_cols if c in df_train.columns]
    test_drop = [c for c in drop_cols if c in df_test.columns]

    X_train = df_train.drop(train_drop, axis=1).values
    y_train = df_train['label'].values

    X_test = df_test.drop(test_drop, axis=1).values
    y_test = df_test['label'].values

    # Scale data (Important for Neural Networks!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

    # Define the Neural Network Model
    class SimpleIDS(nn.Module):
        def __init__(self, n_features):
            super(SimpleIDS, self).__init__()
            # Simple 3-layer network
            self.layer1 = nn.Linear(n_features, 64)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(64, 32)
            self.layer3 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.sigmoid(self.layer3(x))
            return x

    # Initialize model
    input_size = X_train.shape[1]
    model = SimpleIDS(input_size)

    # Binary Cross Entropy Loss is good for 0/1 classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"\nTraining Neural Network ({input_size} features)...")

    epochs = 20
    losses = []

    # Training Loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save loss to plot later
        losses.append(loss.item())

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Plot Training Loss
    print("Saving Loss Graph...")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, marker='o', color='#d35400', label="Training Loss")
    plt.title("Neural Network Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig("training_loss.png", bbox_inches='tight')
    plt.show()

    # Evaluation
    print("\nEvaluating Model...")
    with torch.no_grad():
        preds = model(X_test_tensor)
        # Convert probability to 0 or 1
        predicted_labels = (preds > 0.5).float()

        acc = accuracy_score(y_test, predicted_labels)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, predicted_labels))

    # Save the model
    torch.save(model.state_dict(), "pytorch_model.pth")
    print("Model saved as 'pytorch_model.pth'")


if __name__ == "__main__":
    main()