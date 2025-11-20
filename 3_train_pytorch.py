import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 1. SETUP & LOAD DATA

print("Loading data for Neural Network...")
df_train = pd.read_csv("clean_train.csv")
df_test = pd.read_csv("clean_test.csv")

cols_to_drop = ['label', 'attack_type']
train_drop = [c for c in cols_to_drop if c in df_train.columns]
test_drop = [c for c in cols_to_drop if c in df_test.columns]

X_train = df_train.drop(train_drop, axis=1).values
y_train = df_train['label'].values

X_test = df_test.drop(test_drop, axis=1).values
y_test = df_test['label'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)



# 2. DEFINE THE NEURAL NETWORK

class SimpleIDS(nn.Module):
    def __init__(self, input_dim):
        super(SimpleIDS, self).__init__()
        # Layer 1: Input -> Hidden (64 neurons)
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        # Layer 2: Hidden -> Hidden (32 neurons)
        self.layer2 = nn.Linear(64, 32)
        # Layer 3: Hidden -> Output (1 neuron: Attack or Normal?)
        self.layer3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x


input_dim = X_train.shape[1]
model = SimpleIDS(input_dim)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. TRAIN THE MODEL

print(f"\n--- Training Neural Network ({input_dim} input features) ---")
epochs = 20  # Number of times to loop through the entire dataset

for epoch in range(epochs):
    # 1. Forward pass: Predict
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 4. EVALUATE

print("\n--- Evaluating Neural Network ---")
with torch.no_grad():
    predicted = model(X_test_tensor)
    predicted_labels = (predicted > 0.5).float()

    acc = accuracy_score(y_test, predicted_labels)
    print(f"Neural Network Accuracy: {acc:.4f}")
    print("Neural Network Report:")
    print(classification_report(y_test, predicted_labels))

torch.save(model.state_dict(), "pytorch_model.pth")
print("[Success] Neural Network saved as 'pytorch_model.pth'")