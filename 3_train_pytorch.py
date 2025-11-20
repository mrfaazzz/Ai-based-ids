import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Define the Feed-Forward Neural Network
class IDS_NN(nn.Module):
    def __init__(self, input_dim):
        super(IDS_NN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()  # Binary classification output (0 or 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x


def train_pytorch():
    print("Loading data for PyTorch...")
    df = pd.read_csv('clean_train.csv')

    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Convert to Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape (N, 1)

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize Model
    input_dim = X.shape[1]
    model = IDS_NN(input_dim)

    # Loss and Optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training for 5 epochs...")
    for epoch in range(5):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/5, Loss: {epoch_loss / len(dataloader):.4f}")

    print("Training complete.")

    # Optional: Save PyTorch model state
    torch.save(model.state_dict(), 'ids_model_pytorch.pth')
    print("PyTorch model saved as 'ids_model_pytorch.pth'")


if __name__ == "__main__":
    train_pytorch()