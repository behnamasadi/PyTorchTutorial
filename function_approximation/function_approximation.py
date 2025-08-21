import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set default device for all tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


class FunctionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class FunctionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        x = torch.relu(self.fc3(x))

        x = self.fc4(x)
        return x


def validate_model(model, dataloader, criterion):
    """Validate the model and return validation loss"""
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_val_loss += loss.item()
    return total_val_loss / len(dataloader)


seed = 42
generator = torch.Generator().manual_seed(seed)
batch_size = 64


def main():
    # Generate data
    n_samples = 1000  # More data
    x = torch.linspace(-20, 20, n_samples).reshape(-1, 1)
    y = torch.sin(x) + 0.05 * torch.randn(n_samples, 1)

    # Convert to numpy for plotting
    x_np = x.numpy()
    y_np = y.numpy()

    plt.plot(x_np, y_np, ".")
    plt.title("Generated Training Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Simple normalization - just scale x to [-1, 1] and keep y as is
    x_normalized = x / 20.0  # Scale x to [-1, 1]
    y_normalized = y  # Keep y as is (already in good range)

    print(f"Original x range: [{x.min():.2f}, {x.max():.2f}]")
    print(
        f"Normalized x range: [{x_normalized.min():.2f}, {x_normalized.max():.2f}]")
    print(f"Original y range: [{y.min():.2f}, {y.max():.2f}]")
    print(
        f"Normalized y range: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")

    # Create dataset and dataloader with normalized data
    function_dataset = FunctionDataset(x_normalized, y_normalized)

    train_set, val_set, test_set = random_split(dataset=function_dataset, lengths=[
        0.7, 0.15, 0.15], generator=generator)

    dataloader_train = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)

    dataloader_val = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    # Create model, optimizer, and loss function
    model = FunctionModel()

    # Better initialization
    # for module in model.modules():
    #     if isinstance(module, nn.Linear):
    #         nn.init.xavier_uniform_(module.weight)
    #         nn.init.zeros_(module.bias)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.bias)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True)
    criterion = nn.MSELoss()

    # Training loop with validation
    num_epochs = 600
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    print("Starting training with validation...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    print(f"Test samples: {len(test_set)}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in dataloader_train:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)

        # Validation phase
        avg_val_loss = validate_model(model, dataloader_val, criterion)
        val_losses.append(avg_val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping and best model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best validation loss: {best_val_loss:.6f}")

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Test the model on test set
    model.eval()
    test_loss = validate_model(model, dataloader_test, criterion)
    print(f"Test Loss: {test_loss:.6f}")

    # Test the model on full range (with proper denormalization)
    with torch.no_grad():
        # Generate test data in original scale
        x_test_original = torch.linspace(-20, 20, 1000).reshape(-1, 1)

        # Normalize test data using the same scaling
        x_test_normalized = x_test_original / 20.0

        # Get predictions
        y_pred_normalized = model(x_test_normalized)

        # Denormalize predictions (y was not normalized, so no need to denormalize)
        y_pred_original = y_pred_normalized

        # Convert to numpy for plotting
        x_test_np = x_test_original.numpy()
        y_pred_np = y_pred_original.numpy()

        plt.subplot(1, 2, 2)
        plt.plot(x_np, y_np, ".", alpha=0.5, label="Training Data")
        plt.plot(x_test_np, y_pred_np, "-",
                 linewidth=2, label="Model Prediction")
        plt.plot(x_test_np, np.sin(x_test_np), "--",
                 linewidth=2, label="True sin(x)")
        plt.title("Function Approximation Results (with Simple Normalization)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
