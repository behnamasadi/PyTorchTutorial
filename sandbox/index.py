from sklearn.metrics import r2_score
from torchvision import transforms
import torch
import torchvision.models as models
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import matplotlib.pyplot as plt


class MRIDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class FunctionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


seed = 42
torch.manual_seed(seed)

sample_size = 200
x = torch.linspace(-15, 15, sample_size)
y = torch.sin(x)+torch.randn(sample_size)/15

plt.plot(x, y)
plt.show()


class FunctionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.act = nn.Tanh()           # or nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        # keep output linear for unconstrained regression
        x = self.fc3(x)
        return x


# Fix data format - need to reshape for linear layers
x = x.unsqueeze(1)  # Shape: (200, 1) for linear layers
y = y.unsqueeze(1)  # Shape: (200, 1) for linear layers

dataset = FunctionDataset(data=x, labels=y)

# Create proper splits
n = len(dataset)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(seed)
)
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create separate data loaders for train, validation, and test
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and loss function
model = FunctionModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training parameters
epochs = 100
patience = 15
best_val_loss = float('inf')
best_model_state = None
early_stopping_counter = 0

# Training history
train_losses = []
val_losses = []

print("Starting training...")
print("="*60)

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_batches = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1

    avg_train_loss = train_loss / train_batches
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            val_loss += loss.item()
            val_batches += 1

    avg_val_loss = val_loss / val_batches
    val_losses.append(avg_val_loss)

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(best_model_state)

# Test evaluation
print("\n" + "="*60)
print("TESTING PHASE")
print("="*60)

model.eval()
test_loss = 0.0
test_predictions = []
test_targets = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)
        test_loss += loss.item()

        test_predictions.extend(predictions.squeeze().tolist())
        test_targets.extend(batch_y.squeeze().tolist())

avg_test_loss = test_loss / len(test_loader)

print(f"Final Results:")
print(f"Best Validation Loss: {best_val_loss:.6f}")
print(f"Test Loss: {avg_test_loss:.6f}")
print(f"Test RMSE: {torch.sqrt(torch.tensor(avg_test_loss)):.6f}")

# Calculate R² score
r2 = r2_score(test_targets, test_predictions)
print(f"Test R² Score: {r2:.4f}")

# Plot results
plt.figure(figsize=(15, 5))

# Plot 1: Training curves
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot 2: Test predictions vs actual
plt.subplot(1, 3, 2)
plt.scatter(test_targets, test_predictions, alpha=0.6)
plt.plot([min(test_targets), max(test_targets)], [min(test_targets),
         max(test_targets)], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Test Predictions vs Actual\nR² = {r2:.4f}')
plt.legend()
plt.grid(True)

# Plot 3: Function approximation
plt.subplot(1, 3, 3)
# Get all test data points
test_x = []
test_y = []
for i in range(len(test_dataset)):
    x_val, y_val = test_dataset[i]
    test_x.append(x_val.item())
    test_y.append(y_val.item())

# Sort for plotting
sorted_indices = sorted(range(len(test_x)), key=lambda i: test_x[i])
sorted_x = [test_x[i] for i in sorted_indices]
sorted_y = [test_y[i] for i in sorted_indices]
sorted_pred = [test_predictions[i] for i in sorted_indices]

plt.plot(sorted_x, sorted_y, 'o', label='Actual', alpha=0.6)
plt.plot(sorted_x, sorted_pred, 'r-', label='Predicted', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function Approximation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nModel Analysis:")
print(f"✅ Model architecture: 1 → 10 → 10 → 1 (good for this task)")
print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(val_dataset)}")
print(f"✅ Test samples: {len(test_dataset)}")
print(f"✅ Early stopping: Prevents overfitting")
print(f"✅ Separate validation: Ensures unbiased evaluation")
