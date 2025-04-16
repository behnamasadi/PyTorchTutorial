import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


class BaseTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=10,
        patience=5,
        device=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Initialize optimizer, loss function, and scheduler
        self.optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

        # Early stopping variables
        self.best_val_acc = 0
        self.best_model_state = None
        self.best_optimizer_state = None
        self.best_epoch = 0
        self.patience_counter = 0

        # Move model to device
        self.model = self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {batch_idx}/{len(self.train_loader)} '
                      f'Loss: {total_loss/(batch_idx+1):.6f} '
                      f'Acc: {100.*correct/total:.2f}%')

        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        return total_loss / len(loader), 100. * correct / total

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")

            # Training phase
            train_loss, train_acc = self.train_epoch()

            # Validation phase
            val_loss, val_acc = self.validate(self.val_loader)

            print(
                f'Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}%')

            # Update learning rate
            self.scheduler.step()

            # Save best model based on validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict()
                self.best_optimizer_state = self.optimizer.state_dict()
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save the best model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.best_optimizer_state,
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }
                torch.save(checkpoint, 'best_model.pt')
                print(f'New best model saved with validation accuracy: '
                      f'{val_acc:.2f}%')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f'\nEarly stopping triggered! No improvement for '
                          f'{self.patience} epochs.')
                    print(f'Best validation accuracy: {self.best_val_acc:.2f}% '
                          f'at epoch {self.best_epoch + 1}')
                    break

        # Load the best model at the end of training
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.optimizer.load_state_dict(self.best_optimizer_state)
            print(f'\nRestored best model from epoch {self.best_epoch + 1} '
                  f'with validation accuracy: {self.best_val_acc:.2f}%')

        # Evaluate on test set after training is complete
        test_loss, test_acc = self.validate(self.test_loader)
        print(f'Final Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.2f}%')
