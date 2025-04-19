import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import wandb
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


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
        device=None,
        log_frequency=10,
        model_save_path='best_model.pt'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.log_frequency = log_frequency
        self.model_save_path = model_save_path

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

        # Initialize wandb
        wandb.init(
            project="mnist-classification",
            config={
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "patience": patience,
                "model_architecture": str(model),
                "optimizer": "AdamW",
                "scheduler": "StepLR",
                "device": str(self.device)
            }
        )
        # Log model gradients and parameters
        wandb.watch(model, log="all", log_freq=100)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        global_step = wandb.run.step

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()

            # Log gradients and their statistics before optimizer step
            if batch_idx % self.log_frequency == 0:
                grad_logs = {}
                total_norm = 0.0

                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Get gradient data
                        grad_data = param.grad.detach().cpu().numpy()

                        # Calculate gradient statistics
                        grad_norm = np.linalg.norm(grad_data)
                        total_norm += grad_norm ** 2

                        # Log gradient histograms
                        grad_logs[f"gradients/{name}"] = wandb.Histogram(
                            grad_data)

                        # Log gradient statistics
                        grad_logs.update({
                            f"gradients/{name}/norm": grad_norm,
                            f"gradients/{name}/mean": np.mean(grad_data),
                            f"gradients/{name}/std": np.std(grad_data),
                            f"gradients/{name}/max": np.max(grad_data),
                            f"gradients/{name}/min": np.min(grad_data)
                        })

                        # Log layer-wise learning rates
                        if name in self.optimizer.param_groups[0]['params']:
                            grad_logs[f"learning_rate/{name}"] = self.optimizer.param_groups[0]['lr']

                # Log total gradient norm
                total_norm = total_norm ** 0.5
                grad_logs["gradients/total_norm"] = total_norm

                wandb.log(grad_logs, step=global_step)

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Log per batch metrics with configurable frequency
            if batch_idx % self.log_frequency == 0:
                batch_metrics = {
                    "batch/loss": loss.item(),
                    "batch/accuracy": 100. * predicted.eq(target).sum().item() / target.size(0),
                    "global_step": global_step
                }
                wandb.log(batch_metrics, step=global_step)
                global_step += 1

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {batch_idx}/{len(self.train_loader)} '
                      f'Loss: {total_loss/(batch_idx+1):.6f} '
                      f'Acc: {100.*correct/total:.2f}%')

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        # Log weights and their statistics at epoch end
        weight_logs = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Get weight data
                weight_data = param.detach().cpu().numpy()

                # Log weight histograms
                weight_logs[f"weights/{name}"] = wandb.Histogram(weight_data)

                # Log weight statistics
                weight_logs.update({
                    f"weights/{name}/mean": np.mean(weight_data),
                    f"weights/{name}/std": np.std(weight_data),
                    f"weights/{name}/max": np.max(weight_data),
                    f"weights/{name}/min": np.min(weight_data),
                    f"weights/{name}/norm": np.linalg.norm(weight_data)
                })

                # Log gradients if they exist
                if param.grad is not None:
                    grad_data = param.grad.detach().cpu().numpy()
                    weight_logs[f"gradients/{name}"] = wandb.Histogram(
                        grad_data)
                    weight_logs.update({
                        f"gradients/{name}/mean": np.mean(grad_data),
                        f"gradients/{name}/std": np.std(grad_data),
                        f"gradients/{name}/max": np.max(grad_data),
                        f"gradients/{name}/min": np.min(grad_data),
                        f"gradients/{name}/norm": np.linalg.norm(grad_data)
                    })

        # Log all epoch metrics at once
        epoch_metrics = {
            "epoch/loss": avg_loss,
            "epoch/accuracy": epoch_acc,
            "epoch": wandb.run.step,
            **weight_logs
        }
        wandb.log(epoch_metrics)

        return avg_loss, epoch_acc

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Collect predictions and targets for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)

        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Log the plot to wandb
        wandb.log({
            "confusion_matrix": wandb.Image(plt)
        })
        plt.close()

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

            # Log epoch metrics
            wandb.log({
                "epoch": epoch,
                "epoch/train_loss": train_loss,
                "epoch/train_accuracy": train_acc,
                "epoch/val_loss": val_loss,
                "epoch/val_accuracy": val_acc,
                "epoch/learning_rate": self.optimizer.param_groups[0]['lr']
            })

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
                torch.save(checkpoint, self.model_save_path)
                print(f'New best model saved with validation accuracy: '
                      f'{val_acc:.2f}%')

                # Log best model metrics
                wandb.log({
                    "best/val_accuracy": val_acc,
                    "best/train_accuracy": train_acc,
                    "best/val_loss": val_loss,
                    "best/train_loss": train_loss,
                    "best/epoch": epoch
                })
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

        # Log final test metrics
        wandb.log({
            "final/test_loss": test_loss,
            "final/test_accuracy": test_acc
        })

    def load_best_model(self):
        """Load the best model from the saved checkpoint."""
        if os.path.exists(self.model_save_path):
            checkpoint = torch.load(self.model_save_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_val_acc = checkpoint['val_acc']
            self.best_epoch = checkpoint['epoch']
            print(f"Loaded best model from epoch {self.best_epoch + 1} "
                  f"with validation accuracy: {self.best_val_acc:.2f}%")
            return True
        else:
            print(f"No saved model found at {self.model_save_path}")
            return False

    def predict(self, data):
        """
        Make predictions using the loaded model.
        Args:
            data: Input data tensor
        Returns:
            predictions: Predicted class labels
            probabilities: Class probabilities
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            outputs = self.model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities

    def evaluate_single_batch(self, data, targets):
        """
        Evaluate the model on a single batch of data.
        Args:
            data: Input data tensor
            targets: Target labels tensor
        Returns:
            accuracy: Batch accuracy
            loss: Batch loss
        """
        self.model.eval()
        with torch.no_grad():
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            _, predicted = outputs.max(1)
            accuracy = 100. * \
                predicted.eq(targets).sum().item() / targets.size(0)
        return accuracy, loss.item()
