import torch
import torch.nn as nn
from models.mlp import MLP
import matplotlib.pyplot as plt
import numpy as np
from data import get_mnist_dataloaders
import os
import json
import argparse
import yaml
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class ModelInference:
    def __init__(self, experiment_dir=None, model_path=None, device=None):
        """
        Initialize the inference class.
        Args:
            experiment_dir: Path to the experiment directory (preferred)
            model_path: Path to the model file (alternative)
            device: Device to run inference on (default: CUDA if available)
        """
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_dir = experiment_dir
        self.model_path = model_path
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.config = None
        self.hyperparameters = None

    def load_config(self):
        """Load configuration from config.yaml and hyperparameters.json"""
        if not self.experiment_dir:
            return False

        try:
            # Load config.yaml
            config_path = os.path.join(self.experiment_dir, 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                    print("Loaded configuration from config.yaml")

            # Load hyperparameters.json
            hyperparams_path = os.path.join(
                self.experiment_dir, 'hyperparameters.json')
            if os.path.exists(hyperparams_path):
                with open(hyperparams_path, 'r') as f:
                    self.hyperparameters = json.load(f)
                    print("Loaded hyperparameters from hyperparameters.json")

            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False

    def load_model(self, input_dim=784, hidden_dim=128, output_dim=10):
        """
        Load the trained model from checkpoint.
        Args:
            input_dim: Input dimension of the model (used only if config not available)
            hidden_dim: Hidden dimension of the model (used only if config not available)
            output_dim: Output dimension of the model (used only if config not available)
        Returns:
            bool: True if model loaded successfully
        """
        try:
            # Load configuration if available
            if self.experiment_dir:
                self.load_config()

                # Get model parameters from config if available
                if self.hyperparameters and 'model_params' in self.hyperparameters:
                    model_params = self.hyperparameters['model_params']
                    input_dim = model_params.get('input_dim', input_dim)
                    hidden_dim = model_params.get('hidden_dim', hidden_dim)
                    output_dim = model_params.get('output_dim', output_dim)
                    print(f"Using model parameters from config: "
                          f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
                          f"output_dim={output_dim}")

            # Initialize model with architecture from config or defaults
            self.model = MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )

            # Try to load from experiment directory first
            if self.experiment_dir:
                model_path = os.path.join(self.experiment_dir, 'model.pt')
                if os.path.exists(model_path):
                    self.model_path = model_path
                    # Load metrics for information
                    metrics_path = os.path.join(
                        self.experiment_dir, 'metrics.json')
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)
                            print(f"Model metrics: {metrics}")

            # Load the model
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = torch.load(
                    self.model_path, map_location=self.device)

                # Handle both types of model files
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Full checkpoint (best_model.pt)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
                    print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
                else:
                    # Simple model file (model.pt)
                    self.model.load_state_dict(checkpoint)
                    print("Loaded model weights")

                self.model = self.model.to(self.device)
                self.model.eval()
                return True
            else:
                print(f"No model file found at {self.model_path}")
                return False

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, data):
        """
        Make predictions on input data.
        Args:
            data: Input data tensor
        Returns:
            predictions: Predicted class labels
            probabilities: Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        with torch.no_grad():
            data = data.to(self.device)
            outputs = self.model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities

    def evaluate_batch(self, data, targets):
        """
        Evaluate model on a batch of data.
        Args:
            data: Input data tensor
            targets: Target labels tensor
        Returns:
            accuracy: Batch accuracy
            loss: Batch loss
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        with torch.no_grad():
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            _, predicted = outputs.max(1)
            accuracy = 100. * \
                predicted.eq(targets).sum().item() / targets.size(0)
        return accuracy, loss.item()

    def evaluate_dataset(self, dataloader):
        """
        Evaluate the model on an entire dataset.
        Args:
            dataloader: DataLoader containing the dataset to evaluate
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)

        # Get classification report
        report = classification_report(
            all_targets, all_preds, output_dict=True)

        # Create metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'confusion_matrix': cm,
            'classification_report': report
        }

        return metrics

    def visualize_confusion_matrix(self, cm, save_path=None):
        """
        Visualize the confusion matrix.
        Args:
            cm: Confusion matrix
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def visualize_predictions(self, data, targets, num_samples=10, save_path=None):
        """
        Visualize model predictions on a batch of images.
        Args:
            data: Input data tensor
            targets: Target labels tensor
            num_samples: Number of samples to visualize
            save_path: Path to save the figure (optional)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        predictions, probabilities = self.predict(data)

        # Convert to numpy for plotting
        data = data.cpu().numpy()
        targets = targets.cpu().numpy()
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()

        # Plot samples
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2*num_samples))
        for i in range(num_samples):
            img = data[i].reshape(28, 28)
            true_label = targets[i]
            pred_label = predictions[i]
            confidence = probabilities[i][pred_label] * 100

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(
                f'True: {true_label}, Pred: {pred_label} '
                f'(Conf: {confidence:.1f}%)'
            )
            axes[i].axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Predictions visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MNIST Model Inference')
    parser.add_argument('--experiment_dir', type=str,
                        help='Path to the experiment directory')
    parser.add_argument('--model_path', type=str,
                        help='Path to the model file (alternative to experiment_dir)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize (default: 10)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device to run inference on (default: auto-detect)')
    parser.add_argument('--input_dim', type=int, default=784,
                        help='Input dimension of the model (default: 784)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension of the model (default: 128)')
    parser.add_argument('--output_dim', type=int, default=10,
                        help='Output dimension of the model (default: 10)')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results (default: results)')

    args = parser.parse_args()

    # Validate arguments
    if not args.experiment_dir and not args.model_path:
        parser.error(
            "Either --experiment_dir or --model_path must be provided")

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize inference
    inference = ModelInference(
        experiment_dir=args.experiment_dir,
        model_path=args.model_path,
        device=torch.device(args.device) if args.device else None
    )

    # Load model
    if inference.load_model(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    ):
        # Get test data
        test_loader, = get_mnist_dataloaders(
            batch_size=args.batch_size, train=False)

        # Evaluate on entire test dataset
        print("Evaluating on entire test dataset...")
        metrics = inference.evaluate_dataset(test_loader)

        # Print metrics
        print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Test Loss: {metrics['loss']:.4f}")

        # Print classification report
        print("\nClassification Report:")
        for label, report in metrics['classification_report'].items():
            if isinstance(report, dict):
                print(f"Class {label}:")
                for metric, value in report.items():
                    print(f"  {metric}: {value:.4f}")

        # Visualize confusion matrix
        cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
        inference.visualize_confusion_matrix(
            metrics['confusion_matrix'], cm_path)

        # Get a batch for visualization
        data, targets = next(iter(test_loader))

        # Visualize predictions
        pred_path = os.path.join(args.save_dir, 'predictions.png')
        inference.visualize_predictions(
            data, targets, num_samples=args.num_samples, save_path=pred_path)

        # Save metrics to file
        metrics_path = os.path.join(args.save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_json = {
                'accuracy': float(metrics['accuracy']),
                'loss': float(metrics['loss']),
                'confusion_matrix': metrics['confusion_matrix'].tolist(),
                'classification_report': metrics['classification_report']
            }
            json.dump(metrics_json, f, indent=4)
        print(f"Metrics saved to {metrics_path}")

    else:
        print("Failed to load model. Exiting.")


if __name__ == "__main__":
    main()
