from torchvision import transforms
from .base_dataloader import BaseMNISTDataLoader


class ResNetMNISTDataLoader(BaseMNISTDataLoader):
    def get_transforms(self, train=True):
        """
        Get transforms for ResNet model.

        Args:
            train (bool): If True, include augmentation transforms

        Returns:
            transforms.Compose: The composed transforms
        """
        if train:
            return transforms.Compose([
                # Resize to match ResNet input size
                transforms.Resize((224, 224)),
                # Random rotation up to 10 degrees
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(
                    0.1, 0.1)),  # Random translation
                transforms.ToTensor(),
                # MNIST mean and std
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            return transforms.Compose([
                # Resize to match ResNet input size
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
