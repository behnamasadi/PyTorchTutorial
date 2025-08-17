from torchvision import transforms
import torch
import torchvision.models as models
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn


class MRIDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


seed = 42
torch.manual_seed(seed)

data = torch.randint(low=0, high=20, size=(10,))
labels = torch.randint(low=0, high=4, size=(10,))

# print(data)
# print(labels)


dataset = MRIDataset(data=data, labels=labels)
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(seed))

print(train_dataset)
print(val_dataset)
print(test_dataset)
for d, l in train_dataset:
    print(d.item(), l.item())


batch_size = 4
loader = DataLoader(dataset, batch_size=batch_size,
                    shuffle=True, num_workers=2, pin_memory=True)

for batch in loader:
    targets, input = batch
    print(targets)
    print(input)
    print('='*50)


exit(0)
