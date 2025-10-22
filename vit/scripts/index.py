import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2)  # [B, embed_dim, N]
        x = x.transpose(1, 2)  # [B, N, embed_dim]
        return x


class MiniViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_classes=10, depth=6, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, (img_size // patch_size) ** 2 + 1, embed_dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=depth
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]
        x = x + self.pos_embed[:, :x.size(1)]  # positional encoding

        x = self.transformer(x)  # [B, N+1, D]
        cls_out = x[:, 0]  # CLS token output
        return self.mlp_head(cls_out)  # [B, num_classes]


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512

    transform = transforms.Compose([transforms.ToTensor()])
    input_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    train_len = int(0.9 * len(input_dataset))
    val_len = len(input_dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(
        input_dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=4, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=4)

    model = MiniViT(img_size=32, patch_size=4, in_channels=3, embed_dim=128,
                    num_classes=10, depth=3, num_heads=4).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler()  # 🔥 AMP GradScaler

    best_acc = 0
    patience = 7
    patience_counter = 0

    for epoch in range(1, 51):
        print(f"\nEpoch {epoch}")
        model.train()
        running_loss = 0.0
        for data, target in tqdm(train_loader, desc="Training"):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f"Training Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(device), target.to(device)
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                val_loss += loss.item()
                _, pred = output.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        acc = correct / total
        print(
            f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")

    # Load best model for testing
    model.load_state_dict(torch.load("best_model.pth"))

    # Final Test Evaluation
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            test_loss += loss.item()
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_acc = correct / total
    print(
        f"\nTest Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
