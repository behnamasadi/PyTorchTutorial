import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()

        self.num_patches = (img_size//patch_size)**2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        pass

    def forward(self, x):
        # x.size() B,C,H,W
        # B, C, H, W = x.size()
        # H is usually 224 and patch_size os 16 so 14**2=196
        # W',H' are 14

        x = self.proj(x)  # [B,embed_dim, W',H']
        x = x.flatten(2)  # [B,embed_dim, N=W'*H']
        x = x.permute(0, 2, 1)  # [B, N=W'*H', embed_dim ]

        return x


class MiniVit():
    def __init__(self, img_size=224, in_channels=3, patch_size=16, embed_dim=768, num_heads=8, num_classes=10, depth=6):
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 1,1,768
        self.pos_embed = nn.Parameter(torch.zeros(
            1, (img_size//patch_size)**2+1, embed_dim))  # 1,196+1,768

        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True), num_layers=depth)

        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(
            in_features=embed_dim, out_features=num_classes))

        pass

    def forward(self, x):
        B = x.size(0)
        x = patch_embed(x)  # [B,N=196+1 , embed_dim]

        # cls_token is initially 1,1,768 =>  [B, -1, -1] meaning [B, 1, 768]
        self.cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((self.cls_token, x), dim=1)

        x = x + pos_embed[:, :x.size(1), :]  # positional encoding
        x = self.transformer(x)  # [B, N+1, D]

        pass


# patch_embed = PatchEmbedding()
# B, C, H, W = 1, 3, 224, 224
# img = torch.randn(B, C, H, W)
# output = patch_embed(img)
# print(output.shape)


# x = torch.tensor([[1], [2], [3]])
# print(x)
# print(x.shape)
# print(x.size())

# y = x.expand(-1, 2)

# x[0, 0] = 5

# print(x)
# print(y)

embed_dim = 3
cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 1,1,3
print(cls_token.shape)
B = 2
# cls_token = cls_token.expand(B, -1, -1)
cls_token = cls_token.expand(-1, B, -1)

print(cls_token.shape)
print(cls_token)


img_size = 224
patch_size = 16
embed_dim = 768

pos_embed = nn.Parameter(torch.zeros(
    1, (img_size//patch_size)**2+1, embed_dim))  # 1,196+1,768


B = 64
N = 196
D = 768
x = torch.randn(B, N+1, D)  # [B, N+1, D]

print("x"*50)
print(pos_embed.shape)
print(x.size(1))
tmp = pos_embed[:, :x.size(1)]
print(tmp.shape)


tmp = pos_embed[:, :, :x.size(1)]
print(tmp.shape)


# pos_embed[:, :x.size(1)] and not pos_embed[:, :, :x.size(1)]