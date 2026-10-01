"""The three ViT fine-tuning strategies on one model, one training step each.

  A. frozen backbone + trained head
  B. full fine-tune with layer-wise learning-rate decay (LLRD)
  C. LoRA adapters on the attention projections (base weights frozen)

Companion to ../vit_fine_tuning.ipynb.

    python vit_ft_examples.py --device cuda        # or cpu
    python vit_ft_examples.py --only B

Needs `transformers` (and `peft` for C). The weights are downloaded from the
Hugging Face hub on first use; set HF_HUB_OFFLINE=1 to use a warm cache only.
"""
import argparse
import re
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


class DinoSeg(nn.Module):
    """A ViT backbone with a small conv segmentation head, frozen or not."""

    def __init__(self, n_classes: int, freeze: bool = True, layers=(3, 6, 9, 12)):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(NAME)
        self.freeze = freeze
        if freeze:
            self.backbone.eval().requires_grad_(False)       # 1. freeze
        cfg = self.backbone.config
        self.layers, self.patch = layers, cfg.patch_size
        self.skip = 1 + cfg.num_register_tokens              # CLS + register tokens
        d = cfg.hidden_size
        self.head = nn.Sequential(                           # 2. new, always trained
            nn.Conv2d(d * len(layers), 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, n_classes, 1))

    def train(self, mode: bool = True):
        """A frozen backbone stays in eval mode even when the model is set to train."""
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, x):                                    # x: (B,3,H,W), H,W % patch == 0
        B, _, H, W = x.shape
        with torch.no_grad() if self.freeze else nullcontext():   # 3. no backbone graph
            hs = self.backbone(pixel_values=x, output_hidden_states=True).hidden_states
        h, w = H // self.patch, W // self.patch
        maps = [self.backbone.norm(hs[i])[:, self.skip:]     # drop CLS + registers
                .transpose(1, 2).reshape(B, -1, h, w) for i in self.layers]
        logits = self.head(torch.cat(maps, 1))
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


def llrd_param_groups(model, head_lr=1e-4, backbone_lr=5e-5, decay=0.8, wd=1e-4):
    """One optimiser group per parameter: deeper backbone layers get a larger LR."""
    n = model.backbone.config.num_hidden_layers                  # 12 for ViT-B
    groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            m = re.search(r"\.layer\.(\d+)\.", name)
            layer_id = int(m.group(1)) + 1 if m else (0 if "embeddings" in name else n + 1)
            lr = backbone_lr * decay ** (n + 1 - layer_id)
        else:
            lr = head_lr
        no_wd = p.ndim == 1 or "token" in name                   # norms, biases, CLS/registers
        groups.append({"params": [p], "lr": lr, "weight_decay": 0.0 if no_wd else wd})
    return groups


def add_lora(model, r=8):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model.backbone = get_peft_model(model.backbone, cfg)         # base weights stay frozen
    return model


def one_step(model, opt, device, size=224, n_classes=3):
    """A single forward/backward on random data, just to prove the wiring works."""
    x = torch.randn(1, 3, size, size, device=device)
    y = torch.randint(0, 2, (1, n_classes, size, size), device=device).float()
    loss = F.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=224, help="must divide by the patch size (16)")
    ap.add_argument("--only", choices=["A", "B", "C"], help="run a single strategy")
    args = ap.parse_args()
    dev, run = args.device, (lambda k: args.only in (None, k))

    if run("A"):                                             # frozen backbone + head
        model = DinoSeg(n_classes=3).train().to(dev)
        opt = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
        print("A trainable", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("A step loss %.4f" % one_step(model, opt, dev, args.size))

    if run("B"):                                             # full fine-tune with LLRD
        model = DinoSeg(n_classes=3, freeze=False).train().to(dev)
        groups = llrd_param_groups(model)
        opt = torch.optim.AdamW(groups)
        lrs = [g["lr"] for g in groups]
        print("B lr range %.1e..%.1e groups %d" % (min(lrs), max(lrs), len(groups)))
        print("B backbone grad", next(model.backbone.parameters()).requires_grad)
        print("B step loss %.4f" % one_step(model, opt, dev, args.size))

    if run("C"):                                             # LoRA
        model = add_lora(DinoSeg(n_classes=3, freeze=False)).train().to(dev)
        model.backbone.print_trainable_parameters()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        print("C step loss %.4f" % one_step(model, opt, dev, args.size))


if __name__ == "__main__":
    main()
