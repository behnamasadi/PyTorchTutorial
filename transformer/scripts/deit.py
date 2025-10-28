# fmt: off
# isort: skip_file
# DO NOT reorganize imports - warnings filter must be FIRST!

import warnings
import os

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import torch.nn.functional as F
from timm import create_model
# fmt: on


# --------------------------------------------
# 1. Load Teacher (CNN) and Student (DeiT)
# --------------------------------------------
# Teacher: pretrained CNN (frozen)
# Note: regnety_160 = RegNetY-16GF (160 = 16.0 GFLOPs)

# regnety_016.pycls_in1k	1.6 GF	~11M
# regnety_032.pycls_in1k	3.2 GF	~20M
# regnety_080.pycls_in1k	8.0 GF	~39M
# regnety_160.pycls_in1k	16 GF	84M ← You're using this
# regnety_320.pycls_in1k	32 GF	145M

teacher = create_model('regnety_160.pycls_in1k', pretrained=True)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

# Student: Vision Transformer with [CLS] + [DIST] tokens
student = create_model(
    'deit_base_distilled_patch16_224.fb_in1k', pretrained=False)
# Enable distillation mode to return (cls_logits, dist_logits)
student.distilled_training = True
student.train()  # Must be in training mode to get both cls and dist logits

# Example input (batch of 4 images, 3×224×224)
x = torch.randn(4, 3, 224, 224)
# random ground-truth labels, since we have 1k classes in ImageNet, classes are between 0-1000 and the shape should be (4,)
y = torch.randint(0, 1000, (4,))

# --------------------------------------------
# 2. Forward Pass
# --------------------------------------------
with torch.no_grad():
    teacher_logits = teacher(x)   # [B, num_classes]

print(teacher_logits)

# Student forward:
# timm's DeiT returns:
#   - if distilled: a tuple (cls_logits, dist_logits)
#   - if not distilled: a single tensor
out = student(x)

if isinstance(out, tuple):
    print("distilled")
    cls_logits, dist_logits = out
else:
    print("non-distilled")
    cls_logits = dist_logits = out  # fallback (non-distilled)

# --------------------------------------------
# 3. Define Distillation Loss
# --------------------------------------------


def deit_distillation_loss(cls_logits, dist_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """
    DeiT-style distillation loss
    """
    # Cross-entropy with ground truth (for [CLS] token)
    ce_loss = F.cross_entropy(cls_logits, labels)

    # KL divergence with teacher soft targets (for [DIST] token)
    p_s = F.log_softmax(dist_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    kd_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)

    # Weighted combination
    return (1 - alpha) * ce_loss + alpha * kd_loss


# --------------------------------------------
# 4. Compute total loss
# --------------------------------------------
loss = deit_distillation_loss(cls_logits, dist_logits, teacher_logits, y)
print(f"Total training loss: {loss.item():.4f}")

# --------------------------------------------
# 5. Backpropagation
# --------------------------------------------
optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
optimizer.zero_grad()
loss.backward()
optimizer.step()
