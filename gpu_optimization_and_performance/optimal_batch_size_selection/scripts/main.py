import torch
import timm
from torch.utils.data import TensorDataset, DataLoader


def find_optimal_batch_size(
    model,
    sample_batch,
    criterion,
    max_batch_size=512,
    dtype=torch.float16,
):
    """
    Find the largest batch size that fits into GPU memory.
    Tests both forward and backward passes using AMP autocast.

    sample_batch: one (inputs, labels) batch from dataloader
    criterion: loss function, e.g. nn.CrossEntropyLoss()
    """
    model.eval()
    device = next(model.parameters()).device

    inputs, labels = sample_batch
    one_input = inputs[:1].to(device)
    one_label = labels[:1].to(device)

    low, high = 1, max_batch_size
    best = 1

    while low <= high:
        mid = (low + high) // 2

        try:
            # Construct synthetic batch of size `mid`
            test_inputs = one_input.repeat(mid, 1, 1, 1)
            test_labels = one_label.repeat(mid)

            # Clear gradients
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            # Forward + backward under AMP
            with torch.amp.autocast('cuda', dtype=dtype):
                output = model(test_inputs)
                loss = criterion(output, test_labels)

            loss.backward()

            print(f"✓ Fits: {mid}")
            best = mid
            low = mid + 1

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"✗ OOM at: {mid}")
                torch.cuda.empty_cache()
                high = mid - 1
            else:
                raise e

    return best


device = "cuda" if torch.cuda.is_available() else "cpu"


all_efficient_models = timm.list_models("*efficientnetv2*")
# print(all_efficient_models)

model_name = "tf_efficientnetv2_s"
model = timm.create_model(model_name=model_name, pretrained=True).to(device)

# Get model configuration
cfg = model.default_cfg
print(f"Input size: {cfg['input_size']}")


C, H, W = list(cfg['input_size'])
B = 10
print(B, C, H, W)


num_class = 10

X = torch.randn(1, C, H, W)
Y = torch.randint(0, num_class, (1,))

dataset = TensorDataset(X, Y)

dataloader = DataLoader(batch_size=1, dataset=dataset,
                        pin_memory=True, num_workers=4)


# Get a sample batch from dataloader
sample_batch = next(iter(dataloader))
print(
    f"Sample batch shapes: inputs={sample_batch[0].shape}, labels={sample_batch[1].shape}")


# Detect optimal dtype
if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
    print("dtype is bf16")
else:
    dtype = torch.float16
    print("dtype is f16")


# or

use_bf16 = torch.cuda.is_bf16_supported()

if use_bf16:
    dtype = torch.bfloat16
    scaler = None  # no need for scaling
else:
    dtype = torch.float16
    scaler = torch.amp.GradScaler()  # needed for fp16


max_batch_size = 512

criterion = torch.nn.CrossEntropyLoss()

optimal_batch_size = find_optimal_batch_size(model,
                                             sample_batch,
                                             criterion,
                                             max_batch_size,
                                             dtype,
                                             )


print(f"Optimal Batch Size: {optimal_batch_size}")
# for name, params in model.named_parameters():
#     print(name)
