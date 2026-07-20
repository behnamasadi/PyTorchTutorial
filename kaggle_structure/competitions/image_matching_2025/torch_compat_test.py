import torch, torch.nn as nn
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
print("arch_list:", torch.cuda.get_arch_list(), flush=True)
print("has sm_60 (P100):", "sm_60" in torch.cuda.get_arch_list(), flush=True)
print("has sm_86 (3090):", "sm_86" in torch.cuda.get_arch_list(), flush=True)
print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
# real kernel test
try:
    y = nn.Conv2d(3,8,3).cuda()(torch.randn(1,3,64,64,device="cuda")); torch.cuda.synchronize()
    print("GPU conv OK", flush=True)
except Exception as e:
    print("GPU conv FAILED", str(e)[:120], flush=True)
# kornia DISK + LightGlue smoke test
try:
    import kornia as K, kornia.feature as KF
    print("kornia", K.__version__, flush=True)
    from pathlib import Path
    ds=Path("data/extracted/test/ETs")
    ps=sorted(p for p in ds.iterdir() if p.suffix==".png")[:4]
    disk=KF.DISK.from_pretrained("depth").cuda().eval()
    lg=KF.LightGlueMatcher("disk").cuda().eval()
    with torch.inference_mode():
        img=K.io.load_image(str(ps[0]),K.io.ImageLoadType.RGB32,device="cuda")[None]
        f=disk(img,2048,pad_if_not_divisible=True)[0]
        print("DISK ok, kps", f.keypoints.shape, flush=True)
    import pycolmap; print("pycolmap", pycolmap.__version__, flush=True)
    print("ALL OK — torch 2.5.1 works with kornia+pycolmap on 3090", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
