# Generates reference fixtures for tests/vision_parity.rs.
#
# For every covered torchvision architecture this saves the (randomly
# initialized) state_dict in safetensors format together with a fixed input
# and the eval-mode output. The Rust side loads the same weights into the tch
# implementation and compares outputs: both sides run the same libtorch CPU
# kernels, so any divergence beyond float rounding points at an architecture
# mismatch.
#
# Usage (any Python env with torch/torchvision/safetensors/numpy):
#   uv run --python <venv> tests/generate_vision_parity.py [out_dir]
# then:
#   TCH_VISION_PARITY_DIR=<out_dir> cargo test --release --test vision_parity -- --ignored --nocapture
import sys

import numpy as np
import torch
import torchvision
from safetensors.torch import save_file

out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tv_parity"


def export(name, model, size):
    model.eval()
    # Randomize the batch-norm running stats: with the default zeros/ones the
    # normalization is close to a no-op and would mask eps/momentum mistakes.
    g = torch.Generator().manual_seed(42)
    sd = model.state_dict()
    for k, v in sd.items():
        if k.endswith("running_mean"):
            sd[k] = torch.randn(v.shape, generator=g) * 0.2
        elif k.endswith("running_var"):
            sd[k] = torch.rand(v.shape, generator=g) + 0.5
    model.load_state_dict(sd)
    x = torch.randn(1, 3, size, size, generator=g)
    with torch.no_grad():
        y = model(x)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, f"{out_dir}/{name}.safetensors")
    np.save(f"{out_dir}/{name}_in.npy", x.numpy())
    np.save(f"{out_dir}/{name}_out.npy", y.numpy())
    print(f"{name}: out shape {tuple(y.shape)} max |y| {float(y.abs().max()):.4f}")


m = torchvision.models
export("alexnet", m.alexnet(), 224)
export("vgg16", m.vgg16(), 224)
export("vgg16_bn", m.vgg16_bn(), 224)
export("resnet18", m.resnet18(), 224)
export("resnet50", m.resnet50(), 224)
export("densenet121", m.densenet121(), 224)
export("mobilenet_v2", m.mobilenet_v2(), 224)
export("squeezenet1_0", m.squeezenet1_0(), 224)
export("squeezenet1_1", m.squeezenet1_1(), 224)
# tch's inception does not implement the aux classifier (eval-mode identical)
# nor transform_input (pretrained torchvision weights expect that input
# renormalization; disable it for the architecture comparison).
export("inception_v3", m.inception_v3(aux_logits=False, transform_input=False, init_weights=True), 299)
export("efficientnet_b0", m.efficientnet_b0(), 224)
export("efficientnet_b4", m.efficientnet_b4(), 224)
export("efficientnet_b5", m.efficientnet_b5(), 224)
