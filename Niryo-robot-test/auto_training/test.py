import numpy as np

import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("flash built:", torch.backends.cuda.is_flash_attention_available())
print("flash enabled:", torch.backends.cuda.flash_sdp_enabled())
print("mem efficient enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math enabled:", torch.backends.cuda.math_sdp_enabled())
print("cudnn enabled:", torch.backends.cuda.cudnn_sdp_enabled())
print("cudnn version:", torch.backends.cudnn.version())