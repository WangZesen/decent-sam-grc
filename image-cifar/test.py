import torch
import torch_xla

print(torch_xla.__version__)  # type: ignore
print(torch.tensor(1.0, device='xla').device)
