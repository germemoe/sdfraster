#!/usr/bin/env python
import torch
import sys

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

print("\nTesting gDel3D import...")
try:
    from gdel3d import Del
    print("✓ gDel3D import successful")

    print("\nTesting gDel3D computation...")
    num_points = 20
    torch.manual_seed(189710237)
    vertices = torch.rand((num_points, 3), device='cpu').double()

    v = Del(vertices.shape[0])
    indices_np, prev = v.compute(vertices)
    print(f"✓ gDel3D computation successful! Generated {indices_np.shape[0]} tetrahedra")
    del v

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
