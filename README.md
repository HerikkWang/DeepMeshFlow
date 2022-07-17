# DeepMeshFlow

## Required Packages
- pytorch 1.10.0
- torchsummary
- torchgeometry
- opencv
- numpy
## Modification of torchgeometry
Function ```torch.gesv``` used in torchgeometry is no longer adopted by pytorch of high version, so the code in torchgeometry/core/imgwarp.py (line 258) should be adjusted as follow:
```python
# original code
X, LU = torch.gesv(b, A)
# changed code
X = torch.linalg.solve(A, b)
```
## fomulations:
**coordinates**:
- x - height - vertical axis
- y - width - horizontal axis
- pytorch data format: (NCHW)

