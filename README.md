# DeepMeshFlow
This is a PyTorch reproduction of Paper: [DeepMeshFlow: Content Adaptive Mesh Deformation for Robust Image Registration](https://arxiv.org/abs/1912.05131)
## Required Packages
- pytorch 1.10.0
- torchsummary
- torchgeometry
- opencv-python
- numpy
- scipy
## Modification of torchgeometry
Function ```torch.gesv``` used in torchgeometry is no longer adopted by pytorch of high version, so the code in torchgeometry/core/imgwarp.py (line 258) should be adjusted as follow:
```python
# original code
X, LU = torch.gesv(b, A)
# changed code
X = torch.linalg.solve(A, b)
```
## Fomulations
**coordinates**:
- y - height - vertical axis
- x - width - horizontal axis
- pytorch data format: (NCHW)

## Code Issues:
- Problem in spatial transformer function:  
When using identity homography matrix, warping result is not the same as input image. This problem can be fixed by padding pixel (width 1) around image. But whether this solution is the best needs to be further discussed.
- Unregular homography transformation will cause negative values in transformed tensor.
- Applying homography transformation to each pair of four points to achieve warping is infeasible.
