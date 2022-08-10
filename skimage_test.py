from ast import Del
import numpy as np
# from skimage.transform import PiecewiseAffineTransform, warp
import cv2
from scipy import spatial
import torch
# from skimage import io

# im = io.imread("im_test/square.jpg")

# src_cols = np.linspace(0, 2880, 3)
# src_rows = np.linspace(0, 2880, 3)
# src_rows, src_cols = np.meshgrid(src_rows, src_cols)
# src = np.dstack([src_cols.flat, src_rows.flat])[0]
# print(src)
# dst = src.copy()
# dst[4] = [1940., 1940.]
# dst[0] = [100., 100.]
# dst[8] = [2780., 2780.]
# # spatial.Delaunay()
# tform = PiecewiseAffineTransform()
# tform.estimate(src, dst)
# out = warp(im, tform)
# io.imsave("res.jpg", out)

from torch_geometric.data import Data
from torch_geometric.transforms import Delaunay
# from scipy.spatial import Delaunay
pos = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=torch.float)
data = Data(pos=pos)
data = Delaunay()(data)
print(data)