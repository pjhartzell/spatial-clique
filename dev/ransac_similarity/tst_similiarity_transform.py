import numpy as np
from skimage.measure import ransac
from similarity3d import SixParameterTransform, SevenParameterTransform


src = np.array([[1,2,3],
                [1,3,4],
                [3,2,5],
                [3,6,2],
                [4,5,7],
                [1,3,8]])

M = np.array([[1,0,0,10],
              [0,1,0,-20],
              [0,0,1,0],
              [0,0,0,1]])

homogeneous_src = np.vstack((src.T, np.ones(src.shape[0])))
dst = (M @ homogeneous_src).T[:,0:-1]
dst = 1.1 * dst
dst[0,0] += 0.14

# mymodel, inliers = ransac(
#     (src, dst),
#     SevenParameterTransform,
#     residual_threshold=0.1,
#     min_samples=3
# )

# print(mymodel.params)
# print(inliers)

test = SevenParameterTransform()
test.estimate(src, dst)
print(test.params)