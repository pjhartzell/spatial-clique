import numpy as np

class SevenParameterTransform:
    """Purpose for creating this class is to be able to pass a 2D or 3D (rather
    than only 2D) Similarity model to skimage's RANSAC method. See https://
    scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.ransac.
    But you can also use it to solve a 2D or 3D Similarity transform via
    Umeyama's method.
    """
    def __init__(self):
        self.estimate_scale = True

    def estimate(self, src, dst):
        self.params = self._umeyama(src, dst, self.estimate_scale)

        return True

    def residuals(self, src, dst):
        homogenous_src = np.vstack((src.T, np.ones(src.shape[0])))
        transformed_src = (self.params @ homogenous_src).T[:,0:-1]
        res = np.sqrt(np.sum((dst - transformed_src)**2, axis=1))

        return res

    def _umeyama(self, src, dst, estimate_scale):
        """This is a direct lift from https://github.com/scikit-image/
        scikit-image/blob/master/skimage/transform/_geometric.py

        Estimate N-D similarity transformation with or without scaling.
        Parameters
        ----------
        src : (M, N) array
            Source coordinates.
        dst : (M, N) array
            Destination coordinates.
        estimate_scale : bool
            Whether to estimate scaling factor.
        Returns
        -------
        T : (N + 1, N + 1)
            The homogeneous similarity transformation matrix. The matrix contains
            NaN values only if the problem is not well-conditioned.
        References
        ----------
        .. [1] "Least-squares estimation of transformation parameters between two
                point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
        """

        num = src.shape[0]
        dim = src.shape[1]

        # Compute mean of src and dst.
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        # Subtract mean from src and dst.
        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        # Eq. (38).
        A = dst_demean.T @ src_demean / num

        # Eq. (39).
        d = np.ones((dim,), dtype=np.double)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        T = np.eye(dim + 1, dtype=np.double)

        U, S, V = np.linalg.svd(A)

        # Eq. (40) and (43).
        rank = np.linalg.matrix_rank(A)
        if rank == 0:
            return np.nan * T
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:dim, :dim] = U @ V
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = U @ np.diag(d) @ V
                d[dim - 1] = s
        else:
            T[:dim, :dim] = U @ np.diag(d) @ V

        if estimate_scale:
            # Eq. (41) and (42).
            scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
        else:
            scale = 1.0

        T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
        T[:dim, :dim] *= scale

        return T


class SixParameterTransform(SevenParameterTransform):
    def __init__(self):
        self.estimate_scale = False