import numpy as np

from aoa_sandbox.utils import quat_rotate


def triangulate(aoas, positions):
    """
    Least-squares triangulation from multiple sensors.
    aoas: list of direction vectors (unit vectors) from each sensor
    positions: list of sensor positions
    Returns:
        Estimated source position as a 1D numpy array of shape (3,)
    """
    A = []
    b = []
    for aoa, pos in zip(aoas, positions):
        aoa = np.asarray(aoa)
        pos = np.asarray(pos)
        P = np.eye(3) - np.outer(aoa, aoa)
        A.append(P)
        b.append(P @ pos)

    A = np.concatenate(A, axis=0)
    b = np.concatenate(b, axis=0)

    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x
