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
        aoa = aoa / np.linalg.norm(aoa)
        P = np.eye(3) - np.outer(aoa, aoa)
        A.append(P)
        b.append(P @ pos)

    A = np.concatenate(A, axis=0)
    b = np.concatenate(b, axis=0)

    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x


def triangulate_quat(aoas_local, positions, quaternions):
    """
    Triangulate using multiple sensors in ECEF with orientation.

    aoas_local: list of local AoA vectors (unit vectors in sensor frame)
    positions: list of sensor positions in ECEF
    quaternions: list of sensor orientations as quaternions (w,x,y,z), 
                 rotating local -> ECEF
    """
    A = []
    b = []
    for aoa_local, pos, q in zip(aoas_local, positions, quaternions):
        aoa_ecef = quat_rotate(q, aoa_local)
        aoa_ecef /= np.linalg.norm(aoa_ecef)
        P = np.eye(3) - np.outer(aoa_ecef, aoa_ecef)
        A.append(P)
        b.append(P @ pos)

    A = np.concatenate(A, axis=0)
    b = np.concatenate(b, axis=0)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x
