from unittest import result
import numpy as np
from scipy.optimize import least_squares


def ecef_to_enu_matrix(ref):
    """Return ECEF->ENU rotation matrix for reference point (lat, lon)."""
    x, y, z = ref
    lon = np.arctan2(y, x)
    hyp = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, hyp)
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    R = np.array([
        [-slon,  clon, 0],
        [-slat*clon, -slat*slon, clat],
        [clat*clon,  clat*slon, slat]
    ])
    return R


def estimate_position_from_toa(sensor_positions, toas, c=343.0, refine=True):
    sensor_positions = np.asarray(sensor_positions, dtype=float)
    toas = np.asarray(toas, dtype=float)
    N = sensor_positions.shape[0]
    if N < 4:
        raise ValueError("Need at least 4 sensors for 3D+t0 solution")

    ref_idx = int(np.argmin(toas))
    t_ref = toas[ref_idx]

    centroid = sensor_positions.mean(axis=0)
    P = sensor_positions - centroid
    p_ref_center = P[ref_idx]

    rows = []
    rhs = []
    for i in range(N):
        if i == ref_idx:
            continue
        pi_center = P[i]
        ti = toas[i]
        Ai = np.empty(4, dtype=float)
        Ai[0:3] = -2.0 * (pi_center - p_ref_center)
        Ai[3] = 2.0 * (c**2) * (ti - t_ref)
        bi = (c**2) * (ti**2 - t_ref**2) - (np.dot(sensor_positions[i], sensor_positions[i]) - np.dot(
            sensor_positions[ref_idx], sensor_positions[ref_idx]))
        rows.append(Ai)
        rhs.append(bi)

    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=float)

    x_lin, residuals_ls, rank, svals = np.linalg.lstsq(A, b, rcond=None)
    s_center_lin = x_lin[0:3]
    t0_lin = x_lin[3]
    s_lin = s_center_lin + centroid

    position = s_lin

    if refine:
        def residuals_nl(x):
            pos = x[:3]
            t0 = x[3]
            return (t0 + np.linalg.norm(sensor_positions - pos[None, :], axis=1) / c) - toas

        x0 = np.hstack([s_lin, t0_lin])
        lsq = least_squares(residuals_nl, x0, method='lm',
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000)
        s_ref = lsq.x[:3]
        position = s_ref

    return position
