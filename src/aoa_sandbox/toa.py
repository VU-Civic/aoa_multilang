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


def estimate_position_from_toa(sensor_positions, toas, c=343.0):
    sensor_positions = np.asarray(sensor_positions)
    toas = np.asarray(toas)

    # Reference sensor: remove global offset
    ref_idx = 0
    ref_toa = toas[ref_idx]
    rel_toas = toas - ref_toa
    rel_dists = rel_toas * c

    # Work in local ENU around first sensor
    ref = sensor_positions[ref_idx]
    R = ecef_to_enu_matrix(ref)
    sensors_enu = (R @ (sensor_positions - ref).T).T  # Nx3

    def residual(x):
        # predicted distances relative to ref
        dists = np.linalg.norm(sensors_enu - x, axis=1)
        rel_pred = dists - dists[ref_idx]
        return rel_pred - rel_dists

    # initial guess: somewhere near sensors
    x0 = np.mean(sensors_enu, axis=0)
    res = least_squares(residual, x0)
    est_enu = res.x

    # back to ECEF
    est_ecef = ref + R.T @ est_enu
    return est_ecef
