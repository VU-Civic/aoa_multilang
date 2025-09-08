

def estimate_position_from_toa(sensor_positions, toas, c=343.0):
    """
    Estimate source position from ToA measurements.
    sensor_positions: list of shape (N,3)
    toas: list of shape (N,)
    """
    import numpy as np
    sensor_positions = np.asarray(sensor_positions)
    toas = np.asarray(toas)
    dists = c * toas

    # Use nonlinear least-squares
    def residual(x):
        res = np.linalg.norm(sensor_positions - x, axis=1) - dists
        print("residuals:", res)
        return res

    from scipy.optimize import least_squares
    x0 = np.mean(sensor_positions, axis=0)  # init guess
    res = least_squares(residual, x0)
    return res.x
