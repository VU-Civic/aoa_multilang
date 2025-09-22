import numpy as np
from scipy.signal import correlate

C = 343.0  # speed of sound [m/s]


def gcc_phat(sig, refsig, fs, max_tau=None, interp=1):
    """
    Compute time delay estimation between sig and refsig using GCC-PHAT.
    Returns the delay in seconds.
    """
    n = sig.shape[0] + refsig.shape[0]

    # FFT
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    # PHAT weighting
    R /= np.abs(R) + 1e-15

    # IFFT
    cc = np.fft.irfft(R, n=(interp * n))
    max_shift = int(interp * n / 2)

    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    delay = shift / float(interp * fs)
    return delay


def estimate_aoa(signals, fs, mic_positions, source_pos=None, sensor_pos=None):
    """
    Estimate AoA vector from microphone signals.
    Args:
        signals: list of time-domain signals (already aligned in length)
        fs: sample rate
        mic_positions: (N_mics, 3) array of mic positions in ECEF
    Returns:
        aoa_vec: unit vector in ECEF pointing toward source
    """
    N = len(signals)
    if N < 2:
        raise ValueError("Need at least 2 microphones for AoA")

    # Compute TDOAs using GCC-PHAT between mic 0 and others
    ref = signals[0]
    tdoas = []
    baselines = []
    for i in range(1, N):
        tau = gcc_phat(ref, signals[i], fs)
        tdoas.append(tau)
        baselines.append(mic_positions[i] - mic_positions[0])

    # Solve least-squares for direction vector
    A = []
    b = []
    c = 343.0  # speed of sound
    for tau, baseline in zip(tdoas, baselines):
        baseline = np.asarray(baseline)
        # Equation: (baseline · aoa) = c * tau
        A.append(baseline.reshape(1, -1))
        b.append(c * tau)
    A = np.vstack(A)
    b = np.array(b)

    # Check if microphones are approximately coplanar
    plane_normal = find_plane_normal(mic_positions)

    # Check if microphones are approximately coplanar
    if is_coplanar(mic_positions):
        return solve_coplanar_aoa(A, b, plane_normal, source_pos, sensor_pos)
    else:
        return solve_general_aoa(A, b)


def is_coplanar(mic_positions, tolerance=1e-3):
    """Check if microphones lie approximately in a plane."""
    if len(mic_positions) <= 3:
        return True

    # Fit plane to first 3 points
    p1, p2, p3 = mic_positions[:3]
    v1, v2 = p2 - p1, p3 - p1
    normal = np.cross(v1, v2)

    if np.linalg.norm(normal) < tolerance:
        return True  # First 3 points are collinear

    normal = normal / np.linalg.norm(normal)

    # Check if remaining points lie in the plane
    for p in mic_positions[3:]:
        distance = abs(np.dot(p - p1, normal))
        if distance > tolerance:
            return False

    return True


def find_plane_normal(mic_positions, tolerance=1e-3):
    """
    Find the normal vector to the plane containing the microphones.
    Returns None if mics are not coplanar.
    """
    if len(mic_positions) <= 3:
        if len(mic_positions) == 3:
            # Compute normal from 3 points
            p1, p2, p3 = mic_positions
            v1, v2 = p2 - p1, p3 - p1
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) > tolerance:
                return normal / np.linalg.norm(normal)
        return np.array([0, 0, 1])  # Default for <3 points

    # For >3 points, use SVD to find the best-fit plane
    centered = mic_positions - np.mean(mic_positions, axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=True)

    # If the smallest singular value is small, points are coplanar
    if S[-1] < tolerance * S[0]:
        normal = Vt[-1, :]  # Last row of Vt is the normal
        return normal / np.linalg.norm(normal)

    return None  # Not coplanar


def solve_coplanar_aoa(A, b, plane_normal, source_pos=None, sensor_pos=None):
    """
    Solve AoA for coplanar microphones.
    Returns two solutions differing in the plane-normal component.
    """

    # Create orthonormal basis for the microphone plane
    # plane_normal is perpendicular to the plane
    if abs(plane_normal[2]) < 0.9:
        temp = np.array([0, 0, 1])
    else:
        temp = np.array([1, 0, 0])

    u1 = np.cross(plane_normal, temp)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(plane_normal, u1)
    u2 = u2 / np.linalg.norm(u2)

    # Project the baseline vectors onto the plane
    A_plane = np.column_stack([A @ u1, A @ u2])

    # Solve the 2D problem in the plane
    # This determines the in-plane component of the AoA vector
    if np.linalg.matrix_rank(A_plane) >= 2:
        # Overdetermined case - use least squares
        alpha_beta, residuals, rank, s = np.linalg.lstsq(
            A_plane, b, rcond=None)
    else:
        # Underdetermined case - find minimum norm solution
        alpha_beta = np.linalg.pinv(A_plane) @ b

    alpha, beta = alpha_beta

    # The in-plane component of the AoA vector
    aoa_in_plane = alpha * u1 + beta * u2

    # Now we need to find the plane-normal component such that ||aoa|| = 1
    norm_in_plane_sq = np.linalg.norm(aoa_in_plane)**2

    if norm_in_plane_sq > 1:
        # Scale down the in-plane component
        aoa_in_plane = aoa_in_plane / np.sqrt(norm_in_plane_sq)
        gamma = 0
        aoa_1 = aoa_in_plane
        aoa_2 = aoa_in_plane
    else:
        # Two solutions differing by the sign of the plane-normal component
        gamma = np.sqrt(1 - norm_in_plane_sq)

        aoa_1 = aoa_in_plane + gamma * plane_normal
        aoa_2 = aoa_in_plane - gamma * plane_normal

        # Verify they're unit vectors
        aoa_1 = aoa_1 / np.linalg.norm(aoa_1)
        aoa_2 = aoa_2 / np.linalg.norm(aoa_2)

    if source_pos is not None and sensor_pos is not None:
        # Choose the solution that points toward the source
        sensor_to_source = source_pos - sensor_pos
        sensor_to_source = sensor_to_source / np.linalg.norm(sensor_to_source)
        if np.dot(aoa_1, sensor_to_source) > np.dot(aoa_2, sensor_to_source):
            return aoa_1
        else:
            return aoa_2

    return aoa_1


def solve_general_aoa(A, b):
    """Solve AoA for non-coplanar microphones."""

    def residual_func(aoa_vec):
        return A @ aoa_vec - b

    def constraint_func(aoa_vec):
        return np.linalg.norm(aoa_vec)**2 - 1

    # Initial guess from unconstrained solution
    aoa_init, *_ = np.linalg.lstsq(A, b, rcond=None)
    aoa_init = aoa_init / np.linalg.norm(aoa_init)

    from scipy.optimize import minimize

    result = minimize(
        lambda aoa: np.sum(residual_func(aoa)**2),
        aoa_init,
        constraints={'type': 'eq', 'fun': constraint_func},
        method='SLSQP'
    )

    if result.success:
        aoa_vec = result.x / np.linalg.norm(result.x)  # ensure normalization
        return aoa_vec
    else:
        # Fallback
        return aoa_init
