import numpy as np

try:
    from scipy.optimize import least_squares

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def fit_exponential(F, J, n_grid=200):
    """
    Fit J(F) using power law model J = A * F^B.

    This handles zero or near-zero slope at F=0 (when B < 1), and captures
    concave growth behavior common in materials. Uses nonlinear least-squares
    if SciPy available, else grid-search over B.
    Returns (A, B, C=0) and a callable model(Fq).
    Note: returns C=0 for compatibility with existing code.
    """
    F = np.asarray(F, dtype=float)
    J = np.asarray(J, dtype=float)
    if F.size == 0:
        raise ValueError("Empty F array")

    # Shift negative forces if present
    Fmin = np.nanmin(F)
    if Fmin < 0:
        F = F - Fmin

    # Power law fitting: J = A * F^B
    if SCIPY_AVAILABLE:

        def residuals(p):
            A, B = p
            if A <= 0 or B <= 0:
                return np.full_like(F, 1e10)
            return A * np.power(F, B) - J

        # Initial guess: assume near-linear, so B ~ 1
        A0 = float(np.nanmax(J)) / max(1.0, float(np.nanmax(F)))
        B0 = 0.8
        x0 = np.array([A0, B0], dtype=float)
        lb = [1e-6, 0.1]
        ub = [np.inf, 3.0]
        res = least_squares(
            residuals, x0, bounds=(lb, ub), method="dogbox", ftol=1e-12, xtol=1e-12
        )
        A_fit, B_fit = float(res.x[0]), float(res.x[1])

        def model(Fq):
            Fq = np.asarray(Fq, dtype=float)
            return A_fit * np.power(Fq, B_fit)

        return (A_fit, B_fit, 0.0), model

    # Fallback: grid-search B and linear LS for A
    nbins = min(40, max(5, F.size // 5))
    use_F = F
    use_J = J
    if F.size > nbins * 2:
        edges = np.linspace(float(np.nanmin(F)), float(np.nanmax(F)), nbins + 1)
        inds = np.digitize(F, edges) - 1
        bin_means_F = []
        bin_means_J = []
        for b in range(nbins):
            mask = inds == b
            if not np.any(mask):
                continue
            bin_means_F.append(np.mean(F[mask]))
            bin_means_J.append(np.mean(J[mask]))
        if len(bin_means_F) >= 3:
            use_F = np.array(bin_means_F, dtype=float)
            use_J = np.array(bin_means_J, dtype=float)

    B_grid = np.linspace(0.3, 1.5, n_grid)
    best = None
    for B in B_grid:
        phi = np.power(use_F, B)
        M = phi.reshape(-1, 1)
        sol, *_ = np.linalg.lstsq(M, use_J, rcond=None)
        A = sol[0]
        if A <= 0:
            continue
        pred = A * phi
        sse = np.sum((use_J - pred) ** 2)
        if best is None or sse < best[0]:
            best = (sse, A, B)

    if best is None:
        raise RuntimeError("Power law fit failed")
    _, A_fit, B_fit = best

    def model(Fq):
        Fq = np.asarray(Fq, dtype=float)
        return A_fit * np.power(Fq, B_fit)

    return (A_fit, B_fit, 0.0), model


def fit_tanh(x, J, n_grid=200):
    """
    Fit J(x) = A * tanh(B * x) + C by grid-search over B and linear least-squares for A,C.
    Returns (A, B, C) and a callable.
    """
    x = np.asarray(x, dtype=float)
    J = np.asarray(J, dtype=float)
    if x.size == 0:
        raise ValueError("Empty x array")

    if SCIPY_AVAILABLE:
        # Fit A,B,C via nonlinear least squares
        def residuals(p):
            A, B, C = p
            return A * np.tanh(B * x) + C - J

        A0 = 0.5 * (np.nanmax(J) - np.nanmin(J))
        B0 = 1.0
        C0 = float(np.nanmean(J))
        x0 = np.array([A0, B0, C0], dtype=float)
        # Allow A and C to be any sign, B >= 0
        lb = [-np.inf, 0.0, -np.inf]
        ub = [np.inf, np.inf, np.inf]
        res = least_squares(
            residuals, x0, bounds=(lb, ub), method="trf", ftol=1e-9, xtol=1e-9
        )
        A_fit, B_fit, C_fit = float(res.x[0]), float(res.x[1]), float(res.x[2])

        def model(xq):
            xq = np.asarray(xq, dtype=float)
            return A_fit * np.tanh(B_fit * xq) + C_fit

        return (A_fit, B_fit, C_fit), model

    # Fallback: grid-search B and linear solve for A,C
    B_grid = np.logspace(-3, 1.0, n_grid)
    best = None
    for B in B_grid:
        T = np.tanh(B * x)
        M = np.vstack([T, np.ones_like(T)]).T
        sol, *_ = np.linalg.lstsq(M, J, rcond=None)
        A, C = sol[0], sol[1]
        pred = A * T + C
        sse = np.sum((J - pred) ** 2)
        if best is None or sse < best[0]:
            best = (sse, A, B, C)

    if best is None:
        raise RuntimeError("Tanh fit failed")
    _, A_fit, B_fit, C_fit = best

    def model(xq):
        xq = np.asarray(xq, dtype=float)
        return A_fit * np.tanh(B_fit * xq) + C_fit

    return (A_fit, B_fit, C_fit), model


def fit_polynomial(x, y, degree=3, force_through_origin=False):
    """
    Fit J(x) using polynomial model J = p[0]*x^degree + p[1]*x^(degree-1) + ... + p[degree].

    If force_through_origin=True, forces the polynomial to pass through (0,0),
    effectively removing the constant term. This is physically correct for
    energy release rate where J=0 when F=0.

    Uses numpy.polyfit for least-squares fitting, or manual lstsq if forcing origin.

    Returns: (coefficients_tuple, callable_model)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError("Empty x or y array")

    if force_through_origin:
        # Build design matrix without constant term: [x^degree, x^(degree-1), ..., x^1]
        # This forces J(0) = 0
        M = np.vstack([np.power(x, degree - i) for i in range(degree)]).T
        coeffs_reduced, *_ = np.linalg.lstsq(M, y, rcond=None)
        # Pad with 0 for the constant term to keep polyval compatible
        coeffs = tuple(list(coeffs_reduced) + [0.0])
    else:
        # Standard polyfit allows constant term
        coeffs = np.polyfit(x, y, degree)

    def model(xq):
        xq = np.asarray(xq, dtype=float)
        return np.polyval(coeffs, xq)

    return coeffs, model
