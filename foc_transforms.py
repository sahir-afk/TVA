"""Reference implementation of Field-Oriented Control (FOC) math.

This module is the *golden reference* for a later hand-port to C on an
STM32G431 (single-precision float, hardware FPU). The functions are written
in a deliberately literal, branch-light style so the C translation is close
to line-for-line. All angle conventions are documented in each docstring and
are pinned by the accompanying pytest suite -- do not "fix" a sign without
also changing the tests, because the tests define the contract the C code
must reproduce bit-for-bit-ish (within tolerance).

Conventions used throughout:
  * Clarke transform is the *amplitude-invariant* (2/3 scaled) form, so a
    balanced set with peak amplitude A produces an (alpha, beta) vector of
    magnitude A (not 1.5*A).
  * Park uses theta as the rotor electrical angle; the exact rotation matrix
    is spelled out below.
  * SVPWM uses min-max (a.k.a. midpoint / zero-sequence) common-mode
    injection, which is numerically identical to classic space-vector
    modulation in the linear region but is branchless and continuous.

Only the standard library `math` is used inside the transforms so the code
maps cleanly to C. numpy/pytest are used only by the test suite and the CSV
generator, never by the core math.
"""

# `math` gives us scalar sqrt/sin/cos/hypot that correspond 1:1 to the C
# <math.h> functions (sqrtf/sinf/cosf/hypotf) the STM32 port will call.
import math

# Precompute the two irrational constants the transforms need so the C port
# can hardcode them as `static const float` and so we never pay repeated
# sqrt() calls at runtime.
SQRT3 = math.sqrt(3.0)          # sqrt(3) ~= 1.7320508...
SQRT3_OVER_2 = SQRT3 / 2.0      # sqrt(3)/2 ~= 0.8660254..., appears in Clarke/Park
INV_SQRT3 = 1.0 / SQRT3         # 1/sqrt(3) ~= 0.5773503..., the SVPWM linear limit factor


def clarke(a, b, c):
    """Amplitude-invariant Clarke transform: (a, b, c) -> (alpha, beta).

    Uses the 2/3 scaling so that a balanced three-phase set of peak
    amplitude A maps to a stationary vector (alpha, beta) of magnitude A.

    Definitions (exactly as specified):
        alpha = (2/3) * (a - 0.5*b - 0.5*c)
        beta  = (2/3) * (sqrt(3)/2) * (b - c)

    Note the transform annihilates the common (zero-sequence) mode: adding
    the same constant to a, b, c leaves (alpha, beta) unchanged. The SVPWM
    reconstruction test relies on this property.
    """
    # alpha is the projection onto the phase-A axis; the 2/3 keeps amplitude.
    alpha = (2.0 / 3.0) * (a - 0.5 * b - 0.5 * c)
    # beta is the quadrature (90-degree) component built from (b - c).
    beta = (2.0 / 3.0) * (SQRT3_OVER_2 * (b - c))
    # Return as a plain tuple; the C port will use output pointers instead.
    return alpha, beta


def inverse_clarke(alpha, beta):
    """Inverse Clarke transform: (alpha, beta) -> (a, b, c).

    This is the right inverse of `clarke` for *zero-sequence-free* signals,
    i.e. it always produces a triple with a + b + c == 0. Round-tripping
    clarke(inverse_clarke(alpha, beta)) returns (alpha, beta) exactly (to
    float precision).

    Definitions:
        a =  alpha
        b = -0.5*alpha + (sqrt(3)/2)*beta
        c = -0.5*alpha - (sqrt(3)/2)*beta
    """
    # Phase A lies directly on the alpha axis, so it copies alpha.
    a = alpha
    # Phases B and C are alpha rotated by -/+120 degrees, expressed via beta.
    b = -0.5 * alpha + SQRT3_OVER_2 * beta
    c = -0.5 * alpha - SQRT3_OVER_2 * beta
    return a, b, c


def park(alpha, beta, theta):
    """Park transform: rotate the stationary frame into the rotor frame.

    (alpha, beta) is in the stationary (stator) frame; theta is the rotor
    electrical angle in radians. Output (d, q) is in the rotating frame.

    Sign convention (PINNED by the tests -- do not change silently):
        d = alpha*cos(theta) + beta*sin(theta)
        q = -alpha*sin(theta) + beta*cos(theta)

    Consequences that the tests lock in:
        * theta = 0      -> (d, q) == (alpha, beta)   (identity)
        * theta = pi/2   -> maps (1, 0) to (0, -1)
    """
    # Compute the sine/cosine of the rotor angle once and reuse (matches C).
    c = math.cos(theta)
    s = math.sin(theta)
    # Standard rotation-by-(-theta) applied to the stationary vector.
    d = alpha * c + beta * s
    q = -alpha * s + beta * c
    return d, q


def inverse_park(d, q, theta):
    """Inverse Park transform: rotor frame -> stationary frame.

    Exact inverse of `park`: rotate (d, q) back out by +theta.
        alpha = d*cos(theta) - q*sin(theta)
        beta  = d*sin(theta) + q*cos(theta)
    """
    # Same trig as forward Park; the sign pattern is transposed (inverse rot).
    c = math.cos(theta)
    s = math.sin(theta)
    alpha = d * c - q * s
    beta = d * s + q * c
    return alpha, beta


def svpwm(v_alpha, v_beta, v_dc):
    """Space-vector PWM via min-max common-mode injection.

    Inputs are the commanded stationary-frame voltage (v_alpha, v_beta) and
    the DC-bus voltage v_dc. Outputs are three duty cycles (da, db, dc) in
    [0, 1] plus a boolean `clamped` flag.

    Method (min-max / midpoint injection -- equivalent to classic SVPWM in
    the linear region, but branchless and continuous across all six sector
    boundaries):
        1. Optionally clamp the command magnitude to the linear limit.
        2. Map (v_alpha, v_beta) to the three phase voltages via inverse
           Clarke.
        3. Subtract the midpoint of (max, min) phase voltage from all three.
           This is the zero-sequence injection that centres the waveform and
           extends the linear range to |V| <= v_dc/sqrt(3).
        4. Convert each centred phase voltage to a duty by scaling by 1/v_dc
           and offsetting by 0.5 (so 0 V -> 50% duty).

    Linear region: |V| <= v_dc/sqrt(3). Commands beyond that are clamped in
    magnitude to the limit while preserving the vector angle, and `clamped`
    is returned True so the caller knows saturation occurred.

    Returns:
        (da, db, dc, clamped)
    """
    # Guard the degenerate bus so we never divide by zero and emit NaN.
    # With no bus voltage every phase collapses to the 50% midpoint.
    if v_dc <= 0.0:
        return 0.5, 0.5, 0.5, False

    # The maximum representable line-to-line vector magnitude in the linear
    # region of SVPWM is v_dc/sqrt(3); this is the circle inscribed in the
    # hexagon of reachable space vectors.
    limit = v_dc * INV_SQRT3

    # Magnitude of the commanded vector (hypot is the stable sqrt(x^2+y^2)).
    mag = math.hypot(v_alpha, v_beta)

    # Track whether we had to saturate the request.
    clamped = False
    # Overmodulation handling: if the command exceeds the inscribed circle,
    # shrink it back to the circle while keeping its direction (angle).
    if mag > limit:
        # Scale factor < 1 that lands the vector exactly on the limit circle.
        scale = limit / mag
        v_alpha = v_alpha * scale
        v_beta = v_beta * scale
        clamped = True

    # Reconstruct the (zero-sequence-free) phase voltages from the vector.
    va, vb, vc = inverse_clarke(v_alpha, v_beta)

    # Min-max injection: find the extreme phases and subtract their midpoint.
    v_max = max(va, vb, vc)
    v_min = min(va, vb, vc)
    v_offset = 0.5 * (v_max + v_min)  # common-mode term to inject

    # Apply the common-mode shift equally to all three phases.
    va -= v_offset
    vb -= v_offset
    vc -= v_offset

    # Convert centred phase voltages to duty cycles. Dividing by v_dc scales
    # a +/-(v_dc/2) swing into +/-0.5, and the +0.5 offset centres duties at
    # 50%. With min-max injection at the linear limit these stay within [0,1].
    inv_vdc = 1.0 / v_dc            # single reciprocal, reused (FPU-friendly)
    da = 0.5 + va * inv_vdc
    db = 0.5 + vb * inv_vdc
    dc = 0.5 + vc * inv_vdc

    return da, db, dc, clamped


def _linspace(start, stop, n):
    """Tiny inclusive linspace so the CSV generator needs no numpy.

    Returns n evenly spaced values from start to stop inclusive. Kept private
    (leading underscore) because it is only a helper for `generate_vectors`.
    """
    # Degenerate case: a single sample is just the start point.
    if n == 1:
        return [start]
    # Step between successive samples (n-1 intervals across the range).
    step = (stop - start) / (n - 1)
    # Build the list with a comprehension; index i scales the step.
    return [start + step * i for i in range(n)]


def generate_vectors(path="vectors.csv", v_dc=24.0):
    """Write ~50 reference rows to `path` for cross-checking the C port.

    Each row sweeps a commanded vector (given by an angle `theta` and a
    magnitude as a fraction of the linear limit) and records both the SVPWM
    duties and the Park-transformed (d, q) at that same angle. The C firmware
    can replay the identical (v_alpha, v_beta, v_dc, theta) inputs and diff
    its computed (da, db, dc, d, q) against these columns.

    Columns: v_alpha, v_beta, v_dc, da, db, dc, theta, d, q
    """
    # csv is stdlib; used only here (not in the core math) so the C port is
    # unaffected.
    import csv

    # The inscribed-circle voltage limit for the chosen bus.
    limit = v_dc * INV_SQRT3

    # Ten angles across a full electrical revolution guarantee we visit all
    # six 60-degree SVPWM sectors, including points inside each sector.
    angles = _linspace(0.0, 2.0 * math.pi, 10)

    # Magnitudes as fractions of the linear limit: a few inside the linear
    # region and one (1.2) deliberately in overmodulation to exercise clamping.
    mag_fracs = [0.25, 0.5, 0.75, 0.95, 1.2]

    # Accumulate rows so we can report how many were written.
    rows = []
    # Outer loop over magnitude, inner over angle -> 5 * 10 = 50 rows total.
    for frac in mag_fracs:
        for theta in angles:
            # Commanded magnitude in volts for this row.
            v = frac * limit
            # Decompose the command into stationary-frame components.
            v_alpha = v * math.cos(theta)
            v_beta = v * math.sin(theta)
            # SVPWM duties for this command (clamp flag ignored in the CSV).
            da, db, dc, _clamped = svpwm(v_alpha, v_beta, v_dc)
            # Park transform of the same vector at the same angle.
            d, q = park(v_alpha, v_beta, theta)
            # Record the row in the documented column order.
            rows.append([v_alpha, v_beta, v_dc, da, db, dc, theta, d, q])

    # Write everything out with a header line the C harness can parse.
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["v_alpha", "v_beta", "v_dc", "da", "db", "dc", "theta", "d", "q"]
        )
        writer.writerows(rows)

    # Hand the count back for logging / test assertions.
    return len(rows)


# Running the module directly regenerates the reference CSV -- handy when the
# math changes and the golden vectors must be refreshed.
if __name__ == "__main__":
    n = generate_vectors()
    print(f"wrote {n} rows to vectors.csv")
