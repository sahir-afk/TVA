"""pytest contract for `foc_transforms`.

These tests *define* the numeric behaviour the future STM32 C port must
reproduce. Default tolerance is 1e-9 unless a specific test states otherwise
(SVPWM continuity and reconstruction have their own looser/explicit bounds).

Only numpy (for random test vectors) and pytest are used here.
"""

import math

import numpy as np
import pytest

# Import the reference implementation under test.
from foc_transforms import (
    clarke,
    inverse_clarke,
    park,
    inverse_park,
    svpwm,
    INV_SQRT3,
)

# A single seeded RNG makes the random-vector tests fully reproducible, which
# matters because the C port will be diffed against these exact numbers.
RNG = np.random.default_rng(20260711)

# Default absolute tolerance for the "should be exact math" comparisons.
TOL = 1e-9


# --------------------------------------------------------------------------
# Clarke transform
# --------------------------------------------------------------------------

def test_clarke_balanced_set():
    """The canonical balanced input (1, -0.5, -0.5) must map to (1, 0)."""
    # This input is a unit vector on the phase-A axis; amplitude-invariant
    # Clarke should return exactly alpha=1, beta=0.
    alpha, beta = clarke(1.0, -0.5, -0.5)
    assert alpha == pytest.approx(1.0, abs=TOL)
    assert beta == pytest.approx(0.0, abs=TOL)


def test_clarke_inverse_clarke_roundtrip_balanced():
    """clarke -> inverse_clarke recovers 100 random *balanced* triples."""
    # Balanced means a + b + c == 0; only such triples are recoverable, since
    # Clarke discards the zero-sequence component.
    for _ in range(100):
        # Draw two free phases; fix the third so the sum is zero.
        a = RNG.uniform(-10.0, 10.0)
        b = RNG.uniform(-10.0, 10.0)
        c = -(a + b)
        # Forward then inverse should return the original phases.
        alpha, beta = clarke(a, b, c)
        a2, b2, c2 = inverse_clarke(alpha, beta)
        assert a2 == pytest.approx(a, abs=TOL)
        assert b2 == pytest.approx(b, abs=TOL)
        assert c2 == pytest.approx(c, abs=TOL)


# --------------------------------------------------------------------------
# Park transform
# --------------------------------------------------------------------------

def test_park_identity_at_zero():
    """At theta = 0 the Park transform is the identity map."""
    # Pick an arbitrary vector; rotating by zero must leave it unchanged.
    alpha, beta = 0.7, -0.3
    d, q = park(alpha, beta, 0.0)
    assert d == pytest.approx(alpha, abs=TOL)
    assert q == pytest.approx(beta, abs=TOL)


def test_park_sign_convention_at_half_pi():
    """theta = pi/2 must map (1, 0) -> (0, -1). This pins the sign."""
    # If this fails, the rotation-sign convention was changed; that is a
    # contract break, not a bug to "fix" in the transform.
    d, q = park(1.0, 0.0, math.pi / 2.0)
    assert d == pytest.approx(0.0, abs=TOL)
    assert q == pytest.approx(-1.0, abs=TOL)


def test_park_inverse_park_roundtrip():
    """park -> inverse... actually inverse_park -> park recovers (d, q)."""
    # Random rotor-frame vectors and angles; going out to the stationary
    # frame and back must be lossless.
    for _ in range(100):
        d = RNG.uniform(-5.0, 5.0)
        q = RNG.uniform(-5.0, 5.0)
        theta = RNG.uniform(-4.0 * math.pi, 4.0 * math.pi)
        alpha, beta = inverse_park(d, q, theta)
        d2, q2 = park(alpha, beta, theta)
        assert d2 == pytest.approx(d, abs=TOL)
        assert q2 == pytest.approx(q, abs=TOL)


def test_full_chain_recovers_dq():
    """inverse_park -> inverse_clarke -> clarke -> park recovers (d, q)."""
    # This exercises the whole stationary/rotating/three-phase round trip that
    # a real controller performs each PWM period.
    for _ in range(100):
        d = RNG.uniform(-5.0, 5.0)
        q = RNG.uniform(-5.0, 5.0)
        theta = RNG.uniform(-4.0 * math.pi, 4.0 * math.pi)
        # Rotor frame -> stationary vector.
        alpha, beta = inverse_park(d, q, theta)
        # Stationary vector -> three phase voltages.
        a, b, c = inverse_clarke(alpha, beta)
        # Back to stationary vector (should equal alpha, beta).
        alpha2, beta2 = clarke(a, b, c)
        # Back to rotor frame (should equal the original d, q).
        d2, q2 = park(alpha2, beta2, theta)
        assert d2 == pytest.approx(d, abs=TOL)
        assert q2 == pytest.approx(q, abs=TOL)


# --------------------------------------------------------------------------
# SVPWM
# --------------------------------------------------------------------------

def test_svpwm_zero_command_is_half():
    """A zero voltage command must give all three duties exactly 0.5."""
    da, db, dc, clamped = svpwm(0.0, 0.0, 24.0)
    assert da == 0.5
    assert db == 0.5
    assert dc == 0.5
    # Zero command is inside the linear region, so no clamping.
    assert clamped is False


def test_svpwm_duties_in_range_over_sweep():
    """Duties stay within [0, 1] for a full-angle sweep at 90% of the limit."""
    v_dc = 24.0
    # 90% of the linear limit keeps us safely inside the linear region.
    v = 0.9 * v_dc * INV_SQRT3
    # Sweep the electrical angle across a full revolution.
    for theta in np.linspace(0.0, 2.0 * math.pi, 361):
        v_alpha = v * math.cos(theta)
        v_beta = v * math.sin(theta)
        da, db, dc, clamped = svpwm(v_alpha, v_beta, v_dc)
        # Every duty must be a valid PWM compare value.
        for duty in (da, db, dc):
            assert 0.0 <= duty <= 1.0
        # At 90% we should never trip the overmodulation clamp.
        assert clamped is False


def test_svpwm_continuity_at_sector_boundaries():
    """Duties are continuous across all six 60-degree sector boundaries."""
    v_dc = 24.0
    v = 0.9 * v_dc * INV_SQRT3
    # A tiny angular step to probe just below and just above each boundary.
    eps = 1e-7
    # The six boundaries at 0, 60, 120, 180, 240, 300 degrees.
    for k in range(6):
        boundary = k * (math.pi / 3.0)
        # Evaluate duties immediately below and above the boundary.
        below = svpwm(v * math.cos(boundary - eps),
                      v * math.sin(boundary - eps), v_dc)[:3]
        above = svpwm(v * math.cos(boundary + eps),
                      v * math.sin(boundary + eps), v_dc)[:3]
        # Min-max injection is continuous, so the jump must be negligible.
        for du_lo, du_hi in zip(below, above):
            assert abs(du_hi - du_lo) < 1e-6


def test_svpwm_reconstruction_in_linear_region():
    """Recover the commanded (v_alpha, v_beta) from the emitted duties."""
    v_dc = 24.0
    # Stay in the linear region so no clamping distorts the command.
    v = 0.8 * v_dc * INV_SQRT3
    for theta in np.linspace(0.0, 2.0 * math.pi, 37):
        v_alpha = v * math.cos(theta)
        v_beta = v * math.sin(theta)
        da, db, dc, clamped = svpwm(v_alpha, v_beta, v_dc)
        # Reverse the duty->voltage mapping: remove the common mode (mean)
        # and rescale by v_dc to get zero-sequence-free phase voltages.
        mean_d = (da + db + dc) / 3.0
        v_a = (da - mean_d) * v_dc
        v_b = (db - mean_d) * v_dc
        v_c = (dc - mean_d) * v_dc
        # Clarke of those phases must return the original command, because
        # Clarke ignores the injected common mode.
        alpha_r, beta_r = clarke(v_a, v_b, v_c)
        assert alpha_r == pytest.approx(v_alpha, abs=TOL)
        assert beta_r == pytest.approx(v_beta, abs=TOL)


def test_svpwm_overmodulation_clamps_magnitude_and_keeps_angle():
    """Commanding 1.5x the limit clamps |V| to the limit, angle preserved."""
    v_dc = 24.0
    limit = v_dc * INV_SQRT3
    # Command 1.5x the linear limit at several angles.
    for theta in np.linspace(0.0, 2.0 * math.pi, 25):
        v = 1.5 * limit
        v_alpha = v * math.cos(theta)
        v_beta = v * math.sin(theta)
        da, db, dc, clamped = svpwm(v_alpha, v_beta, v_dc)
        # The clamp flag must be set for an overmodulated command.
        assert clamped is True
        # Reconstruct the actually-applied vector from the duties.
        mean_d = (da + db + dc) / 3.0
        alpha_r, beta_r = clarke((da - mean_d) * v_dc,
                                 (db - mean_d) * v_dc,
                                 (dc - mean_d) * v_dc)
        # Magnitude must equal the limit (within tolerance)...
        assert math.hypot(alpha_r, beta_r) == pytest.approx(limit, abs=1e-9)
        # ...and the angle must match the commanded angle.
        applied_angle = math.atan2(beta_r, alpha_r)
        commanded_angle = math.atan2(v_beta, v_alpha)
        # Compare angles modulo 2*pi via their sine/cosine difference to avoid
        # wrap-around artefacts at +/-pi.
        assert math.sin(applied_angle - commanded_angle) == pytest.approx(0.0, abs=1e-9)
        assert math.cos(applied_angle - commanded_angle) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------

def test_no_nan_on_zero_and_degenerate_inputs():
    """Zero / degenerate inputs must never produce NaN anywhere."""
    # Clarke and Park of all-zeros.
    assert not any(math.isnan(x) for x in clarke(0.0, 0.0, 0.0))
    assert not any(math.isnan(x) for x in inverse_clarke(0.0, 0.0))
    assert not any(math.isnan(x) for x in park(0.0, 0.0, 0.0))
    assert not any(math.isnan(x) for x in inverse_park(0.0, 0.0, 0.0))
    # SVPWM with a zero command and a normal bus.
    assert not any(math.isnan(x) for x in svpwm(0.0, 0.0, 24.0)[:3])
    # SVPWM with a zero bus voltage (guarded division) must stay finite.
    assert not any(math.isnan(x) for x in svpwm(1.0, 1.0, 0.0)[:3])
    # SVPWM with a negative/degenerate bus voltage.
    assert not any(math.isnan(x) for x in svpwm(1.0, -1.0, -5.0)[:3])
