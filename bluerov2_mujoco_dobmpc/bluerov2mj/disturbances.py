"""External disturbance generators (paper Sec. 5).

All wrenches are expressed in the inertial (NED) frame and applied at the
CG - the MuJoCo equivalent of the ROS ``ApplyBodyWrench`` service used in
the paper.  Returns w = [Fx, Fy, Fz, Mx, My, Mz].
"""
import numpy as np


class PeriodicWave:
    """Sinusoids with random amplitude per axis (paper Sec. 5.1).

    Forces 10-16 N in x/y/z, moment 1-2 Nm about z (defaults).
    """

    def __init__(self, f_amp=(10.0, 16.0), m_amp=(1.0, 2.0),
                 period=(2.5, 4.0), seed=1):
        rng = np.random.default_rng(seed)
        self.A = np.zeros(6)
        self.A[:3] = rng.uniform(*f_amp, 3)
        self.A[5] = rng.uniform(*m_amp)
        self.omega = 2 * np.pi / rng.uniform(*period, 6)
        self.phase = rng.uniform(0, 2 * np.pi, 6)

    def __call__(self, t):
        return self.A * np.sin(self.omega * t + self.phase)


class ConstantCurrent:
    """Step current: force/moment switched on at t_on (paper Sec. 5.1)."""

    def __init__(self, force=(10.0, 10.0, 10.0), moment_z=5.0, t_on=10.0):
        self.w = np.array([*force, 0.0, 0.0, moment_z])
        self.t_on = t_on

    def __call__(self, t):
        return self.w if t >= self.t_on else np.zeros(6)


class Superposition:
    def __init__(self, *components):
        self.components = components

    def __call__(self, t):
        return sum(c(t) for c in self.components)


def make(name, seed=1):
    """Factory for the three scenarios in the paper."""
    if name == "periodic":
        return PeriodicWave(seed=seed)
    if name == "constant":
        return ConstantCurrent()
    if name == "mixed":
        return Superposition(
            PeriodicWave(f_amp=(3.0, 6.0), m_amp=(1.0, 2.0), seed=seed),
            ConstantCurrent(force=(10.0, 10.0, 10.0), moment_z=3.0, t_on=4.0))
    if name == "none":
        return lambda t: np.zeros(6)
    raise ValueError(name)
