"""Fossen 6-DOF model of the BlueROV2 (paper Sec. 2), NumPy implementation.

State:  eta = [x y z phi theta psi] (NED / ZYX-Euler),  nu = [u v w p q r] (FRD)
Model:  M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau + w          (Eq. 7)
        eta_dot = J(eta) nu                                        (Eq. 5)
"""
import numpy as np

from . import params as P


# --------------------------------------------------------------- mass matrix
def mass_matrix():
    """M = M_RB + M_A  (Eq. 8, 10, 12), with the m*zg coupling terms."""
    m, zg = P.MASS, P.ZG
    M = np.diag([m + P.ADDED_MASS[0], m + P.ADDED_MASS[1], m + P.ADDED_MASS[2],
                 P.IX + P.ADDED_MASS[3], P.IY + P.ADDED_MASS[4],
                 P.IZ + P.ADDED_MASS[5]])
    M[0, 4] = m * zg
    M[4, 0] = m * zg
    M[1, 3] = -m * zg
    M[3, 1] = -m * zg
    return M


M_TOTAL = mass_matrix()
M_INV = np.linalg.inv(M_TOTAL)
M_RB_DIAG = np.array([P.MASS, P.MASS, P.MASS, P.IX, P.IY, P.IZ])


# ------------------------------------------------------------------ rotation
def rot_ib(phi, theta, psi):
    """R^i_b (Eq. 2): body (FRD) -> inertial (NED), R = Rz(psi)Ry(theta)Rx(phi)."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        [cps * cth, -sps * cph + cps * sth * sph,  sps * sph + cps * cph * sth],
        [sps * cth,  cps * cph + sph * sth * sps, -cps * sph + sth * sps * cph],
        [-sth,       cth * sph,                    cth * cph],
    ])


def t_euler(phi, theta):
    """T(Theta) (Eq. 4): body rates -> Euler angle rates."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, tth = np.cos(theta), np.tan(theta)
    return np.array([
        [1.0, sph * tth, cph * tth],
        [0.0, cph, -sph],
        [0.0, sph / cth, cph / cth],
    ])


def jacobian_eta(eta):
    """J(eta) (Eq. 6)."""
    J = np.zeros((6, 6))
    J[:3, :3] = rot_ib(eta[3], eta[4], eta[5])
    J[3:, 3:] = t_euler(eta[3], eta[4])
    return J


# ----------------------------------------------------------- force terms
def coriolis_rb(nu):
    """C_RB(nu) nu  (Eq. 11), with the m*zg terms dropped (|zg| small)."""
    m = P.MASS
    u, v, w, p, q, r = nu
    Ix, Iy, Iz = P.IX, P.IY, P.IZ
    return np.array([
        m * (q * w - r * v),
        m * (r * u - p * w),
        m * (p * v - q * u),
        (Iz - Iy) * q * r,
        (Ix - Iz) * p * r,
        (Iy - Ix) * p * q,
    ])


def coriolis_added(nu):
    """C_A(nu) nu (Eq. 13) for diagonal M_A."""
    a = P.ADDED_MASS
    u, v, w, p, q, r = nu
    return np.array([
        a[2] * w * q - a[1] * v * r,
        -a[2] * w * p + a[0] * u * r,
        a[1] * v * p - a[0] * u * q,
        a[2] * w * v - a[1] * v * w + a[5] * r * q - a[4] * q * r,
        -a[2] * w * u + a[0] * u * w - a[5] * r * p + a[3] * p * r,
        a[1] * v * u - a[0] * u * v + a[4] * q * p - a[3] * p * q,
    ])


def damping(nu):
    """(D_L + D_NL(nu)) nu  (Eq. 14-16).  Returns D(nu)nu with D positive
    semi-definite, i.e. the dissipative force on the LHS of Eq. 7."""
    return -(P.DL * nu + P.DNL * np.abs(nu) * nu)


def restoring(eta):
    """g(eta)  (Eq. 17)."""
    W, B, zg = P.WEIGHT, P.BUOYANCY, P.ZG
    phi, theta = eta[3], eta[4]
    sph, cph = np.sin(phi), np.cos(phi)
    sth, cth = np.sin(theta), np.cos(theta)
    return np.array([
        (W - B) * sth,
        -(W - B) * cth * sph,
        -(W - B) * cth * cph,
        zg * W * cth * sph,
        zg * W * sth,
        0.0,
    ])


def nu_dot(eta, nu, tau, w):
    """nu_dot = M^-1 (tau + w - C nu - D nu - g)   (Eq. 22)."""
    rhs = tau + w - coriolis_rb(nu) - coriolis_added(nu) - damping(nu) \
        - restoring(eta)
    return M_INV @ rhs


def f_state(x, tau, w):
    """x = [eta; nu] (12,) -> x_dot  (Eq. 22)."""
    eta, nu = x[:6], x[6:]
    return np.concatenate([jacobian_eta(eta) @ nu, nu_dot(eta, nu, tau, w)])


def f_ekf(xa, tau):
    """Augmented EAOB state xa = [eta; nu; w] (18,) -> xa_dot  (Eq. 26)."""
    eta, nu, w = xa[:6], xa[6:12], xa[12:]
    return np.concatenate([jacobian_eta(eta) @ nu,
                           nu_dot(eta, nu, tau, w),
                           np.zeros(6)])


def rk4(f, x, dt, *args):
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# -------------------------------------------------------- quaternion helpers
def quat_to_euler(q):
    """MuJoCo quaternion (w,x,y,z) -> ZYX Euler (phi,theta,psi)."""
    w, x, y, z = q
    phi = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    theta = np.arcsin(s)
    psi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([phi, theta, psi])


def euler_to_quat(phi, theta, psi):
    """ZYX Euler -> MuJoCo quaternion (w,x,y,z)."""
    c1, s1 = np.cos(psi / 2), np.sin(psi / 2)
    c2, s2 = np.cos(theta / 2), np.sin(theta / 2)
    c3, s3 = np.cos(phi / 2), np.sin(phi / 2)
    return np.array([
        c1 * c2 * c3 + s1 * s2 * s3,
        c1 * c2 * s3 - s1 * s2 * c3,
        c1 * s2 * c3 + s1 * c2 * s3,
        s1 * c2 * c3 - c1 * s2 * s3,
    ])


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
