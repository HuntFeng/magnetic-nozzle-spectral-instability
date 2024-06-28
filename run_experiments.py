import numpy as np
import matplotlib.pyplot as plt
from nozzle import Spectral, Params, Nozzle, Boundary
from scipy.integrate import simpson
from scipy.special import legendre
from scipy.optimize import root


def galerkin_method(Mm: float, constant_v: bool, boundary: Boundary):
    N = 30  # number of basis
    M = 101  # number of points
    params = Params(Mm, constant_v, False, boundary)

    spectral = Spectral(M, "symmetric", "CH")
    x = spectral.x
    D1 = spectral.D1
    D2 = spectral.D2

    if boundary.value == "fixed_fixed":
        u = lambda x, n: (legendre(n) - legendre(n + 2))(x)
    else:
        u = lambda x, n: (
            legendre(n)
            + (2 * n + 3) / (n + 2) ** 2 * legendre(n + 1)
            - (n + 1) ** 2 / (n + 2) ** 2 * legendre(n + 2)
        )(x)
    nozzle = Nozzle(params, x, lambda x, n: u(x, n))
    v0 = nozzle.v0

    A2 = np.zeros((N, N), dtype=complex)
    A1 = np.zeros((N, N), dtype=complex)
    A0 = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            A2[i, j] = simpson(u(x, i) * u(x, j), x=x)
            A1[i, j] = 2j * simpson(
                u(x, i) * v0 * (D1 @ u(x, j)) + u(x, i) * (D1 @ v0) * u(x, j), x=x
            )
            A0[i, j] = simpson(
                u(x, i) * (1 - v0**2) * (D2 @ u(x, j))
                - u(x, i) * (3 * v0 + 1 / v0) * (D1 @ v0) * (D1 @ u(x, j))
                - u(x, i) * (1 - 1 / v0**2) * (D1 @ v0) ** 2 * u(x, j)
                - u(x, i) * (v0 + 1 / v0) * (D2 @ v0) * u(x, j),
                x=x,
            )

    C, nozzle.omega = nozzle.solve(A0, A1, A2)
    nozzle.V = np.zeros((x.size, C.shape[1]), dtype=complex)
    for i in range(C.shape[1]):
        for n in range(N):
            nozzle.V[:, i] += C[n, i] * u(x, n)
    nozzle.sort_solutions(real_range=[-0.1, 50])
    nozzle.save_data("galerkin")


def collocation_method(Mm: float, constant_v: bool, boundary: Boundary):
    N = 101
    params = Params(Mm, constant_v, False, boundary)
    spectral = Spectral(N, "symmetric", "CH")
    nozzle = Nozzle(params, spectral.x)
    v0 = nozzle.v0
    x = spectral.x
    D1 = spectral.D1
    D2 = spectral.D2

    if boundary.value == "fixed_fixed":
        A11 = np.zeros_like(D1)
        A12 = np.eye(*D1.shape)
        A21 = (
            -np.diag(1 - v0**2) @ D2
            + np.diag((3 * v0 + 1 / v0) * (D1 @ v0)) @ D1
            + np.diag((1 - 1 / v0**2) * (D1 @ v0) ** 2)
            + np.diag((v0 + 1 / v0) * (D2 @ v0))
        )  # - eta*D4
        A22 = -2j * (np.diag(v0) @ D1 + np.diag(D1 @ v0))  # + eta*1j*np.diag(v0)@D4
        A = np.block(
            [[A11[1:-1, 1:-1], A12[1:-1, 1:-1]], [A21[1:-1, 1:-1], A22[1:-1, 1:-1]]]
        )
        nozzle.V, nozzle.omega = nozzle.solve(A)
        nozzle.V = np.pad(nozzle.V, ((1, 1), (0, 0)))  # pad two ends by 0
    else:
        D1v = D1
        D2v = D2
        D1v[1, :] = D1[1, :] - D1[1, 0] / D1[0, 0] * D1[0, :]
        D2v[1, :] = D2[1, :] - D2[1, 0] / D1[0, 0] * D1[0, :]

        A11 = np.zeros_like(D1)
        A12 = np.eye(*D1.shape)
        A21 = (
            -np.diag(1 - v0**2) @ D2v
            + np.diag((3 * v0 + 1 / v0) * (D1 @ v0)) @ D1v
            + np.diag((1 - 1 / v0**2) * (D1 @ v0) ** 2)
            + np.diag((v0 + 1 / v0) * (D2 @ v0))
        )
        A22 = -2j * (np.diag(v0) @ D1v + np.diag(D1 @ v0))

        A = np.block([[A11[:-1, :-1], A12[:-1, :-1]], [A21[:-1, :-1], A22[:-1, :-1]]])
        V, nozzle.omega = nozzle.solve(A)
        # modify the last row
        V = np.pad(V, ((1, 0), (0, 0)))
        for j in range(V.shape[1]):
            V[0, j] = -(D1[0, 1:] @ V[1:, j]) / D1[0, 0]
        nozzle.V = V
    nozzle.sort_solutions(real_range=[-0.1, 50])
    nozzle.save_data("collocation")


def shooting_method(accelerating: bool):
    params = Params(1, False, accelerating, Boundary.FIXED_FIXED)
    spectral = Spectral(501, "symmetric", "FD")
    nozzle = Nozzle(params, spectral.x)
    D1 = spectral.D1
    D2 = spectral.D2
    x = spectral.x
    v0 = lambda z: np.interp(z, x, nozzle.v0)
    v0_p = lambda z: np.interp(z, x, D1 @ nozzle.v0)
    v0_pp = lambda z: np.interp(z, x, D2 @ nozzle.v0)

    def f(z: float, y: float, params: float):
        """
        ind: index of current position in z array
        y: [v,u] at current position
        """
        omega = params["omega"]
        c0 = params["c0"]
        c1 = params["c1"]
        c2 = params["c2"]

        v, u = y[0], y[1]
        if np.abs(z) < 0.01:
            return np.array([c1, 2 * c2])
        else:
            return np.array(
                [
                    u,
                    -(
                        omega**2 * v
                        + 2j * omega * (v0(z) * u + v0_p(z) * v)
                        - (3 * v0(z) + 1 / v0(z)) * v0_p(z) * u
                        - (1 - 1 / v0(z) ** 2) * v0_p(z) ** 2 * v
                        - (v0(z) + 1 / v0(z)) * v0_pp(z) * v
                    )
                    / (1 - v0(z) ** 2),
                ]
            )

    def RK4(
        f: callable,
        z_arr: np.array,
        y0: np.array,
        params: dict,
        return_array: bool = True,
    ):
        """RK4"""
        dz = z_arr[1] - z_arr[0]
        if return_array:
            y = np.zeros((2, z_arr.size), dtype=complex)
            y[:, 0] = y0
            for n, z in enumerate(z_arr[:-1]):
                yn = y[:, n]
                k1 = f(z, yn, params)
                k2 = f(z + dz / 2, yn + dz * k1 / 2, params)
                k3 = f(z + dz / 2, yn + dz * k2 / 2, params)
                k4 = f(z + dz, yn + dz * k3, params)
                y[:, n + 1] = yn + dz * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            y = y0
            for _, z in enumerate(z_arr[:-1]):
                k1 = f(z, y, params)
                k2 = f(z + dz / 2, y + dz * k1 / 2, params)
                k3 = f(z + dz / 2, y + dz * k2 / 2, params)
                k4 = f(z + dz, y + dz * k3, params)
                y += dz * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return y

    def ivp(z_arr: np.array, omega: complex):
        a = (2j * omega - 4 * v0_p(0)) / (-2 * v0_p(0))
        b = (omega**2 + 2j * omega * v0_p(0) - 2 * v0_pp(0)) / (-2 * v0_p(0))
        c0 = 1
        c1 = -c0 * b / a
        c2 = -c1 * b / (2 * (1 + a))

        params = {"omega": omega, "c0": c0, "c1": c1, "c2": c2}
        y0 = np.array([c0, c1])
        return RK4(f, z_arr, y0, params, return_array=True)

    def shooting(omega: np.array):
        """omega is a size-2 array [real, imag]"""
        omega = complex(*omega)
        a = (2j * omega - 4 * v0_p(0)) / (-2 * v0_p(0))
        b = (omega**2 + 2j * omega * v0_p(0) - 2 * v0_pp(0)) / (-2 * v0_p(0))
        c0 = 1
        c1 = -c0 * b / a
        c2 = -c1 * b / (2 * (1 + a))

        params = {"omega": omega, "c0": c0, "c1": c1, "c2": c2}

        z_arr = x[x <= 0][::-1]
        y0 = np.array([c0, c1])
        y = RK4(f, z_arr, y0, params, return_array=False)
        # we want y[0].real = y[0].imag = 0
        return [y[0].real, y[0].imag]

    # calculate eigenvalues
    omegas = []
    for real_part in tqdm(range(0, 22, 2)):
        result = root(shooting, x0=[real_part, 0])
        if not result.success:
            continue
        omega = complex(*result.x)
        if len(omegas) > 0 and np.isclose(omega, omegas[-1]):
            continue
        omegas.append(omega)
    nozzle.omega = np.array(omegas)

    # build eigenfunctions
    nozzle.V = np.zeros((x.size, len(omegas)), dtype=complex)
    z_left = x[x <= 0][::-1]
    z_right = x[x > 0]
    for i, omega in tqdm(enumerate(omegas)):
        y_left = ivp(z_left, omega)
        y_right = ivp(z_right, omega)
        nozzle.V[:, i] = np.concatenate([y_left[0][::-1], y_right[0]])
    nozzle.save_data("shooting")


if __name__ == "__main__":
    from tqdm import tqdm
    from tqdm.contrib.itertools import product

    Mm_range = [0.5, 1, 1.5]
    is_constant_v = [True, False]
    is_accelerating = [True, False]
    params = product(Mm_range, is_constant_v, is_accelerating, Boundary)
    for param in tqdm(params):
        Mm, constant_v, accelerating, boundary = param
        if Mm == 1:
            shooting_method(accelerating)
        else:
            galerkin_method(Mm, constant_v, boundary)
            collocation_method(Mm, constant_v, boundary)
