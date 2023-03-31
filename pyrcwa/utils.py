import matplotlib.pyplot as plt
import numpy as np
from numpy import pi


def dispersion_relation(kx: float, ky, k0, epsilon):
    return np.conj(np.lib.scimath.sqrt(k0 ** 2 * epsilon - kx ** 2 - ky ** 2))


def deg2rad(deg: float):
    return deg / 180 * pi


def rad2deg(rad):
    return rad / pi * 180


def truncate_harmonics_order(M, N, gamma):
    x = np.arange(-M, M + 1, 1)
    y = np.arange(-N, N + 1, 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = (abs(X / M)) ** (2 * gamma) + (abs(Y / N)) ** (2 * gamma)
    Z = np.zeros_like(R)
    Z[R <= 1] = 1
    x_ticks = ["{}".format(x - M) for x in range(len(x))]
    y_ticks = ["{}".format(- y + N) for x in range(len(y))]

    fig, ax = plt.subplots()
    ax.imshow(Z)
    ax.set_xticks(np.arange(len(x_ticks)), labels=x_ticks)
    ax.set_yticks(np.arange(len(y_ticks)), labels=y_ticks)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(len(x_ticks) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(y_ticks) + 1) - .5, minor=True)
    ax.grid(which='minor', color='black', linestyle = '-', linewidth = 1)
    ax.tick_params(which='minor', bottom=False, left=False)
    plt.tight_layout()
    plt.show()
    return Z
