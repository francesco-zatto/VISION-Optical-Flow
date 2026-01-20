import numpy as np
import utils
from gradhorn import gradhorn
from scipy.signal import convolve2d 

A = np.array([
    [1/12, 1/6, 1/12],
    [1/6, 0, 1/6],
    [1/12, 1/6, 1/12]
])

def eta(f, Ix, Iy, q):
    f_avg = convolve2d(f, A, mode='same', boundary='symm')
    kernel = np.array([[-1, 0, 1]])
    fx = convolve2d(f, kernel / 2, mode='same', boundary='symm')
    fy = convolve2d(f, kernel.T / 2, mode='same', boundary='symm')
    fxy = convolve2d(fx, kernel.T / 2, mode='same', boundary='symm')

    grad_f = np.stack([fx, fy], axis=-1).reshape(fx.shape + (2, 1))

    return f_avg - 2 * Ix * Iy * fxy - (q @ grad_f).reshape(fx.shape)

def compute_V(Ix, Iy, delta):
    denominator = Ix**2 + Iy**2 + 2 * delta
    numerator = np.array([
        [Iy**2 + delta, -Ix * Iy],
        [-Ix * Iy, Ix**2 + delta]
    ])
    return (numerator / denominator).transpose(-2, -1, 0, 1)

def compute_q(V, Ix, Iy, Ixx, Ixy, Iyy, delta):
    denominator = (Ix**2 + Iy**2 + 2 * delta).reshape(Ix.shape + (1, 1))
    grad_I = np.array([Ix, Iy]).reshape(Ix.shape + (1, 2))
    V_0 = np.array([
        [Iyy, -Ixy],
        [-Ixy, Ixx]
    ]).transpose(-2, -1, 0, 1)
    V_1 = 2 * np.array([
        [Ixx, Ixy],
        [Ixy, Iyy]
    ]).transpose(-2, -1, 0, 1)

    V_term = V_0 + V_1 @ V
    q = (grad_I @ V_term)
    return q / denominator


def nagel(I1, I2, N, alpha, delta):
    Ix, Iy, It = gradhorn(I1, I2)
    Ixx, Ixy, _ = gradhorn(Ix, Ix)
    Iyy, _, _ = gradhorn(Iy, Iy)
    u = v = np.zeros_like(Ix)

    V = compute_V(Ix, Iy, delta)
    q = compute_q(V, Ix, Iy, Ixx, Ixy, Iyy, delta)

    for _ in range(N):
        u_eta = eta(u, Ix, Iy, q)
        v_eta = eta(v, Ix, Iy, q)
        update_term = (Ix * u_eta + Iy * v_eta + It) / (alpha + Ix**2 + Iy**2)
        u = u_eta - Ix * update_term
        v = v_eta - Iy * update_term
    return u, v

import matplotlib.pyplot as plt

IM1_PATH = 'data/taxi/taxi9.png'
IM2_PATH = 'data/taxi/taxi10.png'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)

    u, v = nagel(I1, I2, N=1000, alpha=0.01, delta=0.1)
    utils.plot_flow_results(u, v, save_path='taxi_nagel.png')
