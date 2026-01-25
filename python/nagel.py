import numpy as np
import error_functions as err
import matplotlib.pyplot as plt
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


def nagel(I1, I2, N, alpha, delta=0.1):
    Ix, Iy, It = gradhorn(I1, I2)
    Ixx, Ixy, _ = gradhorn(Ix, Ix)
    _, Iyy, _ = gradhorn(Iy, Iy)
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

def run_alpha_search(I1, I2, GT, alphas, delta=0.1, N=1000, plot=False, compute_stats=True, data_name=''):
    errors = []
    stats = {} if GT is not None else None
    optimal_alpha = None
    for alpha in alphas:
        u, v = nagel(I1, I2, N, alpha, delta)
        w_e = np.stack((u, v), axis=2)
        if GT is not None:
            mean, _ = err.angular_error(GT, w_e)
            print(f'Alpha: {alpha}, Error: {mean:.5f}')
        else:
            print(f'Alpha: {alpha}')
        if plot:
            utils.plot_flow_results(u, v, save_path=f'plot_nagel/{data_name}_{alpha}.png')
        if GT is not None and compute_stats:
            stats.update(utils.get_stats(GT, w_e, alpha))
            errors.append(mean)
    if GT is not None:
        optimal_alpha = alphas[np.argmin(errors)]
        print(f'Optimal alpha: {optimal_alpha} with angular error: {min(errors):.5f}')
    return optimal_alpha, stats

def run_nagel(I1, I2, GT=None, delta=0.1, alphas=None, N=1000, plot=False, data_name=''):
    if alphas is None:
        alphas = 10.0 ** np.linspace(-4, 1, 6)

    optimal_alpha, stats = run_alpha_search(I1, I2, GT, alphas, delta, N=N, plot=plot, compute_stats=True, data_name=data_name)

    if optimal_alpha:
        u, v = nagel(I1, I2, N, optimal_alpha, delta)
        utils.plot_flow_results(u, v, save_path=f'plot_nagel/{data_name}_{optimal_alpha}.png')
    if stats:
        utils.print_stats(stats)  

def plot_error_vs_delta(I1, I2, GT, deltas, alpha, N=1000, data_name=''):
    mean_errors = {}
    
    for d in deltas:
        u, v = nagel(I1, I2, N, alpha, delta=d)
        
        if np.isnan(u).any() or np.isinf(u).any():
            mean_errors[d] = np.nan
            continue

        w_e = np.stack((u, v), axis=2)
        mean_error, _ = err.angular_error(GT, w_e)
        mean_errors[d] = mean_error

    plt.figure(figsize=(10, 6))
    plt.plot(deltas, mean_errors.values(), marker='o', linestyle='-', color='b')
    plt.xscale('log') 
    plt.xlabel('Delta (log scale)')
    plt.ylabel('Mean Angular Error')
    plt.title(f'Nagel Method Error vs. Delta (Alpha={alpha}, N={N})')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(f'plot_nagel/{data_name}_delta.png')
    plt.show()

    return mean_errors
