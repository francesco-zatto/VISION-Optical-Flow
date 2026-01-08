from gradhorn import gradhorn
import utils
import error_functions as err
import numpy as np
from scipy.signal import convolve2d

def horn(I1, I2, alpha=0.1, N=1000):
    Ix, Iy, It = gradhorn(I1, I2)
    u = v = np.zeros_like(Ix)
    A = np.array([
        [1/12, 1/6, 1/12],
        [1/6, 0, 1/6],
        [1/12, 1/6, 1/12]
    ])
    for _ in range(N):
        u_avg = convolve2d(u, A, mode='same', boundary='symm')
        v_avg = convolve2d(v, A, mode='same', boundary='symm')
        update_term = (Ix * u_avg + Iy * v_avg + It) / (alpha + Ix**2 + Iy**2)
        u = u_avg - Ix * update_term
        v = v_avg - Iy * update_term
    return u, v

def run_alpha_search(I1, I2, GT, alphas, N=1000, plot=False, compute_stats=True):
    errors = []
    stats = {} if GT is not None else None
    optimal_alpha = None
    for alpha in alphas:
        u, v = horn(I1, I2, alpha, N)
        w_e = np.stack((u, v), axis=2)
        if GT is not None:
            mean, _ = err.angular_error(GT, w_e)
            print(f'Alpha: {alpha}, Error: {mean:.5f}')
        else:
            print(f'Alpha: {alpha}')
        if plot:
            utils.plot_flow_results(u, v)
        if GT is not None and compute_stats:
            stats.update(utils.get_stats(GT, w_e, alpha))
            errors.append(mean)
    if GT is not None:
        optimal_alpha = alphas[np.argmin(errors)]
        print(f'Optimal alpha: {optimal_alpha} with angular error: {min(errors):.5f}')
    return optimal_alpha, stats

def run_horn(I1, I2, GT=None, alphas=None, N=1000, plot=False):
    if alphas is None:
        alphas = 10.0 ** np.linspace(-4, 1, 6)

    optimal_alpha, stats = run_alpha_search(I1, I2, GT, alphas, plot=plot, compute_stats=True)

    if optimal_alpha:
        u, v = horn(I1, I2, optimal_alpha, N)
        utils.plot_flow_results(u, v)
    if stats:
        utils.print_stats(stats)