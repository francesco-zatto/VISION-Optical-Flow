import numpy as np
from gradhorn import gradhorn
from scipy import signal
from skimage.transform import resize, warp
from lucas import lucas
import utils
import error_functions as err

def gaussianKernel(sigma):
    n2 = int(np.ceil(3*sigma))
    x,y = np.meshgrid(np.arange(-n2,n2+1),np.arange(-n2,n2+1))
    kern =  np.exp(-(x**2+y**2)/(2*sigma*sigma))
    return kern/kern.sum()

def applyGaussian(I, sigma):
    filter = gaussianKernel(sigma)
    mirror_filter = np.flip(np.flip(filter, axis=0), axis=1)

    return signal.convolve2d(I, mirror_filter, mode='same')

def warp_image(I, u, v):
    nr, nc = I.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    return warp(I, np.array([row_coords + v, col_coords + u]))

def multires(I1: np.ndarray, I2: np.ndarray, W: int = 3, window_type='square', K = 5):
    #K = int(np.ceil(np.log2(min(I1.shape[0], I1.shape[1]))))
    sigma = 2
    
    # level 1
    I1_Ks = [I1]
    I2_Ks = [I2]

    for _ in range(1, K):
        I1 = applyGaussian(I1, sigma)
        I1 = I1[::2,::2]

        I2 = applyGaussian(I2, sigma)
        I2 = I2[::2,::2]

        I1_Ks.append(I1)
        I2_Ks.append(I2)

        print(I1.shape)

    u = np.zeros_like(I1_Ks[-1])
    v = np.zeros_like(I1_Ks[-1])

    for n in range(K-1, -1, -1):
        I1 = I1_Ks[n]
        I2 = I2_Ks[n]

        u = resize(u, I1.shape, order=1, anti_aliasing=False) * 2
        v = resize(v, I1.shape, order=1, anti_aliasing=False) * 2

        I2_shifted = warp_image(I2, u, v)

        du, dv = lucas(I1,I2_shifted, window_type=window_type, window_size=W)

        u = u + du
        v = v + dv

    return u, v


def run_window_size_search(I1, I2, GT, window_sizes, window_type='square', plot=False, compute_stats=True, data_name=''):
    print(f"MULTI-RESOLUTION --- {window_type}")
    errors = []
    stats = {}
    optimal_window_size = None
    for window_size in window_sizes:
        u, v = multires(I1, I2, window_size, window_type)
        w_e = np.stack((u, v), axis=2)
        if GT is not None:
            mean, _ = err.angular_error(GT, w_e)
            print(f'window_size: {window_size}, Error: {mean:.5f}')
        else:
            print(f'window_size: {window_size}')
        if plot:
            utils.plot_flow_results(u, v, save_path=f'{data_name}_{window_size}.png')
        if GT is not None and compute_stats:
            stats.update(utils.get_stats(GT, w_e, window_size))
            errors.append(mean)
    if GT is not None:
        optimal_window_size = window_sizes[np.argmin(errors)]
        print(f'Optimal window_size: {optimal_window_size} with angular error: {min(errors):.5f}')
    return optimal_window_size, stats

def run_window_search(
    I1, I2, GT,
    window_sizes,
    window_types=['square','gaussian','circular'],
    plot=False,
    compute_stats=True,
    data_name='',
    K = 5
):
    print("\nMULTI-RESOLUTION")

    errors = {}          # (window_type, window_size) -> mean error
    stats = {}           # aggregated stats
    optimal = None       # (window_type, window_size)

    for window_type in window_types:
        print(f"\nWINDOW TYPE: {window_type}")

        for window_size in window_sizes:
            u, v = multires(I1, I2, window_size, window_type, K)
            w_e = np.stack((u, v), axis=2)

            if GT is not None:
                mean, _ = err.angular_error(GT, w_e)
                errors[(window_type, window_size)] = mean
                print(f'  window_size: {window_size}, error: {mean:.5f}')
            else:
                print(f'  window_size: {window_size}')

            if plot:
                utils.plot_flow_results(
                    u, v,
                    save_path=f'{data_name}_{window_type}_{window_size}.png'
                )

            if GT is not None and compute_stats:
                stats[(window_type, window_size)] = \
                    utils.get_stats_double(GT, w_e, window_size, window_type)

    if GT is not None and errors:
        optimal = min(errors, key=lambda k: errors[k])
        print(
            f'\nOptimal configuration: '
            f'window_type={optimal[0]}, window_size={optimal[1]} '
            f'with angular error: {errors[optimal]:.5f}'
        )

        return optimal[0], optimal[1], optimal, stats
    else:
        return None, None, None, stats


def run_multires(I1, I2, GT=None, window_sizes=[3,5,7,9,11], window_types=['square', 'gaussian', 'circular'], plot=False, data_name='', K=5):
    optimal_window_type, optimal_window_size, optimal, stats = run_window_search(I1, I2, GT, window_sizes, window_types, plot=plot, compute_stats=True, data_name=data_name, K=K)

    if optimal_window_size and optimal_window_type:
        u, v = multires(I1, I2, optimal_window_size, optimal_window_type, K)
        utils.plot_flow_results(u, v, save_path=f'../plot2/multires_{data_name}_{optimal_window_size}_{optimal_window_type}_{K}.png')
    if stats:
        utils.print_stats_double(stats)

    return stats