import numpy as np
from gradhorn import gradhorn
import utils
import error_functions as err

def lucas(I1, I2, window_size = 5, window_type='square'):
    """
    window_type: 'square', 'gaussian', or 'circular'
    """

    assert window_size > 0 and window_size % 2 == 1, "window_size must be a positive odd integer"

    Ix, Iy, It = gradhorn(I1, I2)
    skip = window_size // 2
    u = np.zeros_like(I1, dtype=float)
    v = u.copy()
    n, m = I1.shape

    if window_type == 'square':
        W = np.ones(window_size**2)
    elif window_type == 'gaussian':
        sigma = window_size / 6
        kernel_1d = np.linspace(-skip, skip, window_size)
        gauss_1d = np.exp(-kernel_1d**2 / (2 * sigma**2))
        W = np.outer(gauss_1d, gauss_1d).flatten()
    elif window_type == 'circular':
        y, x = np.ogrid[-skip:skip+1, -skip:skip+1]
        mask = x**2 + y**2 <= (window_size/2)**2
        W = mask.astype(float).flatten()
    else:
        raise ValueError("unknown window_type")

    W /= np.sum(W)
    W = np.diag(W)

    for i in range(skip, n-skip):
        for j in range(skip, m-skip):
            Ix_window_flattened = np.ravel(Ix[i-skip:i+skip+1, j-skip:j+skip+1])
            Iy_window_flattened = np.ravel(Iy[i-skip:i+skip+1, j-skip:j+skip+1])
            It_window_flattened = np.ravel(It[i-skip:i+skip+1, j-skip:j+skip+1])

            B = -It_window_flattened.T
            A = np.stack([Ix_window_flattened, Iy_window_flattened], axis=1)

            # here we are weighting least squares by using the W diagonal matrix
            AT_W = A.T @ W 
            AT_W_A = AT_W @ A
            AT_W_B = AT_W @ B

            AT_A = A.T @ A

            det = AT_W_A[0,0]*AT_W_A[1,1] - AT_W_A[1,0] * AT_W_A[0,1]

            if det < 1e-6:
                continue

            AT_W_A_inv = np.array([[AT_A[1,1], -AT_A[0,1]],[-AT_A[1,0], AT_A[0,0]]]) / det

            w = AT_W_A_inv @ AT_W_B

            u[i,j] = w[0]
            v[i,j] = w[1]
    
    return u, v

def run_window_size_search(I1, I2, GT, window_sizes, window_type='square', plot=False, compute_stats=True, data_name=''):
    errors = []
    stats = {} if GT is not None else None
    optimal_window_size = None
    for window_size in window_sizes:
        u, v = lucas(I1, I2, window_size, window_type)
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


def run_lucas(I1, I2, GT=None, window_sizes=[3,5,7,9,11], window_type='square', plot=False, data_name=''):
    optimal_window_size, stats = run_window_size_search(I1, I2, GT, window_sizes, window_type, plot=plot, compute_stats=True, data_name=data_name)

    if optimal_window_size:
        u, v = lucas(I1, I2, optimal_window_size)
        utils.plot_flow_results(u, v, save_path=f'{data_name}_{optimal_window_size}.png')
    if stats:
        utils.print_stats(stats)