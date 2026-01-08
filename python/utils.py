import numpy as np
import matplotlib.pyplot as plt
from middlebury import computeColor
from horn import horn
import error_functions as err

def plot_flow_results(u, v, step=10, title="Optical Flow"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(computeColor(u, v))
    
    plt.subplot(1, 2, 2)
    x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))

    plt.quiver(x[::step, ::step], y[::step, ::step], 
               u[::step, ::step], v[::step, ::step], 
               color='red', angles='xy')
    
    plt.tight_layout()
    plt.show()

def get_stats(w_r, w_e, alpha):  
    return {
        alpha: {
            "EPE": err.end_point_error(w_r, w_e),
            "Angular": err.angular_error(w_r, w_e),
            "Norm": err.norm_error(w_r, w_e),
            "RelNorm": err.relative_norm_error(w_r, w_e)
        }
    }

def run_alpha_search(I1, I2, GT, alphas, N=1000, plot=False, compute_stats=True):
    errors = []
    stats = {}
    for alpha in alphas:
        u, v = horn(I1, I2, alpha, N)
        mean, _ = err.angular_error((u, v), GT)
        print(f'Alpha: {alpha}, Error: {mean:.5f}')
        if plot:
            plot_flow_results(u, v, title=f'Alpha: {alpha}, Error: {mean:.5f}')
        if compute_stats:
            stats.update(get_stats(GT, (u, v), alpha))
        errors.append(mean)
    optimal_alpha = alphas[np.argmin(errors)]
    print(f'Optimal alpha: {optimal_alpha} with angular error: {min(errors):.5f}')
    return optimal_alpha, stats