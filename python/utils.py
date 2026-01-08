import numpy as np
import matplotlib.pyplot as plt
from middlebury import computeColor
import error_functions as err

def plot_flow_results(u, v, step=10):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(computeColor(u, v))
    
    plt.subplot(1, 2, 2)
    x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))

    plt.quiver(x[::step, ::step], y[::step, ::step], 
               u[::step, ::step], v[::step, ::step], 
               color='red', angles='xy')
    plt.gca().invert_yaxis()

    plt.show()

def get_stats(w_r, w_e, alpha):  
    epe_m, epe_s = err.end_point_error(w_r, w_e)
    ang_m, ang_s = err.angular_error(w_r, w_e)
    nrm_m, nrm_s = err.norm_error(w_r, w_e)
    rel_m, rel_s = err.relative_norm_error(w_r, w_e)

    return {
        alpha: {
            "EPE": {"mean": epe_m, "std": epe_s},
            "Angular": {"mean": ang_m, "std": ang_s},
            "Norm": {"mean": nrm_m, "std": nrm_s},
            "RelNorm": {"mean": rel_m, "std": rel_s}
        }
    }

def print_stats(stats):
    for alpha, measures in stats.items():
        print(f"Alpha: {alpha}")
        for measure, values in measures.items():
            print(f"  {measure}: {values['mean']:.5f} +- {values['std']:.5f}")