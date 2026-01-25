import numpy as np
import matplotlib.pyplot as plt
from middlebury import computeColor
import error_functions as err
from multires import run_multires

def plot_flow_results(u, v, step=10, save_path='result.png'):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(computeColor(u, v))
    
    plt.subplot(1, 2, 2)
    x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))

    plt.quiver(x[::step, ::step], y[::step, ::step], 
               u[::step, ::step], v[::step, ::step], 
               color='red', angles='xy')
    plt.gca().invert_yaxis()
    plt.savefig(save_path)

    #plt.show()

def get_stats_double(w_r, w_e, window_size, window_type):  
    epe_m, epe_s = err.end_point_error(w_r, w_e)
    ang_m, ang_s = err.angular_error(w_r, w_e)
    nrm_m, nrm_s = err.norm_error(w_r, w_e)
    rel_m, rel_s = err.relative_norm_error(w_r, w_e)
    ang_spacetime_m, ang_spacetime_s = err.angular_error_space_time(w_r, w_e)

    return {
        (window_size, window_type): {
            "EPE": {"mean": epe_m, "std": epe_s},
            "Angular": {"mean": ang_m, "std": ang_s},
            "Norm": {"mean": nrm_m, "std": nrm_s},
            "RelNorm": {"mean": rel_m, "std": rel_s},
            "AngularSpaceTime": {"mean": ang_spacetime_m, "std": ang_spacetime_s},
        }
    }

def get_stats(w_r, w_e, alpha):  
    epe_m, epe_s = err.end_point_error(w_r, w_e)
    ang_m, ang_s = err.angular_error(w_r, w_e)
    nrm_m, nrm_s = err.norm_error(w_r, w_e)
    rel_m, rel_s = err.relative_norm_error(w_r, w_e)
    ang_spacetime_m, ang_spacetime_s = err.angular_error_space_time(w_r, w_e)

    return {
        alpha: {
            "EPE": {"mean": epe_m, "std": epe_s},
            "Angular": {"mean": ang_m, "std": ang_s},
            "Norm": {"mean": nrm_m, "std": nrm_s},
            "RelNorm": {"mean": rel_m, "std": rel_s},
            "AngularSpaceTime": {"mean": ang_spacetime_m, "std": ang_spacetime_s},
        }
    }

def print_stats(stats):
    for optimal, measures in stats.items():
        for measure, values in measures.items():
            print(f"  {measure}: {values['mean']:.5f} +- {values['std']:.5f}")

def print_stats_double(stats):
    for optimal, measures in stats.items():
        print(f"Configuration: {optimal}")

        # measures is {(21,'square'): {...}}
        for (win, shape), metrics in measures.items():
            print(f"  Window: {win}, Type: {shape}")

            for measure, values in metrics.items():
                print(
                    f"    {measure}: "
                    f"{values['mean']:.5f} +- {values['std']:.5f}"
                )


def plot_error_K(I1, I2, GT, window_types, window_sizes, data_name):
    largest_K = int(np.ceil(np.log2(min(I1.shape[0], I1.shape[1]))))
    Ks = range(2, largest_K)

    means = []
    stddevs = []

    window_type = window_types[0]
    window_size = window_sizes[0]
    
    outer_key = (window_type, window_size)
    inner_key = (window_size, window_type)

    for K in Ks:
        print(f"Levels: {K}")
        stats = run_multires(I1,I2, GT, window_sizes, window_types, plot=False, data_name=data_name, K = K)
        angular_mean = stats[outer_key][inner_key]['Angular']['mean']
        angular_std = stats[outer_key][inner_key]['Angular']['std']
        print(f"Angular Mean: {angular_mean}")
        print(f"Angular Std:  {angular_std}")
        means.append(angular_mean)
        stddevs.append(angular_std)
    
    x_values = list(Ks)

    plt.figure(figsize=(10, 6))

    plt.errorbar(x_values, means, fmt='-o', linewidth=2, capsize=5, label='Mean Angular Error')

    plt.title(f'Angular Error vs. Pyramid Levels (K)\nWindow: Square, Size: 9')
    plt.xlabel('Number of Pyramid Levels (K)')
    plt.ylabel('Angular Error')
    plt.xticks(x_values) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.25, 0.55)
    plt.legend()

    plt.savefig(f'../plot2/multires_{data_name}_angular_error_vs_K.png', dpi=300, bbox_inches='tight')