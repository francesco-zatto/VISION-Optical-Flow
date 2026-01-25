from middlebury import readflo
import matplotlib.pyplot as plt
import utils
from multires import run_multires
from lucas import run_lucas_window
import numpy as np

IM1_PATH = '../data2/other-data-gray/Urban2/frame10.png'
IM2_PATH = '../data2/other-data-gray/Urban2/frame11.png'
GT_PATH = '../data2/other-gt-flow/Urban2/flow10.flo'
    
if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    GT = readflo(GT_PATH)

    window_types = ['square']
    window_sizes = [15, 21, 27, 33, 39, 45]

    # run regular LK for comparison
    run_lucas_window(I1,I2, GT, window_sizes, window_types, plot=False, data_name='urban2')

    # determine best window size for multires
    window_sizes = [5, 7, 11, 15, 21]
    run_multires(I1,I2, GT, window_sizes, window_types, plot=False, data_name='Urban2')

    window_sizes = [9]

    # plot error as a function of K and run multires
    utils.plot_error_K(I1, I2, GT, window_types, window_sizes, 'Urban2')

    #GT_u = GT[:, :, 0]
    #GT_v = GT[:, :, 1]
    #utils.plot_flow_results(GT_u, GT_v, save_path=f'../plot2/urban2_gt.png')