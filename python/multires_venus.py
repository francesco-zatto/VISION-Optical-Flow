from middlebury import readflo
import matplotlib.pyplot as plt
import utils
from multires import run_multires
from lucas import run_lucas_window

IM1_PATH = '../data2/other-data-gray/Venus/frame10.png'
IM2_PATH = '../data2/other-data-gray/Venus/frame11.png'
GT_PATH = '../data2/other-gt-flow/Venus/flow10.flo'
    
if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    GT = readflo(GT_PATH)

    window_types = ['square']
    window_sizes = [15, 21, 27, 33]

    # run 
    run_lucas_window(I1,I2, GT, window_sizes, window_types, plot=False, data_name='venus')
    
    # determine best window size for multires
    window_sizes = [5, 7, 11, 15, 21]
    run_multires(I1,I2, GT, window_sizes, window_types, plot=False, data_name='venus')

    # plot error as a function of K and run multires
    window_sizes = [7]
    utils.plot_error_K(I1, I2, GT, window_types, window_sizes, 'Venus')

    GT_u = GT[:, :, 0]
    GT_v = GT[:, :, 1]
    utils.plot_flow_results(GT_u, GT_v, save_path=f'../plot2/venus_gt.png')