from horn import run_horn
import matplotlib.pyplot as plt
from middlebury import computeColor, readflo
import utils

IM1_PATH = '../data/mysine/mysine9.png'
IM2_PATH = '../data/mysine/mysine10.png'
GT_PATH = '../data/mysine/correct_mysine.flo'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    GT = readflo(GT_PATH)

    alphas = [0.01, 0.1, 1.0, 10.0]

    run_horn(I1, I2, GT=GT, alphas=alphas, N=1000, plot=True, data_name='mysine')

    GT_u = GT[:, :, 0]
    GT_v = GT[:, :, 1]
    utils.plot_flow_results(GT_u, GT_v, save_path=f'mysine_gt.png')
    