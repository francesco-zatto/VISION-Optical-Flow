from nagel import run_nagel, plot_error_vs_delta
from middlebury import readflo
import matplotlib.pyplot as plt

IM1_PATH = 'data/yosemite/yos9.png'
IM2_PATH = 'data/yosemite/yos10.png'
GT_PATH = 'data/yosemite/correct_yos.flo'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    GT = readflo(GT_PATH)

    alphas = [0.01, 0.05, 0.1, 0.2]

    run_nagel(I1, I2, GT=GT, N=1000, alphas=alphas, plot=True, data_name='yosemite')

    plot_error_vs_delta(I1, I2, GT, deltas=[0.01, 0.05, 0.1, 1], alpha=0.01, N=500, data_name='yosemite')
