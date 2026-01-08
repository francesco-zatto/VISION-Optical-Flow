from horn import horn
import utils
import matplotlib.pyplot as plt
import numpy as np
from middlebury import readflo

IM1_PATH = '../data/mysine/mysine9.png'
IM2_PATH = '../data/mysine/mysine10.png'
GT_PATH = '../data/mysine/correct_mysine.flo'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    GT = readflo(GT_PATH)
    GT = GT[:-1, :-1].transpose(2, 0, 1)

    alphas = 10.0 ** np.linspace(-5, 1, 7)
    optimal_alpha, stats = utils.run_alpha_search(I1, I2, GT, alphas, plot=True, compute_stats=True)

    u, v = horn(I1, I2, optimal_alpha)
    if not stats:
        stats = utils.get_stats(GT, (u, v))
    utils.plot_flow_results(u, v)
    