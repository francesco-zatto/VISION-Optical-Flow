from horn import horn
import error_functions
import matplotlib.pyplot as plt
import numpy as np
from middlebury import readflo, computeColor

IM1_PATH = '../data/mysine/mysine9.png'
IM2_PATH = '../data/mysine/mysine10.png'
GT_PATH = '../data/mysine/correct_mysine.flo'

def find_optimal_alpha(I1, I2, GT, alphas, N=1000):
    errors = []
    for alpha in alphas:
        horn_flow = horn(I1, I2, alpha, N)
        mean, _ = error_functions.end_point_error(horn_flow, GT)
        errors.append(mean)
    optimal_alpha = alphas[np.argmin(errors)]
    print(f'Optimal alpha: {optimal_alpha} with EPE: {min(errors)}')
    return optimal_alpha

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    GT = readflo(GT_PATH)
    GT = GT[:-1, :-1].transpose(2, 0, 1)

    alphas = 10.0 ** np.linspace(-5, 1, 7)
    optimal_alpha = find_optimal_alpha(I1, I2, GT, alphas)

    u, v = horn(I1, I2, 0.1, N=1000)
    computeColored = computeColor(u, v, True)
    plt.imshow(computeColored)
    plt.show()
    