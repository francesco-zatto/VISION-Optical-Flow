from horn import run_horn
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

    u, v, optimal_alpha, stats = run_horn(I1, I2, GT=GT, N=1000, plot=True)
    