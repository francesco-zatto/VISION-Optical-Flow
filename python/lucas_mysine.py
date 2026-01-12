from lucas import run_lucas
import matplotlib.pyplot as plt
from middlebury import readflo

IM1_PATH = '../data/mysine/mysine9.png'
IM2_PATH = '../data/mysine/mysine10.png'
GT_PATH = '../data/mysine/correct_mysine.flo'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    GT = readflo(GT_PATH)

    window_sizes = [3,5,7,9,11]

    run_lucas(I1, I2, GT=GT, window_sizes=window_sizes, window_type='square', plot=False, data_name='lucas_mysine')
    