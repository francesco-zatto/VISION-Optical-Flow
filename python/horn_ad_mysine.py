import matplotlib.pyplot as plt
from middlebury import computeColor, readflo
import utils
from horn_ad import run_horn_ad

IM1_PATH = 'data/mysine/mysine9.png'
IM2_PATH = 'data/mysine/mysine10.png'
GT_PATH = 'data/mysine/correct_mysine.flo'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    GT = readflo(GT_PATH)

    u, v = run_horn_ad(I1, I2, norm='l2', alpha=100, GT=GT, plot=True, lr=1e-1, data_name='mysine_100')