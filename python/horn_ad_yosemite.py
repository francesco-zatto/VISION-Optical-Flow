import matplotlib.pyplot as plt
from middlebury import computeColor, readflo
from horn_ad import run_horn_ad
import utils

IM1_PATH = 'data/yosemite/yos9.png'
IM2_PATH = 'data/yosemite/yos10.png'
GT_PATH = 'data/yosemite/correct_yos.flo'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)
    GT = readflo(GT_PATH)

    u, v = run_horn_ad(I1, I2, norm='l2', alpha=4500, GT=GT, data_name='yosemite_4500', plot=True)    