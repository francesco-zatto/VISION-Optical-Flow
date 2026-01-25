from nagel import run_nagel, plot_error_vs_delta
from middlebury import readflo
import numpy as np
from PIL import Image

IM1_PATH = 'data/rubberwhale/frame10.png'
IM2_PATH = 'data/rubberwhale/frame11.png'
GT_PATH = 'data/rubberwhale/correct_rubberwhale10.flo'

if __name__ == "__main__":
    I1 = np.array(Image.open(IM1_PATH).convert('L'), dtype=np.float32)
    I2 = np.array(Image.open(IM2_PATH).convert('L'), dtype=np.float32)
    GT = readflo(GT_PATH)

    I1 /= 255.0
    I2 /= 255.0

    alphas = [0.001, 0.01, 0.1, 1.0]

    # run_nagel(I1, I2, GT, N=1000, alphas=alphas, plot=True, data_name='rubberwhale')

    plot_error_vs_delta(I1, I2, GT, deltas=[0.01, 0.05, 0.1, 1], alpha=0.01, N=500, data_name='rubberwhale')

    