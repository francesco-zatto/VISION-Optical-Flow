from middlebury import readflo
from horn_ad import run_horn_ad
from PIL import Image
import numpy as np

IM1_PATH = 'data/rubberwhale/frame10.png'
IM2_PATH = 'data/rubberwhale/frame11.png'
GT_PATH = 'data/rubberwhale/correct_rubberwhale10.flo'

if __name__ == "__main__":
    I1 = np.array(Image.open(IM1_PATH).convert('L'), dtype=np.float32)
    I2 = np.array(Image.open(IM2_PATH).convert('L'), dtype=np.float32)
    GT = readflo(GT_PATH)

    I1 /= 255.0
    I2 /= 255.0

    u, v = run_horn_ad(I1, I2, norm='l2', alpha=700 , GT=GT, plot=True, data_name='rubberwhale_700', lr=0.5, max_iter=1000)