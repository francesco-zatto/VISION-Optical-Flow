from horn import horn
from middlebury import computeColor
import matplotlib.pyplot as plt

IM1_PATH = '../data/rubberwhale/frame10.png'
IM2_PATH = '../data/rubberwhale/frame11.png'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)

    u, v = horn(I1, I2)
    computeColored = computeColor(u, v, True)
    plt.imshow(computeColored)
    plt.show()