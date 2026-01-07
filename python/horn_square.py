from gradhorn import gradhorn
from horn import horn
from middlebury import computeColor
import matplotlib.pyplot as plt


IM1_PATH = '../data/square/square9.png'
IM2_PATH = '../data/square/square10.png'

def debug_gradient(I1, I2, show=False):
    Ix, Iy, It = gradhorn(I1, I2)
    if show:
        plt.imshow(Ix, cmap='gray')
        plt.show()
        plt.imshow(Iy, cmap='gray')
        plt.show()
        plt.imshow(It, cmap='gray')
        plt.show()
    

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)

    debug_gradient(I1, I2)

    u, v = horn(I1, I2)
    computeColored = computeColor(u, v, True)
    plt.imshow(computeColored)
    plt.show()