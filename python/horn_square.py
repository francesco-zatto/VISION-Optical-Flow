from gradhorn import gradhorn
from horn import run_horn
from middlebury import readflo
import matplotlib.pyplot as plt


IM1_PATH = '../data/square/square9.png'
IM2_PATH = '../data/square/square10.png'
GT_PATH = '../data/square/correct_square.flo'

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
    GT = readflo(GT_PATH)

    debug_gradient(I1, I2)

    alphas = [0.001, 0.01, 0.1, 1.0]

    run_horn(I1, I2, GT=GT, N=1000, alphas=alphas, plot=True, data_name='square')