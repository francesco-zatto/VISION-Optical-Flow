from horn import run_horn
import matplotlib.pyplot as plt

IM1_PATH = '../data/taxi/taxi9.png'
IM2_PATH = '../data/taxi/taxi10.png'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)

    run_horn(I1, I2, N=1000, plot=True)