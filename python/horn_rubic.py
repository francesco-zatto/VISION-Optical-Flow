from horn import run_horn
import matplotlib.pyplot as plt

IM1_PATH = '../data/rubic/rubic9.png'
IM2_PATH = '../data/rubic/rubic10.png'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

    run_horn(I1, I2, N=1000, alphas=alphas, plot=True, data_name='rubic')