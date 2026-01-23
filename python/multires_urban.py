from multires import run_multires
import matplotlib.pyplot as plt
import utils

IM1_PATH = '../data2/eval-data-gray/Urban/frame09.png'
IM2_PATH = '../data2/eval-data-gray/Urban/frame10.png'

if __name__ == "__main__":
    I1 = plt.imread(IM1_PATH)
    I2 = plt.imread(IM2_PATH)

    print(I1.shape)

    u, v = run_multires(I1, I2, W=7)
    utils.plot_flow_results(u, v)
