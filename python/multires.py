import numpy as np
from gradhorn import gradhorn
from scipy import signal
from skimage.transform import resize, warp
from lucas import lucas

def gaussianKernel(sigma):
    n2 = int(np.ceil(3*sigma))
    x,y = np.meshgrid(np.arange(-n2,n2+1),np.arange(-n2,n2+1))
    kern =  np.exp(-(x**2+y**2)/(2*sigma*sigma))
    return kern/kern.sum()

def applyGaussian(I, sigma):
    filter = gaussianKernel(sigma)
    mirror_filter = np.flip(np.flip(filter, axis=0), axis=1)

    return signal.convolve2d(I, mirror_filter, mode='same')

def warp_image(I, u, v):
    nr, nc = I.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    return warp(I, np.array([row_coords + v, col_coords + u]))

def run_multires(I1: np.ndarray, I2: np.ndarray, W: int = 3):
    K = int(np.ceil(np.log2(min(I1.shape[0], I1.shape[1]))))
    sigma = 2

    print(K)
    
    # level 1
    I1_Ks = [I1]
    I2_Ks = [I2]

    for _ in range(1, K):
        I1 = applyGaussian(I1, sigma)
        I1 = I1[::2,::2]

        I2 = applyGaussian(I2, sigma)
        I2 = I2[::2,::2]

        I1_Ks.append(I1)
        I2_Ks.append(I2)

    print([I.shape for I in I1_Ks])
    u = np.zeros_like(I1_Ks[-1])
    v = np.zeros_like(I1_Ks[-1])

    print(u.shape)

    for n in range(K-1, -1, -1):
        I1 = I1_Ks[n]
        I2 = I2_Ks[n]

        u = resize(u, I1.shape, order=1, anti_aliasing=False) * 2
        v = resize(v, I1.shape, order=1, anti_aliasing=False) * 2

        I2_shifted = warp_image(I2, u, v)

        du, dv = lucas(I1,I2_shifted, window_size=W)

        u = u + du
        v = v + dv

    return u, v