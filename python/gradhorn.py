import numpy as np

def gradhorn(I1, I2):
    h, w = I1.shape
    
    Ix_valid = 0.25 * (I1[:-1, 1:] - I1[:-1, :-1] + I1[1:, 1:] - I1[1:, :-1] +
                       I2[:-1, 1:] - I2[:-1, :-1] + I2[1:, 1:] - I2[1:, :-1])
    
    Iy_valid = 0.25 * (I1[1:, :-1] - I1[:-1, :-1] + I1[1:, 1:] - I1[:-1, 1:] +
                       I2[1:, :-1] - I2[:-1, :-1] + I2[1:, 1:] - I2[:-1, 1:])
    
    It_valid = 0.25 * (I2[:-1, :-1] - I1[:-1, :-1] + I2[1:, :-1] - I1[1:, :-1] +
                       I2[:-1, 1:] - I1[:-1, 1:] + I2[1:, 1:] - I1[1:, 1:])

    Ix = np.zeros((h, w), dtype=np.float32)
    Iy = np.zeros((h, w), dtype=np.float32)
    It = np.zeros((h, w), dtype=np.float32)

    Ix[:-1, :-1] = Ix_valid
    Iy[:-1, :-1] = Iy_valid
    It[:-1, :-1] = It_valid

    Ix[:, -1] = Ix[:, -2]
    Iy[:, -1] = Iy[:, -2]
    It[:, -1] = It[:, -2]
    
    Ix[-1, :] = Ix[-2, :]
    Iy[-1, :] = Iy[-2, :]
    It[-1, :] = It[-2, :]

    return Ix, Iy, It