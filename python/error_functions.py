import numpy as np

epsilon = 1e-8

def end_point_error(w_r, w_e):
    epe = np.sqrt(np.sum((w_r - w_e) ** 2, axis=2))
    return np.mean(epe), np.std(epe)

def angular_error(w_r, w_e):
    norm_r = np.linalg.norm(w_r, axis=2)
    norm_e = np.linalg.norm(w_e, axis=2)

    cos_sim = (np.sum(w_r * w_e, axis=2)) / (norm_r * norm_e + epsilon)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    ang_error = np.arccos(cos_sim)
    
    return np.mean(ang_error), np.std(ang_error)

import numpy as np

def angular_error_space_time(w_r, w_e):
    dot_product = np.sum(w_r * w_e, axis=2)
    num = 1 + dot_product
    
    den_r = np.sqrt(1 + np.sum(w_r**2, axis=2))
    den_e = np.sqrt(1 + np.sum(w_e**2, axis=2))

    cos = num / (den_r * den_e)
    cos = np.clip(cos, -1.0, 1.0)
    
    ang_error = np.arccos(cos)
    return np.mean(ang_error), np.std(ang_error)

def norm_error(w_r, w_e):
    norm_r = np.linalg.norm(w_r, axis=2)
    norm_e = np.linalg.norm(w_e, axis=2)
    norm_error = np.abs(norm_r - norm_e)
    return np.mean(norm_error), np.std(norm_error)

def relative_norm_error(w_r, w_e):
    norm_r = np.linalg.norm(w_r, axis=2)
    norm_e = np.linalg.norm(w_e, axis=2)
    rel_norm_error = np.abs(norm_r - norm_e) / (norm_r + epsilon)
    return np.mean(rel_norm_error), np.std(rel_norm_error)