import torch
import numpy as np
from gradhorn import gradhorn
import utils

def gradient(f: torch.Tensor):
    fx_valid = f[:, 2:] - f[:, :-2]
    fx = torch.zeros_like(f)
    fx[:, 1:-1] = fx_valid / 2

    fy_valid = f[2:, :] - f[:-2, :]
    fy = torch.zeros_like(f)
    fy[1:-1, :] = fy_valid / 2

    return torch.stack([fx, fy], dim=-1)

def horn_ad_loss(u: torch.Tensor, v: torch.Tensor, Ix: torch.Tensor, Iy: torch.Tensor, It: torch.Tensor, alpha=1e-3, norm='l2', huber_delta=0.01):
    """
    norm: `'l1'`, `'l2'` or `'huber'`
    """
    data_term = torch.sum((Ix * u + Iy * v + It) ** 2)
    
    u_grad = gradient(u)
    v_grad = gradient(v)

    if norm == 'l1':
        reg_term = torch.mean(torch.abs(u_grad)) + torch.mean(torch.abs(v_grad))
    elif norm == 'l2':
        reg_term = torch.mean(u_grad ** 2) + torch.mean(v_grad ** 2)
    elif norm == 'huber':
        huber = torch.nn.functional.huber_loss
        reg_term = huber(u_grad, torch.zeros_like(u_grad), reduction='mean', delta=huber_delta) + huber(v_grad, torch.zeros_like(v_grad), reduction='mean', delta=huber_delta)
    else:
        raise ValueError(f"Unknown norm: {norm}. Use 'l1', 'l2', or 'huber'")
    
    return data_term + alpha * reg_term

def run_horn_ad(I1: np.ndarray, I2: np.ndarray, alpha=1e-1, norm='l2', huber_delta=0.01, lr=1e-1, max_iter=1000, GT=None, plot=False, data_name=''):
    Ix, Iy, It = gradhorn(I1, I2)

    Ix = torch.tensor(Ix, dtype=torch.float32)
    Iy = torch.tensor(Iy, dtype=torch.float32)
    It = torch.tensor(It, dtype=torch.float32)

    u = torch.zeros(I1.shape, dtype=torch.float32, requires_grad=True)
    v = torch.zeros(I1.shape, dtype=torch.float32, requires_grad=True)
    
    optimizer = torch.optim.LBFGS([u,v], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        L = horn_ad_loss(u, v, Ix, Iy, It, alpha=alpha, norm=norm, huber_delta=huber_delta)
        L.backward()
        return L

    optimizer.step(closure)
    
    u_np = u.detach().numpy()
    v_np = v.detach().numpy()

    if GT is not None:
        w_e = np.stack((u_np, v_np), axis=2)
        utils.print_stats(utils.get_stats(GT, w_e, 0))

    if plot:
        utils.plot_flow_results(u_np, v_np, save_path=f'{data_name}.png')    

    return u_np, v_np