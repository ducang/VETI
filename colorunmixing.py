import torch
import numpy as np
import cv2 as cv
import json
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

def ncg(f_fn, jac_fn, x, lamb, rho, maxiter=100, gtol=1e-6, step_tol=1e-8):
    P, D = x.shape
    x_cur = x.clone().clamp(0.0, 1.0)

    def grad_clip(xv, grad):
        g = grad.clone()
        g[(xv <= 0) & (g > 0)] = 0.0
        g[(xv >= 1) & (g < 0)] = 0.0
        return g

    g = grad_clip(x_cur, jac_fn(x_cur, lamb, rho))
    d = -g.clone()

    active = g.norm(dim=-1) >= gtol

    for _ in range(maxiter):
        if not active.any():
            break

        gd = (g * d).sum(dim=-1)

        bad_dir = active & (gd >= 0)
        d = torch.where(bad_dir.unsqueeze(-1), -g, d)

        fx = f_fn(x_cur, lamb, rho)
        gd = (g * d).sum(dim=-1)

        alpha = torch.ones(P, 1, device=x.device, dtype=x.dtype)
        x_next = x_cur.clone()

        ls_active = active.clone()
        c1 = 1e-4

        for _ in range(20):
            if not ls_active.any():
                break

            x_cand = (x_cur + alpha * d).clamp(0.0, 1.0)
            fx_cand = f_fn(x_cand, lamb, rho)

            accept = ls_active & (fx_cand <= fx + c1 * alpha.squeeze(-1) * gd)

            x_next = torch.where(accept.unsqueeze(-1), x_cand, x_next)

            shrink = ls_active & (~accept)
            alpha = torch.where(shrink.unsqueeze(-1), alpha * 0.5, alpha)

            ls_active = shrink

        step_small = (x_next - x_cur).norm(dim=-1) < step_tol

        x_cur = torch.where(active.unsqueeze(-1), x_next, x_cur)

        g_new = grad_clip(x_cur, jac_fn(x_cur, lamb, rho))

        num = (g_new * (g_new - g)).sum(dim=-1)
        den = (g * g).sum(dim=-1) + 1e-16
        beta = (num / den).clamp(min=0.0)

        d_new = -g_new + beta.unsqueeze(-1) * d

        g = torch.where(active.unsqueeze(-1), g_new, g)
        d = torch.where(active.unsqueeze(-1), d_new, d)

        grad_small = g_new.norm(dim=-1) < gtol

        active = active & (~step_small) & (~grad_small)

    return x_cur

#  SCU  
def solver_SCU(c, mus, sigma_invs, sigmaa=10):

    P = c.shape[0]
    N = mus.shape[0]

    diffs = c.unsqueeze(1) - mus.unsqueeze(0)         
    sinv_diffs = (sigma_invs @ diffs.unsqueeze(-1)).squeeze(-1)  
    dists = (diffs * sinv_diffs).sum(-1)  

    best_idx = dists.argmin(dim=1)  

    alphas0 = torch.zeros(P, N, device=DEVICE, dtype=DTYPE)
    colors0 = mus.unsqueeze(0).expand(P, -1, -1).clone()  

    alphas0[torch.arange(P, device=DEVICE), best_idx] = 1.0
    colors0[torch.arange(P, device=DEVICE), best_idx] = c  

    x0 = torch.cat([alphas0, colors0.reshape(P, N * 3)], dim=1)  

    def G(x):
        a = x[:, :N]                            
        u = x[:, N:].reshape(P, N, 3)           
        h_color = (a.unsqueeze(-1) * u).sum(1) - c  
        h_alpha = a.sum(1, keepdim=True) - 1.0           
        return torch.cat([h_color ** 2, h_alpha ** 2], dim=1)  

    def objective_energy(x):
        a = x[:, :N]                            
        u = x[:, N:].reshape(P, N, 3)           
        diff = u - mus.unsqueeze(0)             
        sinv_diff = (sigma_invs @ diff.unsqueeze(-1)).squeeze(-1)
        dist = (diff * sinv_diff).sum(-1) 
        energy = (a * dist).sum(1)            

        s1 = a.sum(1)                           
        s2 = (a ** 2).sum(1) + 1e-6           
        sparsity = sigmaa * (s1 / s2 - 1.0)
        mask = (a ** 2).sum(1) > 1e-10
        sparsity = torch.where(mask, sparsity, torch.zeros_like(sparsity))

        return energy + sparsity

    def f(x, lamb, rho):
        Gx = G(x)
        return objective_energy(x) + (lamb * Gx).sum(1) + 0.5 * rho * (Gx ** 2).sum(1)

    def jacobian(x, lamb, rho):
        a = x[:, :N]                            
        u = x[:, N:].reshape(P, N, 3)            

        s1 = a.sum(1)                             
        s2 = (a ** 2).sum(1) + 1e-6             

        h_color = (a.unsqueeze(-1) * u).sum(1) - c  
        h_alpha = a.sum(1) - 1.0                          

        q = 2.0 * lamb[:, :3] * h_color + 2.0 * rho.unsqueeze(1) * (h_color ** 3)
        p = 2.0 * lamb[:, 3:] * h_alpha.unsqueeze(-1) + 2.0 * rho.unsqueeze(1) * (h_alpha ** 3).unsqueeze(-1)

        diff = u - mus.unsqueeze(0)                              
       
        sinv_diff = (sigma_invs @ diff.unsqueeze(-1)).squeeze(-1) 

        grad_energy_u = 2.0 * a.unsqueeze(-1) * sinv_diff       
        dist = (diff * sinv_diff).sum(-1)  

        mask = (a ** 2).sum(1, keepdim=True) > 1e-10  
        grad_sp = sigmaa * (s2.unsqueeze(1) - 2.0 * a * s1.unsqueeze(1)) / (s2.unsqueeze(1) ** 2)
        grad_sp = torch.where(mask.expand_as(grad_sp), grad_sp, torch.zeros_like(grad_sp))

        grad_energy_a = dist + grad_sp  

        grad_G_a = (q.unsqueeze(1) * u).sum(-1) + p  
        grad_G_u = q.unsqueeze(1) * a.unsqueeze(-1)            

        grad_a = grad_energy_a + grad_G_a       
        grad_u = grad_energy_u + grad_G_u       

        return torch.cat([grad_a, grad_u.reshape(P, N * 3)], dim=1)

    x = x0
    lamb = torch.full((P, 4), 0.1, device=DEVICE, dtype=DTYPE)
    rho = torch.full((P,), 0.1, device=DEVICE, dtype=DTYPE)
    beta_mul = 10.0
    gamma = 0.25
    eps = 1e-6

    outer_maxiter = 10
    ncg_maxiter = 10

    G_prev_norm = G(x).norm(dim=1)
    active = torch.ones(P, device=DEVICE, dtype=torch.bool)

    for _ in range(outer_maxiter):
        if not active.any():
            break

        x_next = ncg(f, jacobian, x, lamb, rho, maxiter=ncg_maxiter, gtol=1e-6)
        Gx = G(x_next)

        lamb_new = lamb + rho.unsqueeze(1) * Gx
        G_cur_norm = Gx.norm(dim=1)
        rho_new = torch.where(G_cur_norm > gamma * G_prev_norm, beta_mul * rho, rho)

        delta = (x_next - x).norm(dim=1)

        done = (delta < eps) & (G_cur_norm < eps)

        x = torch.where(active.unsqueeze(1), x_next, x)
        lamb = torch.where(active.unsqueeze(1), lamb_new, lamb)
        rho = torch.where(active, rho_new, rho)
        G_prev_norm = torch.where(active, G_cur_norm, G_prev_norm)

        active = active & (~done)

    alphas = x[:, :N]
    colors = x[:, N:].reshape(P, N, 3)
    return alphas, colors


#  Matte regularization 
def matte_regu(im_np, alphas_np, rad=60):

    H, W, N = alphas_np.shape
    scale = np.sqrt((H * W) / 1_000_000.0)
    radius = max(1, int(rad * scale))

    guide = np.clip(im_np * 255.0, 0, 255).astype(np.uint8)
    eps = 0.0001 * 255.0 * 255.0
    filtered = np.zeros_like(alphas_np, dtype=np.float32)
    for i in range(N):
        alpha_u8 = np.clip(alphas_np[:, :, i] * 255.0, 0, 255).astype(np.uint8)
        f = cv.ximgproc.guidedFilter(guide, alpha_u8, radius, eps)
        filtered[:, :, i] = f.astype(np.float32) / 255.0

    filtered = np.clip(filtered, 0.0, 1.0)
    s = np.sum(filtered, axis=2, keepdims=True)
    s[s <= 1e-8] = 1e-8
    return filtered / s


#  Color refinement 
def color_refine(c, mus, sigma_invs, alphashat, scu_colors):

    P = c.shape[0]
    N = mus.shape[0]

    x0 = scu_colors.reshape(P, N * 3).clone()

    def G(x):
        u = x.reshape(P, N, 3)
        h = (alphashat.unsqueeze(-1) * u).sum(1) - c   
        return h ** 2   
    
    def objective_energy(x):
        u = x.reshape(P, N, 3)
        diff = u - mus.unsqueeze(0)
        dist = (diff * (sigma_invs @ diff.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return (alphashat * dist).sum(1)

    def f(x, lamb, rho):
        Gx = G(x)
        return objective_energy(x) + (lamb * Gx).sum(1) + 0.5 * rho * (Gx ** 2).sum(1)

    def jacobian(x, lamb, rho):
        u = x.reshape(P, N, 3)
        diff = u - mus.unsqueeze(0)
        sinv_diff = (sigma_invs @ diff.unsqueeze(-1)).squeeze(-1)
        grad_energy_u = 2.0 * alphashat.unsqueeze(-1) * sinv_diff  

        h = (alphashat.unsqueeze(-1) * u).sum(1) - c
        q = 2.0 * lamb * h + 2.0 * rho.unsqueeze(1) * (h ** 3)
        grad_aug_u = q.unsqueeze(1) * alphashat.unsqueeze(-1)   
        return (grad_energy_u + grad_aug_u).reshape(P, N * 3)

    x = x0
    lamb = torch.full((P, 3), 0.1, device=DEVICE, dtype=DTYPE)
    rho = torch.full((P,), 0.1, device=DEVICE, dtype=DTYPE)
    beta_mul = 10.0
    gamma = 0.25
    eps = 1e-6
    outer_maxiter = 10
    ncg_maxiter = 10

    G_prev_norm = G(x).norm(dim=1)
    active = torch.ones(P, device=DEVICE, dtype=torch.bool)

    for _ in range(outer_maxiter):
        if not active.any():
            break

        x_next = ncg(f, jacobian, x, lamb, rho,
                      maxiter=ncg_maxiter, gtol=1e-6)
        Gx = G(x_next)

        lamb_new = lamb + rho.unsqueeze(1) * Gx
        G_cur_norm = Gx.norm(dim=1)
        rho_new = torch.where(G_cur_norm > gamma * G_prev_norm,
                              beta_mul * rho, rho)

        delta = (x_next - x).norm(dim=1)
        done = (delta < eps) & (G_cur_norm < eps)

        x = torch.where(active.unsqueeze(1), x_next, x)
        lamb = torch.where(active.unsqueeze(1), lamb_new, lamb)
        rho = torch.where(active, rho_new, rho)
        G_prev_norm = torch.where(active, G_cur_norm, G_prev_norm)

        active = active & (~done)

    colors = x.reshape(P, N, 3)
    return alphashat, colors

def distr_to_torch(color_distr_np):
    out = []
    for d in color_distr_np:
        out.append({
            'mu': torch.tensor(d['mu'], device=DEVICE, dtype=DTYPE),
            'sigma_inv': torch.tensor(d['sigma_inv'], device=DEVICE, dtype=DTYPE)
        })
    return out

def save_rgb_png(path, rgb01):
    rgb_u8 = (np.clip(rgb01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    bgr_u8 = cv.cvtColor(rgb_u8, cv.COLOR_RGB2BGR)
    cv.imwrite(str(path), bgr_u8)

def save_rgba_png(path, rgb01, alpha01):
    rgb_u8 = (np.clip(rgb01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    a_u8   = (np.clip(alpha01, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    bgra_u8 = np.dstack([rgb_u8[:, :, 2], rgb_u8[:, :, 1], rgb_u8[:, :, 0], a_u8])
    cv.imwrite(str(path), bgra_u8)