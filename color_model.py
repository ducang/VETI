import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def regularize_cov_np(sigma, min_eig=1e-4):
    sigma = 0.5 * (sigma + sigma.T)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, min_eig, None)
    sigma_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
    sigma_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    return sigma_reg.astype(np.float64), sigma_inv.astype(np.float64)

def _mahalanobis(c, mus, sigma_invs_reg):
    diff = c.unsqueeze(1) - mus.unsqueeze(0)
    sinv_diff = (sigma_invs_reg @ diff.unsqueeze(-1)).squeeze(-1)
    return (diff * sinv_diff).sum(-1).min(dim=1).values


def approx_RepScore(c, mus, sigma_invs_reg, tau):
    if mus.shape[0] == 0:
        return torch.full((c.shape[0],), tau ** 2 + 1, device=DEVICE, dtype=DTYPE)
    return _mahalanobis(c, mus, sigma_invs_reg)

def get_vote(gradMag, repScore):
    return torch.exp(-gradMag) * (1.0 - torch.exp(-repScore))

def pick_seed(bin_indices, gradMag, best_bin, active_mask):
    H, W = gradMag.shape
    match = (
        (bin_indices[:, :, 0] == best_bin[0])
        & (bin_indices[:, :, 1] == best_bin[1])
        & (bin_indices[:, :, 2] == best_bin[2])
        & active_mask
    )
    if not match.any():
        return None

    match_f = match.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones(1, 1, 21, 21, device=DEVICE, dtype=torch.float32)
    S_p = F.conv2d(match_f, kernel, padding=10).squeeze()
    score = S_p * torch.exp(-gradMag.float())
    score[~match] = -float("inf")

    flat_idx = score.argmax()
    return int(flat_idx // W), int(flat_idx % W)

def estimate_normal(im, seed, gf, neighborhood_radius):
    H, W, _ = im.shape
    y, x = seed

    impulse = np.zeros((H, W), dtype=np.float32)
    impulse[y, x] = 1.0

    weights = gf.filter(impulse).astype(np.float64)
    weights = np.clip(weights, 0.0, None)

    mask = np.zeros((H, W), dtype=np.float64)
    y0, y1 = max(0, y - neighborhood_radius), min(H, y + neighborhood_radius + 1)
    x0, x1 = max(0, x - neighborhood_radius), min(W, x + neighborhood_radius + 1)
    mask[y0:y1, x0:x1] = 1.0

    weights *= mask
    w_sum = weights.sum()
    if w_sum <= 1e-12:
        return None

    weights /= w_sum
    pixels = im.reshape(-1, 3)
    w = weights.reshape(-1, 1)

    mu = (pixels * w).sum(axis=0)
    diff = pixels - mu
    sigma = (diff * w).T @ diff
    return mu, sigma

MIN_EIG_REP_DEFAULT = 1e-4   
MIN_EIG_SCU_DEFAULT = 5e-3    

def extract_color_model(
    im,
    tau=5.0,
    gf_eps=1e-2,
    min_vote=100,
    min_eig_scu=MIN_EIG_SCU_DEFAULT,
    min_eig_rep=MIN_EIG_REP_DEFAULT,
    gf_radius=None,
    neighborhood_radius=None,
    return_debug=False
):
    H, W, _ = im.shape

    scale = np.sqrt((H * W) / 1_000_000.0)
    if gf_radius is None:
        gf_radius = max(10, int(20 * scale))
    if neighborhood_radius is None:
        neighborhood_radius = gf_radius

    gx = cv.Sobel(im, cv.CV_64F, 1, 0)
    gy = cv.Sobel(im, cv.CV_64F, 0, 1)
    gradMag = torch.tensor(
        np.sqrt(np.sum(gx ** 2 + gy ** 2, axis=2)),
        device=DEVICE, dtype=DTYPE,
    )

    bin_indices_np = np.clip((im * 10).astype(np.int32), 0, 9)
    bin_indices = torch.tensor(bin_indices_np, device=DEVICE, dtype=torch.long)

    c_flat = torch.tensor(im, device=DEVICE, dtype=DTYPE).reshape(-1, 3)

    gf = cv.ximgproc.createGuidedFilter(im.astype(np.float32), radius=gf_radius, eps=gf_eps )

    flat_bin = bin_indices[:, :, 0] * 100 + bin_indices[:, :, 1] * 10 + bin_indices[:, :, 2]

    represented_mask = torch.zeros(H, W, device=DEVICE, dtype=torch.bool)
    mus_t = torch.empty(0, 3, device=DEVICE, dtype=DTYPE)
    sigma_invs_rep_t = torch.empty(0, 3, 3, device=DEVICE, dtype=DTYPE)

    color_distr = []


    while True:
        active = ~represented_mask
        active_flat = active.reshape(-1)

        c_active = c_flat[active_flat]
        repScore_active = approx_RepScore(c_active, mus_t, sigma_invs_rep_t, tau)

        repScore_flat = torch.zeros(H * W, device=DEVICE, dtype=DTYPE)
        repScore_flat[active_flat] = repScore_active
        repScore = repScore_flat.reshape(H, W)

        represented_mask |= (repScore <= tau ** 2) & active

        active = ~represented_mask
        vp = torch.zeros(H, W, device=DEVICE, dtype=DTYPE)
        vp[active] = get_vote(gradMag[active], repScore[active])

        bins_flat = torch.zeros(1000, device=DEVICE, dtype=DTYPE)
        bins_flat.scatter_add_(0, flat_bin[active].reshape(-1), vp[active].reshape(-1))
        bins = bins_flat.reshape(10, 10, 10)

        best_flat = bins.argmax()
        best_bin = (int(best_flat // 100), int((best_flat % 100) // 10), int(best_flat % 10))

        if bins[best_bin].item() < min_vote:
            break

        seed = pick_seed(bin_indices, gradMag, best_bin, active)
        if seed is None:
            break

        result = estimate_normal(im, seed, gf, neighborhood_radius)
        if result is None:
            break

        mu, sigma = result

        sigma_rep, sigma_inv_rep = regularize_cov_np(sigma, min_eig=min_eig_rep)
        sigma_scu, sigma_inv_scu = regularize_cov_np(sigma, min_eig=min_eig_scu)

        color_distr.append({"mu": mu, "sigma": sigma_scu, "sigma_inv": sigma_inv_scu})

        mus_t = torch.cat([mus_t, torch.tensor(mu, device=DEVICE, dtype=DTYPE).unsqueeze(0)])
        sigma_invs_rep_t = torch.cat([
            sigma_invs_rep_t,
            torch.tensor(sigma_inv_rep, device=DEVICE, dtype=DTYPE).unsqueeze(0),
        ])

    if return_debug:
        debug = {
            "gf_radius": gf_radius,
            "gf_eps": gf_eps,
            "neighborhood_radius": neighborhood_radius,
            "tau": tau,
            "min_vote": min_vote,
            "min_eig_rep": min_eig_rep,
            "min_eig_scu": min_eig_scu,
            "num_distributions": len(color_distr),
        }
        return color_distr, debug

    return color_distr