# ============================================================
# NBGL: Noise-Aware Boundary-Enhanced Generative Learning
# for 3D Ultrasound Speckle Reduction
#
# Train: mixed blind despeckling across configured noise levels
# Val: mixed full-volume sliding evaluation
# Test: per-noise-level sliding inference, NIfTI saving, CSV metrics
# NIWG-wFiLM: interaction weights are estimated from the input itself
# via parameter-free 3D Laplacian + MAD noise estimation.
# ============================================================

import glob
import os
import random
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.checkpoint import checkpoint
from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Config
# ============================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DATA_ROOT = "./Databases/"

# --- NEW HYPERPARAMETER: Specify the exact noise levels to train/val/test together ---
TARGET_NOISE_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

# Consolidated output root for blind denoising model
OUT_ROOT = "Result/NBGL_Blind_Mixed_Noise"
os.makedirs(OUT_ROOT, exist_ok=True)

MAX_EPOCHS = 200
VAL_INTERVAL = 1

TRAIN_BS = 1  # outer batch: number of cases per loader step
VAL_BS = 2  # full-volume validation batch
TEST_BS = 2  # full-volume test batch

MICRO_BATCH_SIZE = 3  # number of patches per optimization step
NUM_WORKERS = 0

EPS = 1e-6

RUN_TRAIN = True
RUN_TEST = False
TEST_USE = "best"  # "best" | "latest"

PATCH_SIZE = (192, 192, 96)
SLIDE_CHUNK = 64
SLIDE_OVERLAP = 16

LATEST_CKPT = os.path.join(OUT_ROOT, "latest_ckpt.pth")
BEST_CKPT = os.path.join(OUT_ROOT, "best_ckpt.pth")
BEST_GEN = os.path.join(OUT_ROOT, "best_generator.pth")
BEST_DISC = os.path.join(OUT_ROOT, "best_discriminator.pth")

RESUME_TRAIN = True
RESUME_PATH = LATEST_CKPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Using device: {device}")

# =========================
# Train Monitoring Outputs
# =========================
DEBUG_GRAD = False
DEBUG_GRAD_EPOCHS = 5
SAVE_FIG_EVERY = 1
SMOOTH_WINDOW = 5
FIG_DIR = os.path.join(OUT_ROOT, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def nii_as_float32(path: str) -> np.ndarray:
    img = nib.load(path)
    arr = np.asarray(img.dataobj)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def odd_kernel_leq(base_k, size, max_k=None):
    if max_k is None:
        max_k = base_k
    k = min(base_k, max_k, size)
    if k % 2 == 0:
        k -= 1
    return max(k, 1)


def grad_report(module, name: str):
    n_all, n_has, norm_sum = 0, 0, 0.0
    for p in module.parameters():
        n_all += 1
        if p.grad is not None:
            n_has += 1
            norm_sum += float(p.grad.detach().data.norm(2).item())
    print(f"[{name}] params_with_grad: {n_has}/{n_all}, grad_norm_sum={norm_sum:.6e}")


def zero_all_grads(generator, discriminator):
    generator.zero_grad(set_to_none=True)
    discriminator.zero_grad(set_to_none=True)


def masked_rmse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    B = pred.shape[0]
    pred_f = pred.reshape(B, -1)
    gt_f = gt.reshape(B, -1)
    mask_f = mask.reshape(B, -1)

    denom = mask_f.sum(dim=1).clamp_min(1.0)
    mse = ((pred_f - gt_f) ** 2 * mask_f).sum(dim=1) / denom
    rmse = torch.sqrt(mse.clamp_min(eps))
    return rmse.mean()


def sanitize_ring_k(k_inner: int, k_outer: int):
    k_inner = max(1, int(k_inner))
    k_outer = max(1, int(k_outer))

    if k_inner % 2 == 0: k_inner -= 1
    if k_outer % 2 == 0: k_outer -= 1
    k_inner = max(1, k_inner)
    k_outer = max(1, k_outer)

    if k_inner >= k_outer:
        k_outer = max(3, k_inner + 2)
        if k_outer % 2 == 0:
            k_outer += 1
    return k_inner, k_outer


def dilate3d(mask01: torch.Tensor, k: int = 7) -> torch.Tensor:
    p = k // 2
    return F.max_pool3d(mask01, kernel_size=k, stride=1, padding=p)


# ============================================================
# Parameter-Free 3D Noise Estimator (MAD)
# ============================================================
def estimate_noise_mad_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Parameter-free noise estimation using 3D Laplacian and Median Absolute Deviation (MAD).
    x: [B, 1, D, H, W]
    Returns: sigma [B]
    """
    B, C, D, H, W = x.shape

    kernel = torch.zeros((1, 1, 3, 3, 3), dtype=x.dtype, device=x.device)
    kernel[0, 0, 1, 1, 1] = 6.0
    kernel[0, 0, 0, 1, 1] = -1.0
    kernel[0, 0, 2, 1, 1] = -1.0
    kernel[0, 0, 1, 0, 1] = -1.0
    kernel[0, 0, 1, 2, 1] = -1.0
    kernel[0, 0, 1, 1, 0] = -1.0
    kernel[0, 0, 1, 1, 2] = -1.0
    kernel = kernel / 6.0

    with torch.no_grad():
        high_freq_res = F.conv3d(x, kernel, padding=1)
        abs_res = torch.abs(high_freq_res).view(B, -1)
        median_val = torch.median(abs_res, dim=1)[0]
        sigma = 1.4826 * median_val

    return sigma


def cnr_snr_from_anno(vol: torch.Tensor, anno_roi: torch.Tensor, k_inner: int = 5, k_outer: int = 11,
                      eps: float = 1e-6):
    roi = (anno_roi > 0.5).float()
    B = vol.shape[0]

    k_inner, k_outer = sanitize_ring_k(k_inner, k_outer)

    dil_outer = dilate3d(roi, k=k_outer)
    dil_inner = dilate3d(roi, k=k_inner)
    ring = (dil_outer - dil_inner).clamp(0, 1)

    v = vol.reshape(B, -1)
    r = roi.reshape(B, -1)
    b = ring.reshape(B, -1)

    denom_r = r.sum(dim=1)
    denom_b = b.sum(dim=1)
    valid = (denom_r > 0) & (denom_b > 0)

    denom_r_safe = denom_r.clamp_min(1.0)
    denom_b_safe = denom_b.clamp_min(1.0)

    mu_r = (v * r).sum(dim=1) / denom_r_safe
    mu_b = (v * b).sum(dim=1) / denom_b_safe

    var_r = (((v - mu_r[:, None]) ** 2) * r).sum(dim=1) / denom_r_safe
    std_r = torch.sqrt(var_r.clamp_min(eps))

    var_b = (((v - mu_b[:, None]) ** 2) * b).sum(dim=1) / denom_b_safe
    std_b = torch.sqrt(var_b.clamp_min(eps))

    snr = mu_r / (std_r + eps)
    cnr = torch.abs(mu_r - mu_b) / (std_b + eps)

    nan = vol.new_full((B,), float("nan"))
    snr = torch.where(valid, snr, nan)
    cnr = torch.where(valid, cnr, nan)

    return cnr.mean(), snr.mean(), mu_r.mean(), mu_b.mean(), std_r.mean(), std_b.mean()


def _rmse_2d_torch_masked_batch(A3: torch.Tensor, B3: torch.Tensor, M3: torch.Tensor, eps: float = 1e-12):
    M = M3.float()
    denom = M.sum(dim=(1, 2))
    denom_safe = denom.clamp_min(1.0)
    mse = (((A3 - B3) ** 2) * M).sum(dim=(1, 2)) / denom_safe
    rmse = torch.sqrt(mse.clamp_min(eps))
    rmse = torch.where(denom > 0, rmse, torch.full_like(rmse, float("nan")))
    return rmse


# ============================================================
# Datasets (Patch + Full Volume)
# ============================================================
class DenoiseDataset3D(Dataset):
    def __init__(self, clean_dir, noisy_dir, anno_dir):
        self.items = []
        clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.nii")))
        for fp in clean_files:
            name = os.path.basename(fp)
            noisy_fp = os.path.join(noisy_dir, name)
            anno_fp = os.path.join(anno_dir, name)
            self.items.append({"clean": fp, "noisy": noisy_fp, "anno": anno_fp, "name": name})

    def __len__(self):
        return len(self.items)

    def _num_patches(self, shape):
        D, H, W = shape
        pD, pH, pW = PATCH_SIZE
        total = max(D - pD + 1, 1) * max(H - pH + 1, 1) * max(W - pW + 1, 1)
        if total == 1:
            return 1
        elif total < 8:
            return 2
        else:
            return 4

    def _random_crop(self, noisy, clean, anno, bnd):
        D, H, W = clean.shape
        pD, pH, pW = PATCH_SIZE
        z0 = np.random.randint(0, max(D - pD + 1, 1))
        y0 = np.random.randint(0, max(H - pH + 1, 1))
        x0 = np.random.randint(0, max(W - pW + 1, 1))
        return (
            noisy[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW],
            clean[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW],
            anno[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW],
            bnd[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW],
        )

    def _boundary_crop(self, noisy, clean, anno, bnd):
        coords = np.where(bnd > 0.5)
        if coords[0].size == 0:
            coords2 = np.where(anno > 0.5)
            if coords2[0].size == 0:
                return self._random_crop(noisy, clean, anno, bnd)
            j = np.random.randint(coords2[0].size)
            cz, cy, cx = coords2[0][j], coords2[1][j], coords2[2][j]
        else:
            j = np.random.randint(coords[0].size)
            cz, cy, cx = coords[0][j], coords[1][j], coords[2][j]

        pD, pH, pW = PATCH_SIZE
        D, H, W = clean.shape
        z0 = np.clip(cz - pD // 2, 0, max(D - pD, 0))
        y0 = np.clip(cy - pH // 2, 0, max(H - pH, 0))
        x0 = np.clip(cx - pW // 2, 0, max(W - pW, 0))

        return (
            noisy[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW],
            clean[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW],
            anno[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW],
            bnd[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW],
        )

    @staticmethod
    def _pad_to_patch_np(x: np.ndarray, patch_size, mode: str):
        pD, pH, pW = patch_size
        D, H, W = x.shape
        pd = max(pD - D, 0)
        ph = max(pH - H, 0)
        pw = max(pW - W, 0)
        if pd == 0 and ph == 0 and pw == 0:
            return x

        pad_width = ((0, pd), (0, ph), (0, pw))
        if mode == "edge":
            return np.pad(x, pad_width, mode="edge")
        elif mode == "constant0":
            return np.pad(x, pad_width, mode="constant", constant_values=0.0)
        else:
            raise ValueError(f"Unknown pad mode: {mode}")

    def __getitem__(self, idx):
        item = self.items[idx]

        clean = nii_as_float32(item["clean"])
        noisy = nii_as_float32(item["noisy"])
        anno = nii_as_float32(item["anno"])

        if clean.shape != noisy.shape or clean.shape != anno.shape:
            raise ValueError(
                f"Shape mismatch(train): {item['name']} | clean{clean.shape} noisy{noisy.shape} anno{anno.shape}")

        anno = (anno > 0.5).astype(np.float32)

        with torch.no_grad():
            a = torch.from_numpy(anno)[None, None]
            bnd = boundary_from_anno(a).squeeze(0).squeeze(0).numpy().astype(np.float32)

        n_patch = self._num_patches(clean.shape)
        noisy_list, clean_list, anno_list, bnd_list, valid_list = [], [], [], [], []

        def _make_valid_mask(x):
            return np.ones_like(x, dtype=np.float32)

        # Force first patch to be boundary crop
        n0, c0, a0, g0 = self._boundary_crop(noisy, clean, anno, bnd)
        v0 = _make_valid_mask(c0)

        n0 = self._pad_to_patch_np(n0, PATCH_SIZE, mode="edge")
        c0 = self._pad_to_patch_np(c0, PATCH_SIZE, mode="edge")
        a0 = self._pad_to_patch_np(a0, PATCH_SIZE, mode="constant0")
        g0 = self._pad_to_patch_np(g0, PATCH_SIZE, mode="constant0")
        v0 = self._pad_to_patch_np(v0, PATCH_SIZE, mode="constant0")

        noisy_list.append(torch.from_numpy(n0).unsqueeze(0))
        clean_list.append(torch.from_numpy(c0).unsqueeze(0))
        anno_list.append(torch.from_numpy(a0).unsqueeze(0))
        bnd_list.append(torch.from_numpy(g0).unsqueeze(0))
        valid_list.append(torch.from_numpy(v0).unsqueeze(0))

        remaining = n_patch - 1
        for _ in range(remaining):
            n1, c1, a1, g1 = self._random_crop(noisy, clean, anno, bnd)
            v1 = _make_valid_mask(c1)

            n1 = self._pad_to_patch_np(n1, PATCH_SIZE, mode="edge")
            c1 = self._pad_to_patch_np(c1, PATCH_SIZE, mode="edge")
            a1 = self._pad_to_patch_np(a1, PATCH_SIZE, mode="constant0")
            g1 = self._pad_to_patch_np(g1, PATCH_SIZE, mode="constant0")
            v1 = self._pad_to_patch_np(v1, PATCH_SIZE, mode="constant0")

            noisy_list.append(torch.from_numpy(n1).unsqueeze(0))
            clean_list.append(torch.from_numpy(c1).unsqueeze(0))
            anno_list.append(torch.from_numpy(a1).unsqueeze(0))
            bnd_list.append(torch.from_numpy(g1).unsqueeze(0))
            valid_list.append(torch.from_numpy(v1).unsqueeze(0))

        return {
            "noisy": torch.stack(noisy_list),
            "clean": torch.stack(clean_list),
            "anno": torch.stack(anno_list),
            "bnd": torch.stack(bnd_list),
            "valid": torch.stack(valid_list),
            "name": item["name"],
        }


class FullVolumeDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, anno_dir):
        self.items = []
        clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.nii")))
        if len(clean_files) == 0:
            raise FileNotFoundError(f"No .nii found in clean_dir: {clean_dir}")

        for cfp in clean_files:
            name = os.path.basename(cfp)
            nfp = os.path.join(noisy_dir, name)
            afp = os.path.join(anno_dir, name)
            self.items.append({"clean": cfp, "noisy": nfp, "anno": afp, "name": name})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        gt_nii = nib.load(item["clean"])
        noisy_nii = nib.load(item["noisy"])
        anno_nii = nib.load(item["anno"])

        gt = np.asarray(gt_nii.dataobj).astype(np.float32, copy=False)
        noisy = np.asarray(noisy_nii.dataobj).astype(np.float32, copy=False)
        anno = np.asarray(anno_nii.dataobj).astype(np.float32, copy=False)

        # Strict shape consistency check for validation / test.
        # Affine consistency alone is not sufficient.
        if gt.shape != noisy.shape or gt.shape != anno.shape:
            raise ValueError(
                f"Shape mismatch(val/test): {item['name']} | "
                f"gt{gt.shape} noisy{noisy.shape} anno{anno.shape}"
            )

        def _affine_close(A, B, tol=1e-2):
            return np.allclose(A, B, atol=tol, rtol=0)

        aff_gt = gt_nii.affine
        aff_noisy = noisy_nii.affine
        aff_anno = anno_nii.affine

        if (not _affine_close(aff_gt, aff_noisy)) or (not _affine_close(aff_gt, aff_anno)):
            raise ValueError(
                f"Affine mismatch(val/test): {item['name']} | "
                f"gt_affine != noisy_affine or anno_affine"
            )

        ref_affine = aff_gt
        anno = (anno > 0.5).astype(np.float32)

        return {
            "gt": torch.from_numpy(gt).unsqueeze(0),
            "noisy": torch.from_numpy(noisy).unsqueeze(0),
            "anno": torch.from_numpy(anno).unsqueeze(0),
            "affine": ref_affine,
            "name": item["name"],
        }


def train_patch_collate_fn(batch):
    out = {}
    tensor_keys = ["noisy", "clean", "anno", "bnd", "valid"]
    for k in tensor_keys:
        out[k] = torch.cat([item[k] for item in batch], dim=0)
    out["name"] = [item["name"] for item in batch]
    return out


def full_volume_list_collate_fn(batch):
    return batch


def binary_erode3d(a: torch.Tensor, k: int = 3) -> torch.Tensor:
    p = k // 2
    inv = 1.0 - a
    inv = F.pad(inv, (p, p, p, p, p, p), mode="constant", value=1.0)
    inv_max = F.max_pool3d(inv, kernel_size=k, stride=1, padding=0)
    return 1.0 - inv_max


def boundary_from_anno(anno_bin: torch.Tensor) -> torch.Tensor:
    eroded = binary_erode3d(anno_bin, k=3)
    boundary = torch.clamp(anno_bin - eroded, 0.0, 1.0)
    return boundary


def soft_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return ((2.0 * inter + eps) / (denom + eps)).mean()


def masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, data_range: float = 1.0,
                eps: float = 1e-12) -> torch.Tensor:
    B = pred.shape[0]
    pred_f, gt_f, mask_f = pred.reshape(B, -1), gt.reshape(B, -1), mask.reshape(B, -1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    mse = (((pred_f - gt_f) ** 2 * mask_f).sum(dim=1) / denom).clamp_min(eps)
    return (10.0 * torch.log10((data_range ** 2) / mse)).mean()


# ============================================================
# Conditional 3D PatchGAN Discriminator
# ============================================================
class PatchDiscriminator3D(nn.Module):
    def __init__(self, cond_ch: int = 1, target_ch: int = 1, base_ch: int = 32, n_layers: int = 3, max_ch: int = 256,
                 norm: str = "instance"):
        super().__init__()
        in_ch = int(cond_ch) + int(target_ch)

        if norm == "instance":
            Norm = lambda c: nn.InstanceNorm3d(c, affine=True)
        elif norm == "batch":
            Norm = lambda c: nn.BatchNorm3d(c)
        else:
            Norm = None

        def conv_block(cin, cout, stride, use_norm=True):
            bias = not (use_norm and Norm is not None)
            layers = [nn.Conv3d(cin, cout, kernel_size=4, stride=stride, padding=1, bias=bias)]
            if use_norm and Norm is not None: layers.append(Norm(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers += conv_block(in_ch, base_ch, stride=2, use_norm=False)

        ch = base_ch
        for _ in range(1, n_layers):
            ch_next = min(ch * 2, max_ch)
            layers += conv_block(ch, ch_next, stride=2, use_norm=True)
            ch = ch_next

        ch_next = min(ch * 2, max_ch)
        layers += conv_block(ch, ch_next, stride=1, use_norm=True)
        layers.append(nn.Conv3d(ch_next, 1, kernel_size=4, stride=1, padding=1, bias=True))

        self.net = nn.Sequential(*layers)

    def forward(self, cond, target):
        return self.net(torch.cat([cond, target], dim=1))


# ============================================================
# U-Net Blocks
# ============================================================
def _make_gn(ch: int, ng: int = 8) -> nn.GroupNorm:
    g = min(ng, ch)
    while ch % g != 0 and g > 1: g -= 1
    return nn.GroupNorm(g, ch)


class GNAct(nn.Module):
    def __init__(self, ch: int, ng: int = 8):
        super().__init__()
        self.gn = _make_gn(ch, ng=ng)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x): return self.act(self.gn(x))


class ResBlock3D(nn.Module):
    def __init__(self, ch: int, ng: int = 8, drop: float = 0.0):
        super().__init__()
        self.p1 = GNAct(ch, ng)
        self.c1 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.p2 = GNAct(ch, ng)
        self.c2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.drop = nn.Dropout3d(drop) if (drop and drop > 0) else nn.Identity()

    def forward(self, x):
        h = self.c2(self.p2(self.drop(self.c1(self.p1(x)))))
        return x + h


class FiLMResidual3D(nn.Module):
    def __init__(self, target_ch: int, cond_ch: int = None, ng: int = 8, hidden_ratio: float = 0.5):
        super().__init__()
        cond_ch = target_ch if cond_ch is None else cond_ch
        hidden = max(16, int(cond_ch * hidden_ratio))

        self.norm = _make_gn(target_ch, ng=ng)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(cond_ch, hidden)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(hidden, target_ch * 2)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x_tgt, x_cond):
        B, C = x_tgt.shape[:2]
        s = self.pool(x_cond).flatten(1)
        gamma, beta = self.fc2(self.act(self.fc1(s))).chunk(2, dim=1)
        return gamma.view(B, C, 1, 1, 1) * self.norm(x_tgt) + beta.view(B, C, 1, 1, 1)


class SharedStem3D(nn.Module):
    def __init__(self, base_ch=32, ng=8, drop=0.0):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(1, base_ch, 3, padding=1, bias=False), GNAct(base_ch, ng),
                                   ResBlock3D(base_ch, ng, drop))

    def forward(self, x):
        if self.training and torch.is_grad_enabled(): return checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)


class BranchEncoderL23(nn.Module):
    def __init__(self, base_ch=32, ng=8, drop=0.0):
        super().__init__()
        self.pool1 = nn.AvgPool3d(2)
        self.enc2 = nn.Sequential(nn.Conv3d(base_ch, base_ch * 2, 3, padding=1, bias=False), GNAct(base_ch * 2, ng),
                                  ResBlock3D(base_ch * 2, ng, drop))
        self.pool2 = nn.AvgPool3d(2)
        self.enc3 = nn.Sequential(nn.Conv3d(base_ch * 2, base_ch * 4, 3, padding=1, bias=False), GNAct(base_ch * 4, ng),
                                  ResBlock3D(base_ch * 4, ng, drop))

    def forward_stage2(self, f1):
        x = self.pool1(f1)
        if self.training and torch.is_grad_enabled(): return checkpoint(self.enc2, x, use_reentrant=False)
        return self.enc2(x)

    def forward_stage3(self, f2):
        x = self.pool2(f2)
        if self.training and torch.is_grad_enabled(): return checkpoint(self.enc3, x, use_reentrant=False)
        return self.enc3(x)


class BranchDecoder3D(nn.Module):
    def __init__(self, base_ch=32, out_ch=1, ng=8, drop=0.0):
        super().__init__()
        self.up2_proj = nn.Sequential(nn.Conv3d(base_ch * 4, base_ch * 2, 1, bias=False), _make_gn(base_ch * 2, ng=ng))
        self.dec2 = nn.Sequential(nn.Conv3d(base_ch * 4, base_ch * 2, 3, padding=1, bias=False), GNAct(base_ch * 2, ng),
                                  ResBlock3D(base_ch * 2, ng, drop))
        self.up1_proj = nn.Sequential(nn.Conv3d(base_ch * 2, base_ch, 1, bias=False), _make_gn(base_ch, ng=ng))
        self.dec1 = nn.Sequential(nn.Conv3d(base_ch * 2, base_ch, 3, padding=1, bias=False), GNAct(base_ch, ng),
                                  ResBlock3D(base_ch, ng, drop))
        self.out = nn.Conv3d(base_ch, out_ch, 1)

    def forward(self, f1_shared, f2_branch, f3_branch):
        ckpt = (lambda fn, *args: checkpoint(fn, *args, use_reentrant=False)) if (
                    self.training and torch.is_grad_enabled()) else (lambda fn, *args: fn(*args))
        u2 = ckpt(self.up2_proj,
                  F.interpolate(f3_branch, size=f2_branch.shape[2:], mode="trilinear", align_corners=False))
        d2 = ckpt(self.dec2, torch.cat([u2, f2_branch], dim=1))
        u1 = ckpt(self.up1_proj, F.interpolate(d2, size=f1_shared.shape[2:], mode="trilinear", align_corners=False))
        d1 = ckpt(self.dec1, torch.cat([u1, f1_shared], dim=1))
        return self.out(d1)


# ============================================================
# Dynamic FiLM Weights Network (MAD-Based)
# ============================================================
class NBGLGenerator3D(nn.Module):
    """
    Shared-stem dual-encoder dual-decoder model with parameter-free adaptive FiLM.

    Dynamic interaction rule
    ------------------------
    - The true noise level is NOT used as an external condition.
    - Interaction strength is estimated only from the current input patch using
      MAD-based noise estimation.
    - Estimated sigma is mapped in log-sigma space to avoid immediate saturation.
    - Low estimated noise -> interaction weights stay near lambda_min.
    - High estimated noise -> interaction weights smoothly approach lambda_max.

    Returned dynamic weights
    ------------------------
    w_d2_from_e2 : edge -> denoising interaction at stage 2
    w_e2_from_d2 : denoising -> edge interaction at stage 2
    w_d3_from_e3 : edge -> denoising interaction at stage 3
    w_e3_from_d3 : denoising -> edge interaction at stage 3
    """

    def __init__(self,
                 base_ch: int = 32,
                 ng: int = 8,
                 drop: float = 0.0,
                 lambda_min: float = 0.5,
                 lambda_max: float = 1.5,
                 noise_decay_gamma: float = 1.2,
                 sigma_floor: float = 0.003,
                 sigma_ceiling: float = 0.12,
                 stage3_edge_scale: float = 0.2):
        super().__init__()

        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.gamma = float(noise_decay_gamma)

        # Reference range for patch-wise MAD-estimated sigma.
        # sigma <= sigma_floor   -> weights near lambda_min
        # sigma >= sigma_ceiling -> weights near lambda_max
        self.sigma_floor = float(sigma_floor)
        self.sigma_ceiling = float(sigma_ceiling)

        # Preserve the original asymmetric stage-3 prior:
        # edge guidance to denoising remains strong,
        # reverse guidance to edge branch stays weaker.
        self.stage3_edge_scale = float(stage3_edge_scale)

        self.shared_stem = SharedStem3D(base_ch=base_ch, ng=ng, drop=drop)
        self.den_enc = BranchEncoderL23(base_ch=base_ch, ng=ng, drop=drop)
        self.edge_enc = BranchEncoderL23(base_ch=base_ch, ng=ng, drop=drop)

        self.film_d2_from_e2 = FiLMResidual3D(base_ch * 2, base_ch * 2, ng=ng)
        self.film_e2_from_d2 = FiLMResidual3D(base_ch * 2, base_ch * 2, ng=ng)
        self.film_d3_from_e3 = FiLMResidual3D(base_ch * 4, base_ch * 4, ng=ng)
        self.film_e3_from_d3 = FiLMResidual3D(base_ch * 4, base_ch * 4, ng=ng)

        self.den_dec = BranchDecoder3D(base_ch=base_ch, out_ch=1, ng=ng, drop=0.05)
        self.edge_dec = BranchDecoder3D(base_ch=base_ch, out_ch=1, ng=ng, drop=0.0)

    def _get_dynamic_weights(self, x):
        """
        Compute dynamic FiLM interaction weights from the current input patch only.

        Parameters
        ----------
        x : torch.Tensor
            Input patch, shape [B, 1, D, H, W].

        Returns
        -------
        w_d2_from_e2, w_e2_from_d2, w_d3_from_e3, w_e3_from_d3, sigma
            sigma is the raw MAD-based noise estimate (not artificially scaled).
        """
        B = x.shape[0]

        # Parameter-free patch-wise noise estimation from current input patch.
        sigma = estimate_noise_mad_3d(x).clamp_min(1e-6)  # [B]

        # Map sigma in log space to avoid immediate saturation.
        log_sigma = torch.log(sigma)

        sigma_floor_t = torch.as_tensor(self.sigma_floor, dtype=sigma.dtype, device=sigma.device)
        sigma_ceiling_t = torch.as_tensor(self.sigma_ceiling, dtype=sigma.dtype, device=sigma.device)

        log_floor = torch.log(sigma_floor_t.clamp_min(1e-6))
        log_ceiling = torch.log(sigma_ceiling_t.clamp_min(1e-6))

        denom = (log_ceiling - log_floor).clamp_min(1e-6)
        sigma_norm = ((log_sigma - log_floor) / denom).clamp(0.0, 1.0)

        # Monotonic non-linear transition factor in [0, 1].
        transition_factor = sigma_norm.pow(self.gamma)

        # Stage-2 symmetric interaction strength.
        w_base = self.lambda_min + (self.lambda_max - self.lambda_min) * transition_factor

        w_d2_from_e2 = w_base.view(B, 1, 1, 1, 1)
        w_e2_from_d2 = w_base.view(B, 1, 1, 1, 1)

        # Stage-3 keeps the original asymmetric reverse-guidance design.
        w_d3_from_e3 = w_base.view(B, 1, 1, 1, 1)
        w_e3_from_d3 = (w_base * self.stage3_edge_scale).view(B, 1, 1, 1, 1)

        return w_d2_from_e2, w_e2_from_d2, w_d3_from_e3, w_e3_from_d3, sigma

    def _encode(self, x):
        w_d2_from_e2, w_e2_from_d2, w_d3_from_e3, w_e3_from_d3, _ = self._get_dynamic_weights(x)

        f1 = self.shared_stem(x)
        fd2_base = self.den_enc.forward_stage2(f1)
        fe2_base = self.edge_enc.forward_stage2(f1)

        fd2 = fd2_base + w_d2_from_e2 * self.film_d2_from_e2(fd2_base, fe2_base)
        fe2 = fe2_base + w_e2_from_d2 * self.film_e2_from_d2(fe2_base, fd2_base)

        fd3_base = self.den_enc.forward_stage3(fd2)
        fe3_base = self.edge_enc.forward_stage3(fe2)

        fd3 = fd3_base + w_d3_from_e3 * self.film_d3_from_e3(fd3_base, fe3_base)
        fe3 = fe3_base + w_e3_from_d3 * self.film_e3_from_d3(fe3_base, fd3_base)

        return f1, fd2, fd3, fe2, fe3

    def forward(self, x):
        f1, fd2, fd3, fe2, fe3 = self._encode(x)
        den = x + self.den_dec(f1, fd2, fd3)
        edge_logits = self.edge_dec(f1, fe2, fe3)
        return {"den": den, "edge_logits": edge_logits, "edge_prob": torch.sigmoid(edge_logits)}

    def forward_train(self, x):
        return self.forward(x)

# ============================================================
# Sliding forward (1D along any axis) + multi-axis average
# ============================================================
@torch.no_grad()
def _forward_sliding_1d(model, x, dim: int, chunk=64, overlap=16,
                        return_edge: bool = False,
                        return_weight_stats: bool = False):
    """
    Sliding inference along one spatial axis.

    If return_weight_stats=True, this function also accumulates the actual
    chunk-wise dynamic FiLM weights used during sliding inference and returns
    their weighted average over all chunks on this axis.
    """
    model.eval()
    B, C, D, H, W = x.shape
    size = x.shape[dim]

    chunk = int(min(chunk, size))
    overlap = int(min(max(overlap, 0), chunk - 1))
    step = max(chunk - overlap, 1)

    if size <= chunk:
        starts = [0]
    else:
        starts = list(range(0, size - chunk + 1, step))
        if starts[-1] != size - chunk:
            starts.append(size - chunk)

    out_den = torch.zeros_like(x)
    wgt = torch.zeros_like(x)

    if return_edge:
        out_edge = torch.zeros_like(x)
        edge_wgt = torch.zeros_like(x)
    else:
        out_edge, edge_wgt = None, None

    if return_weight_stats:
        stat_names = [
            "estimated_noise_mad",
            "w_d2_from_e2",
            "w_e2_from_d2",
            "w_d3_from_e3",
            "w_e3_from_d3",
        ]
        stat_sum = {k: torch.zeros(B, device=x.device, dtype=torch.float32) for k in stat_names}
        stat_den = torch.zeros(B, device=x.device, dtype=torch.float32)
    else:
        stat_sum, stat_den = None, None

    win_1d = torch.hann_window(chunk, device=x.device, dtype=x.dtype).clamp_min(1e-6)
    if dim == 2:
        win_full = win_1d.view(1, 1, chunk, 1, 1)
    elif dim == 3:
        win_full = win_1d.view(1, 1, 1, chunk, 1)
    else:
        win_full = win_1d.view(1, 1, 1, 1, chunk)

    amp_on = (x.device.type == "cuda")

    for s0 in starts:
        s1 = s0 + chunk
        sl = [slice(None)] * 5
        sl[dim] = slice(s0, s1)
        xp = x[tuple(sl)]

        with autocast(enabled=amp_on):
            out = model(xp)

        den = out["den"].clamp(0, 1)
        out_den[tuple(sl)] += den * win_full
        wgt[tuple(sl)] += win_full

        if return_edge:
            edge = out["edge_prob"].clamp(0, 1)
            out_edge[tuple(sl)] += edge * win_full
            edge_wgt[tuple(sl)] += win_full

        if return_weight_stats:
            # Collect the actual dynamic weights used for this chunk.
            w_d2, w_e2, w_d3, w_e3, sigma_est = model._get_dynamic_weights(xp)
            chunk_mass = win_full.sum().detach().float()

            stat_sum["estimated_noise_mad"] += sigma_est.detach().float() * chunk_mass
            stat_sum["w_d2_from_e2"] += w_d2.view(B, -1).mean(dim=1).detach().float() * chunk_mass
            stat_sum["w_e2_from_d2"] += w_e2.view(B, -1).mean(dim=1).detach().float() * chunk_mass
            stat_sum["w_d3_from_e3"] += w_d3.view(B, -1).mean(dim=1).detach().float() * chunk_mass
            stat_sum["w_e3_from_d3"] += w_e3.view(B, -1).mean(dim=1).detach().float() * chunk_mass
            stat_den += torch.ones(B, device=x.device, dtype=torch.float32) * chunk_mass

    den = out_den / wgt.clamp_min(1e-6)

    stats = None
    if return_weight_stats:
        stats = {k: v / stat_den.clamp_min(1e-6) for k, v in stat_sum.items()}

    if return_edge:
        edge = out_edge / edge_wgt.clamp_min(1e-6)
        if return_weight_stats:
            return den, edge, stats
        return den, edge

    if return_weight_stats:
        return den, stats
    return den

@torch.no_grad()
def forward_sliding(model, x, chunk=64, overlap=16,
                    return_edge: bool = False,
                    return_weight_stats: bool = False,
                    axes=(4,)):
    """
    Multi-axis sliding inference.

    If return_weight_stats=True, returns the axis-averaged dynamic FiLM weights
    that were actually used during sliding inference.
    """
    axes = tuple(axes) if axes is not None else (4,)

    if len(axes) == 1:
        if return_edge and return_weight_stats:
            return _forward_sliding_1d(
                model, x, dim=axes[0], chunk=chunk, overlap=overlap,
                return_edge=True, return_weight_stats=True
            )
        elif return_edge:
            return _forward_sliding_1d(
                model, x, dim=axes[0], chunk=chunk, overlap=overlap,
                return_edge=True, return_weight_stats=False
            )
        elif return_weight_stats:
            return _forward_sliding_1d(
                model, x, dim=axes[0], chunk=chunk, overlap=overlap,
                return_edge=False, return_weight_stats=True
            )
        else:
            return _forward_sliding_1d(
                model, x, dim=axes[0], chunk=chunk, overlap=overlap,
                return_edge=False, return_weight_stats=False
            )

    den_sum = torch.zeros_like(x)
    edge_sum = torch.zeros_like(x) if return_edge else None
    stats_sum = None

    for dim in axes:
        if return_edge and return_weight_stats:
            den_i, edge_i, stats_i = _forward_sliding_1d(
                model, x, dim=dim, chunk=chunk, overlap=overlap,
                return_edge=True, return_weight_stats=True
            )
            den_sum += den_i
            edge_sum += edge_i
        elif return_edge:
            den_i, edge_i = _forward_sliding_1d(
                model, x, dim=dim, chunk=chunk, overlap=overlap,
                return_edge=True, return_weight_stats=False
            )
            den_sum += den_i
            edge_sum += edge_i
            stats_i = None
        elif return_weight_stats:
            den_i, stats_i = _forward_sliding_1d(
                model, x, dim=dim, chunk=chunk, overlap=overlap,
                return_edge=False, return_weight_stats=True
            )
            den_sum += den_i
        else:
            den_i = _forward_sliding_1d(
                model, x, dim=dim, chunk=chunk, overlap=overlap,
                return_edge=False, return_weight_stats=False
            )
            den_sum += den_i
            stats_i = None

        if return_weight_stats:
            if stats_sum is None:
                stats_sum = {k: v.clone() for k, v in stats_i.items()}
            else:
                for k in stats_sum.keys():
                    stats_sum[k] += stats_i[k]

    den = (den_sum / float(len(axes))).clamp(0, 1)

    stats = None
    if return_weight_stats:
        stats = {k: v / float(len(axes)) for k, v in stats_sum.items()}

    if return_edge:
        edge = (edge_sum / float(len(axes))).clamp(0, 1)
        if return_weight_stats:
            return den, edge, stats
        return den, edge

    if return_weight_stats:
        return den, stats
    return den




# ============================================================
# Datasets Aggregation for Blind Denoising
# ============================================================
train_datasets = []
val_datasets = []
test_loaders = {}

for level in TARGET_NOISE_LEVELS:
    d = os.path.join(DATA_ROOT, f"noise_{level}_norm01")
    if not os.path.exists(d):
        print(f"Warning: Directory not found for noise level {level}: {d}")
        continue

    # Build Train Datasets to concatenate
    ds_train = DenoiseDataset3D(
        clean_dir=os.path.join(d, "train", "clean"),
        noisy_dir=os.path.join(d, "train", "noisy"),
        anno_dir=os.path.join(d, "train", "annotations")
    )
    train_datasets.append(ds_train)

    # Build validation datasets to concatenate into one mixed validation loader
    ds_val = FullVolumeDataset(
        clean_dir=os.path.join(d, "valid", "clean"),
        noisy_dir=os.path.join(d, "valid", "noisy"),
        anno_dir=os.path.join(d, "valid", "annotations")
    )
    val_datasets.append(ds_val)

    # Build individual Test Loaders
    ds_test = FullVolumeDataset(
        clean_dir=os.path.join(d, "test", "clean"),
        noisy_dir=os.path.join(d, "test", "noisy"),
        anno_dir=os.path.join(d, "test", "annotations")
    )
    test_loaders[level] = DataLoader(ds_test, batch_size=TEST_BS, shuffle=False, num_workers=0, collate_fn=full_volume_list_collate_fn)

if train_datasets:
    mixed_train_ds = ConcatDataset(train_datasets)
    train_loader = DataLoader(mixed_train_ds, batch_size=TRAIN_BS, shuffle=True, num_workers=NUM_WORKERS, collate_fn=train_patch_collate_fn)
else:
    train_loader = []
    print("Warning: No valid training datasets found.")

if val_datasets:
    mixed_val_ds = ConcatDataset(val_datasets)
    val_loader = DataLoader(mixed_val_ds, batch_size=VAL_BS, shuffle=False, num_workers=0, collate_fn=full_volume_list_collate_fn)
else:
    val_loader = []
    print("Warning: No valid validation datasets found.")


# ============================================================
# Instantiate Models
# ============================================================
generator = NBGLGenerator3D(
    base_ch=32,
    ng=8,
    drop=0.0,
    lambda_min=0.5,
    lambda_max=1.5,
    noise_decay_gamma=2.0
).to(device)

discriminator = PatchDiscriminator3D(
    cond_ch=1, target_ch=1, base_ch=32, n_layers=3, max_ch=256, norm="instance"
).to(device)

opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

scaler_g = torch.cuda.amp.GradScaler(enabled=use_amp)
scaler_d = torch.cuda.amp.GradScaler(enabled=use_amp)

# ============================================================
# Resume Training
# ============================================================
start_epoch = 1
best_psnr = -1e9
best_bdice = -1e9

history_csv = os.path.join(OUT_ROOT, "train_history.csv")
history = {k: [] for k in
           ["epoch", "lambda_a", "loss_rec", "loss_ssim", "loss_edge",
            "loss_adv_g", "loss_g", "loss_d", "val_psnr", "val_bdice"]}

if RUN_TRAIN and RESUME_TRAIN and os.path.exists(RESUME_PATH):
    print(f"Resuming training from: {RESUME_PATH}")
    ckpt = torch.load(RESUME_PATH, map_location=device)

    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])

    if "scaler_g" in ckpt:
        scaler_g.load_state_dict(ckpt["scaler_g"])
    if "scaler_d" in ckpt:
        scaler_d.load_state_dict(ckpt["scaler_d"])

    start_epoch = int(ckpt["epoch"]) + 1

    # 优先从 best_ckpt 恢复最优指标，避免 best 记录丢失
    if os.path.exists(BEST_CKPT):
        best_ckpt = torch.load(BEST_CKPT, map_location=device)
        best_psnr = float(best_ckpt.get("best_val_psnr", -1e9))
        best_bdice = float(best_ckpt.get("best_val_bdice", -1e9))
    else:
        val_metrics = ckpt.get("val_metrics", {})
        best_psnr = float(val_metrics.get("val_psnr", -1e9)) if np.isfinite(val_metrics.get("val_psnr", np.nan)) else -1e9
        best_bdice = float(val_metrics.get("val_bdice", -1e9)) if np.isfinite(val_metrics.get("val_bdice", np.nan)) else -1e9

    # 继续历史曲线
    if os.path.exists(history_csv):
        old_hist = pd.read_csv(history_csv)
        for k in history.keys():
            if k in old_hist.columns:
                history[k] = old_hist[k].tolist()

    print(f"Resume start_epoch = {start_epoch}")
    print(f"Loaded best_psnr = {best_psnr:.6f}, best_bdice = {best_bdice:.6f}")
else:
    print("Training from scratch.")
if RUN_TRAIN and start_epoch > MAX_EPOCHS:
    raise ValueError(
        f"start_epoch={start_epoch} > MAX_EPOCHS={MAX_EPOCHS}. "
        f"If you want to continue training, increase MAX_EPOCHS."
) 

# ============================================================
# Loss & Helpers
# ============================================================
bce_logits = nn.BCEWithLogitsLoss(reduction="none")


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def masked_charbonnier_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3,
                            reduce: str = "mean") -> torch.Tensor:
    diff = charbonnier(pred - gt, eps=eps) * mask
    denom = mask.sum().clamp_min(1.0)
    loss = diff.sum() / denom
    return loss if reduce == "mean" else loss


def bbox_from_mask_3d(mask, min_size: int = 1):
    if isinstance(mask, np.ndarray):
        idx = np.where(mask)
        if idx[0].size == 0: return None
        d0, d1 = int(idx[0].min()), int(idx[0].max())
        h0, h1 = int(idx[1].min()), int(idx[1].max())
        w0, w1 = int(idx[2].min()), int(idx[2].max())
    else:
        idx = torch.nonzero(mask, as_tuple=False)
        if idx.numel() == 0: return None
        d0, d1 = int(idx[:, 0].min().item()), int(idx[:, 0].max().item())
        h0, h1 = int(idx[:, 1].min().item()), int(idx[:, 1].max().item())
        w0, w1 = int(idx[:, 2].min().item()), int(idx[:, 2].max().item())

    if min((d1 - d0 + 1), (h1 - h0 + 1), (w1 - w0 + 1)) < int(min_size): return None
    return d0, d1, h0, h1, w0, w1


def crop_by_bbox_3d(vol, bb):
    if bb is None: return None
    d0, d1, h0, h1, w0, w1 = bb
    return vol[d0:d1 + 1, h0:h1 + 1, w0:w1 + 1]


def ssim_loss_bbox(pred: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0, base_k: int = 7,
                   eps_mask: float = 1e-6) -> torch.Tensor:
    B = pred.shape[0]
    losses = []
    for b in range(B):
        m = (gt[b, 0] > eps_mask)
        bb = bbox_from_mask_3d(m)
        if bb is None: continue

        d0, d1, h0, h1, w0, w1 = bb
        pred_c = pred[b:b + 1, :, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1]
        gt_c = gt[b:b + 1, :, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1]
        _, _, Dz, Hy, Wx = pred_c.shape
        kz, ky, kx = odd_kernel_leq(base_k, Dz), odd_kernel_leq(base_k, Hy), odd_kernel_leq(base_k, Wx)

        if min(kz, ky, kx) < 3: continue
        with autocast(enabled=False):
            ssim = tm_ssim(pred_c.float(), gt_c.float(), data_range=float(data_range), kernel_size=(kz, ky, kx))
        losses.append(1.0 - ssim)

    return torch.stack(losses).mean() if len(losses) > 0 else pred.new_tensor(0.0)


def soft_dice_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None,
                     eps: float = 1e-6) -> torch.Tensor:
    if mask is not None: pred, target = pred * mask, target * mask
    B = pred.shape[0]
    pred_f, tar_f = pred.reshape(B, -1), target.reshape(B, -1)
    inter = (pred_f * tar_f).sum(dim=1)
    denom = pred_f.sum(dim=1) + tar_f.sum(dim=1)
    return ((2.0 * inter + eps) / (denom + eps)).mean()


def edge_loss_bce_dice(edge_logits: torch.Tensor, edge_gt: torch.Tensor, roi_mask: torch.Tensor, band_ks: int = 5,
                       eps: float = 1e-6):
    with autocast(enabled=False):
        edge_band = F.max_pool3d(edge_gt, kernel_size=band_ks, stride=1, padding=band_ks // 2)
        wmap = (edge_band * roi_mask).detach()

        logits32, gt32 = edge_logits.float(), edge_gt.float()
        prob32 = torch.sigmoid(logits32)
        wmap32, roi32 = wmap.float(), roi_mask.float()

        raw = bce_logits(logits32, gt32)
        denom = wmap32.sum().clamp_min(1.0)
        loss_bce = (raw * wmap32).sum() / denom

        loss_dice = 1.0 - soft_dice_masked(prob32, gt32, mask=roi32, eps=eps)
        loss_total = loss_bce + loss_dice

    return loss_bce, loss_dice, loss_total


def lsgan_d_loss(d_real, d_fake):
    return 0.5 * ((d_real - 1) ** 2).mean() + 0.5 * (d_fake ** 2).mean()


def lsgan_g_loss(d_fake):
    return 0.5 * ((d_fake - 1) ** 2).mean()


def sample_patch_3d(cond: torch.Tensor, fake: torch.Tensor, real: torch.Tensor, roi: torch.Tensor, patch: int = 64):
    B, _, D, H, W = cond.shape
    if (D < patch) or (H < patch) or (W < patch): return cond, fake, real

    cond_list, fake_list, real_list = [], [], []
    for b in range(B):
        m = (roi[b, 0] > 0)
        if m.any():
            idx = torch.nonzero(m, as_tuple=False)
            j = torch.randint(0, idx.shape[0], (1,), device=cond.device).item()
            cz, cy, cx = [int(v.item()) for v in idx[j]]
            z0, y0, x0 = max(0, min(cz - patch // 2, D - patch)), max(0, min(cy - patch // 2, H - patch)), max(0,
                                                                                                               min(cx - patch // 2,
                                                                                                                   W - patch))
        else:
            z0, y0, x0 = torch.randint(0, D - patch + 1, (1,)).item(), torch.randint(0, H - patch + 1,
                                                                                     (1,)).item(), torch.randint(0,
                                                                                                                 W - patch + 1,
                                                                                                                 (1,)).item()

        cond_list.append(cond[b:b + 1, :, z0:z0 + patch, y0:y0 + patch, x0:x0 + patch])
        fake_list.append(fake[b:b + 1, :, z0:z0 + patch, y0:y0 + patch, x0:x0 + patch])
        real_list.append(real[b:b + 1, :, z0:z0 + patch, y0:y0 + patch, x0:x0 + patch])

    return torch.cat(cond_list, dim=0), torch.cat(fake_list, dim=0), torch.cat(real_list, dim=0)


# ============================================================
# Validation Evaluator
# ============================================================
@torch.no_grad()
def validate_full_sliding(generator, val_loader, device, SLIDE_CHUNK, SLIDE_OVERLAP, EPS=1e-6):
    generator.eval()
    sum_psnr, n_psnr = 0.0, 0
    sum_bdice, n_bdice = 0.0, 0

    for batch_list in val_loader:
        for b in batch_list:
            gt, noisy, anno = b["gt"].unsqueeze(0).to(device).float(), b["noisy"].unsqueeze(0).to(device).float(), b[
                "anno"].unsqueeze(0).to(device).float()
            den, edge = forward_sliding(generator, noisy, chunk=SLIDE_CHUNK, overlap=SLIDE_OVERLAP, return_edge=True,
                                        axes=(2, 3, 4))
            den, edge = den.clamp(0, 1), edge.clamp(0, 1)

            mask = (gt > EPS).float()
            if mask.sum().item() >= 1.0:
                sum_psnr += float(masked_psnr(den, gt, mask, data_range=1.0).item())
                n_psnr += 1

            boundary = boundary_from_anno((anno > 0.5).float())
            if boundary.sum().item() >= 1.0:
                bb = bbox_from_mask_3d((anno[0, 0] > 0.5))
                if bb is not None:
                    d0, d1, h0, h1, w0, w1 = bb
                    bnd_c, edge_c, roi_c = boundary[:, :, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1], edge[
                        :, :, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1], (
                                anno[:, :, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1] > 0.5).float()
                    sum_bdice += float(soft_dice(edge_c * roi_c, bnd_c * roi_c).item())
                    n_bdice += 1

    return {
        "val_psnr": float(sum_psnr / n_psnr) if n_psnr > 0 else float("nan"),
        "val_bdice": float(sum_bdice / n_bdice) if n_bdice > 0 else float("nan"),
    }


# ============================================================
# Train Monitoring Plotting
# ============================================================
def _rolling_mean(y, win):
    y = np.asarray(y, dtype=np.float64)
    if (win is None) or (win <= 1): return y
    return pd.Series(y).rolling(win, min_periods=1).mean().to_numpy()


def plot_and_save_history(history: dict, out_dir: str, smooth_window: int = 0):
    df = pd.DataFrame(history)
    if len(df) < 2: return
    x = df["epoch"].to_numpy()

    def save_plot(fname, ys, labels, title, yscale=None):
        plt.figure(figsize=(8, 5))
        for y, lab in zip(ys, labels):
            y2 = _rolling_mean(np.asarray(y, dtype=np.float64), smooth_window)
            if yscale == "log": y2 = np.clip(y2, 1e-12, None)
            plt.plot(x, y2, label=lab)
        plt.title(title);
        plt.xlabel("epoch");
        plt.grid(True, alpha=0.3);
        plt.legend();
        plt.tight_layout()
        if yscale is not None: plt.yscale(yscale)
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()

    save_plot("fig_loss_total.png", [df["loss_g"].to_numpy(), df["loss_d"].to_numpy()], ["loss_g", "loss_d"],
              "Total Losses", yscale="log")
    save_plot("fig_loss_components.png",
              [df["loss_rec"].to_numpy(), df["loss_ssim"].to_numpy(), df["loss_edge"].to_numpy(),
               df["loss_adv_g"].to_numpy()], ["rec", "ssim", "edge", "adv_g"], "Generator Loss Components")
    save_plot("fig_val_psnr.png", [df["val_psnr"].to_numpy()], ["val_psnr_mean"], "Validation Mean PSNR")
    save_plot("fig_val_bdice.png", [df["val_bdice"].to_numpy()], ["val_bdice_mean"], "Validation Mean Boundary Dice")


# ============================================================
# Train Loop
# ============================================================
if RUN_TRAIN:
    delta_psnr = 0.01

    lambda_r, lambda_g, lambda_e = 1.0, 4.0, 0.5

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        grad_debug_printed_this_epoch = False
        generator.train()
        discriminator.train()

        lambda_a = 0.0 if epoch < 10 else (0.01 if epoch < 30 else 0.02)
        ep = {k: 0.0 for k in ["loss_rec", "loss_ssim", "loss_edge", "loss_adv_g", "loss_g", "loss_d"]}
        n_iter = 0

        for b in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}"):
            noisy_all, clean_all, anno_all, valid_all, edge_gt_all = b["noisy"].to(device).float(), b["clean"].to(
                device).float(), b["anno"].to(device).float(), b["valid"].to(device).float(), b["bnd"].to(
                device).float()
            P = noisy_all.shape[0]

            for i0 in range(0, P, MICRO_BATCH_SIZE):
                i1 = min(i0 + MICRO_BATCH_SIZE, P)
                noisy, clean, anno, valid, edge_gt = noisy_all[i0:i1], clean_all[i0:i1], anno_all[i0:i1], valid_all[
                    i0:i1], edge_gt_all[i0:i1]
                mask_den, mask_edge = ((clean > EPS).float() * valid), (anno > 0.5).float()

                if DEBUG_GRAD and epoch <= DEBUG_GRAD_EPOCHS and not grad_debug_printed_this_epoch and i0 == 0:
                    generator.train()
                    discriminator.eval()
                    with autocast(enabled=use_amp):
                        out_dbg = generator.forward_train(noisy)
                        _, _, loss_edge_only_dbg = edge_loss_bce_dice(out_dbg["edge_logits"], edge_gt, mask_edge,
                                                                      band_ks=5)
                    zero_all_grads(generator, discriminator)
                    loss_edge_only_dbg.backward()
                    print(f"\n[GRAD DEBUG] epoch={epoch} (edge-only backward on first patch)")
                    grad_report(generator.edge_dec, "edge_dec");
                    grad_report(generator.den_dec, "den_dec");
                    grad_report(generator.shared_stem, "shared_stem");
                    grad_report(generator.edge_enc, "edge_enc");
                    grad_report(generator.den_enc, "den_enc")
                    zero_all_grads(generator, discriminator)
                    grad_debug_printed_this_epoch = True

                for p in discriminator.parameters(): p.requires_grad_(False)
                opt_g.zero_grad(set_to_none=True)

                with autocast(enabled=use_amp):
                    out = generator.forward_train(noisy)
                    den_raw, edge_logits = out["den"], out["edge_logits"]
                    den_clp = den_raw.clamp(0, 1)

                    loss_rec = masked_charbonnier_loss(den_raw, clean, mask_den, eps=1e-3)
                    loss_ssim = ssim_loss_bbox(den_clp, clean * valid, data_range=1.0, base_k=7, eps_mask=EPS)
                    _, _, loss_edge = edge_loss_bce_dice(edge_logits, edge_gt, mask_edge, band_ks=5)
                    cond_p, fake_p, real_p = sample_patch_3d(noisy, den_clp, clean, mask_den, patch=64)

                with autocast(enabled=False):
                    loss_adv_g = lsgan_g_loss(discriminator(cond_p.float(), fake_p.float()))

                loss_g = lambda_r * loss_rec.float() + lambda_g * loss_ssim.float() + lambda_e * loss_edge.float() + lambda_a * loss_adv_g.float()
                scaler_g.scale(loss_g).backward()
                scaler_g.step(opt_g)
                scaler_g.update()

                for p in discriminator.parameters(): p.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)

                with autocast(enabled=False):
                    loss_d = lsgan_d_loss(discriminator(cond_p.float(), real_p.float()),
                                          discriminator(cond_p.float(), fake_p.detach().float()))

                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()

                ep["loss_rec"] += float(loss_rec.detach().item());
                ep["loss_ssim"] += float(loss_ssim.detach().item());
                ep["loss_edge"] += float(loss_edge.detach().item())
                ep["loss_adv_g"] += float(loss_adv_g.detach().item());
                ep["loss_g"] += float(loss_g.detach().item());
                ep["loss_d"] += float(loss_d.detach().item())
                n_iter += 1

        if n_iter > 0:
            for k in ep.keys(): ep[k] /= float(n_iter)

        # Validation across the mixed validation loader
        if (epoch % VAL_INTERVAL) == 0:
            valm = validate_full_sliding(generator, val_loader, device, SLIDE_CHUNK, SLIDE_OVERLAP, EPS)
        else:
            valm = {"val_psnr": float("nan"), "val_bdice": float("nan")}

        torch.save({"epoch": epoch, "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "scaler_g": scaler_g.state_dict(),
                    "scaler_d": scaler_d.state_dict(), "val_metrics": valm}, LATEST_CKPT)

        if np.isfinite(valm["val_psnr"]):
            is_better = ((valm["val_psnr"] > best_psnr + delta_psnr) or (
                        abs(valm["val_psnr"] - best_psnr) <= delta_psnr and valm["val_bdice"] > best_bdice))
            if is_better:
                best_psnr, best_bdice = float(valm["val_psnr"]), float(valm["val_bdice"])
                torch.save(generator.state_dict(), BEST_GEN)
                torch.save(discriminator.state_dict(), BEST_DISC)
                torch.save({"epoch": epoch, "best_val_psnr": best_psnr, "best_val_bdice": best_bdice,
                            "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                            "val_metrics": valm}, BEST_CKPT)

        history["epoch"].append(epoch);
        history["lambda_a"].append(lambda_a)
        for k in ["loss_rec", "loss_ssim", "loss_edge", "loss_adv_g", "loss_g", "loss_d"]: history[k].append(ep[k])
        history["val_psnr"].append(valm["val_psnr"]);
        history["val_bdice"].append(valm["val_bdice"])

        pd.DataFrame(history).to_csv(os.path.join(OUT_ROOT, "train_history.csv"), index=False)
        if (epoch % SAVE_FIG_EVERY) == 0: plot_and_save_history(history, FIG_DIR, SMOOTH_WINDOW)

    print("Training finished. Saved latest:", LATEST_CKPT, "| Saved best:", BEST_CKPT)


# ============================================================
# Test & Metrics Saving (Independent Evaluator per Noise Level)
# ============================================================
def save_nifti_like(ref, data_3d: np.ndarray, out_path: str, dtype=np.float32):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = data_3d.astype(dtype, copy=False)
    if isinstance(ref, nib.Nifti1Image):
        affine, hdr = ref.affine, ref.header.copy()
    else:
        affine = np.asarray(ref, dtype=np.float64)
        hdr = nib.Nifti1Header()
        A = affine[:3, :3]
        zooms = np.where(np.sqrt((A * A).sum(axis=0)) > 0, np.sqrt((A * A).sum(axis=0)), 1.0).astype(np.float32)
        try:
            hdr.set_data_shape(arr.shape); hdr.set_zooms((float(zooms[0]), float(zooms[1]), float(zooms[2])))
        except:
            pass
    img = nib.Nifti1Image(arr, affine=affine, header=hdr)
    img.set_qform(affine, code=1);
    img.set_sform(affine, code=1);
    img.set_data_dtype(dtype)
    nib.save(img, out_path)


def _bbox_from_mask_np(mask3d: np.ndarray):
    idx = np.where(mask3d > 0)
    if idx[0].size == 0: return None
    return int(idx[0].min()), int(idx[0].max()), int(idx[1].min()), int(idx[1].max()), int(idx[2].min()), int(
        idx[2].max())


def _psnr_2d_torch_masked_batch(A3: torch.Tensor, B3: torch.Tensor, M3: torch.Tensor, data_range=1.0, eps=1e-12):
    M = M3.float()
    denom_safe = M.sum(dim=(1, 2)).clamp_min(1.0)
    mse = ((((A3 - B3) ** 2) * M).sum(dim=(1, 2)) / denom_safe).clamp_min(eps)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    return torch.where(M.sum(dim=(1, 2)) > 0, psnr, torch.full_like(psnr, float("nan")))


def _ssim_2d_torch_batch(A3: torch.Tensor, B3: torch.Tensor, data_range=1.0):
    N, H, W = A3.shape
    ky, kx = odd_kernel_leq(11, H), odd_kernel_leq(11, W)
    if min(ky, kx) < 3: return torch.full((N,), float("nan"), device=A3.device)
    with autocast(enabled=False):
        try:
            return tm_ssim(A3[:, None, :, :].float(), B3[:, None, :, :].float(), data_range=float(data_range),
                           kernel_size=(ky, kx), reduction="none")
        except TypeError:
            return torch.stack([tm_ssim(A3[i:i + 1, None].float(), B3[i:i + 1, None].float(),
                                        data_range=float(data_range), kernel_size=(ky, kx)) for i in range(N)], dim=0)


def _ssim_3d_bbox_torch(a3: torch.Tensor, b3: torch.Tensor, data_range=1.0):
    _, _, D, H, W = a3.shape
    kz, ky, kx = odd_kernel_leq(11, D), odd_kernel_leq(11, H), odd_kernel_leq(11, W)
    if min(kz, ky, kx) < 3: return torch.tensor(float("nan"), device=a3.device)
    with autocast(enabled=False):
        return tm_ssim(a3.float(), b3.float(), data_range=float(data_range), kernel_size=(kz, ky, kx))


def _soft_dice_2d_torch_batch_roi(pred3: torch.Tensor, tgt3: torch.Tensor, roi3: torch.Tensor, eps=1e-6):
    p, t = (pred3 * roi3).reshape(pred3.shape[0], -1), (tgt3 * roi3).reshape(tgt3.shape[0], -1)
    t_sum = t.sum(dim=1)
    dice = (2.0 * (p * t).sum(dim=1) + eps) / (p.sum(dim=1) + t_sum + eps)
    return torch.where(t_sum > 0, dice, torch.full_like(dice, float("nan")))


def sanitize_ring_k2d(k_inner: int, k_outer: int):
    k_inner, k_outer = max(1, int(k_inner)), max(1, int(k_outer))
    if k_inner % 2 == 0: k_inner -= 1
    if k_outer % 2 == 0: k_outer -= 1
    k_inner, k_outer = max(1, k_inner), max(3, k_outer)
    if k_inner >= k_outer: k_outer = k_inner + 2 + (1 if (k_inner + 2) % 2 == 0 else 0)
    return k_inner, k_outer


def dilate2d_batch(mask01: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool2d(mask01.unsqueeze(1), kernel_size=k, stride=1, padding=k // 2).squeeze(1)


def cnr_snr_2d_batch_from_roi(vol3: torch.Tensor, roi3: torch.Tensor, k_inner: int = 5, k_outer: int = 11,
                              eps: float = 1e-6):
    k_inner, k_outer = sanitize_ring_k2d(k_inner, k_outer)
    roi = (roi3 > 0.5).float()
    ring = (dilate2d_batch(roi, k_outer) - dilate2d_batch(roi, k_inner)).clamp(0, 1)
    v, r, b = vol3.reshape(vol3.shape[0], -1), roi.reshape(roi.shape[0], -1), ring.reshape(ring.shape[0], -1)

    valid = (r.sum(dim=1) > 0) & (b.sum(dim=1) > 0)
    mu_r = (v * r).sum(dim=1) / r.sum(dim=1).clamp_min(1.0)
    mu_b = (v * b).sum(dim=1) / b.sum(dim=1).clamp_min(1.0)
    std_r = torch.sqrt((((v - mu_r[:, None]) ** 2) * r).sum(dim=1) / r.sum(dim=1).clamp_min(1.0)).clamp_min(eps)
    std_b = torch.sqrt((((v - mu_b[:, None]) ** 2) * b).sum(dim=1) / b.sum(dim=1).clamp_min(1.0)).clamp_min(eps)

    snr = mu_r / std_r
    cnr = torch.abs(mu_r - mu_b) / std_b
    nan = vol3.new_full((vol3.shape[0],), float("nan"))
    return torch.where(valid, cnr, nan), torch.where(valid, snr, nan), torch.where(valid, mu_r, nan), torch.where(valid,
                                                                                                                  mu_b,
                                                                                                                  nan), torch.where(
        valid, std_r, nan), torch.where(valid, std_b, nan)


@torch.no_grad()
def run_test_and_save(loader, save_dir: str):
    """
    Test one noise level at a time.

    Outputs
    -------
    1) Per-case NIfTI files
    2) Per-case metrics CSV: test_case_metrics.csv
    3) One summary CSV for this noise level: test_summary_metrics.csv

    Notes
    -----
    Dynamic FiLM statistics exported here are the actual chunk/axis-averaged
    weights used during sliding inference, not a single whole-volume estimate.
    """
    generator.eval()
    os.makedirs(save_dir, exist_ok=True)
    out_cases = os.path.join(save_dir, "cases")
    os.makedirs(out_cases, exist_ok=True)
    case_records = []

    for batch_list in tqdm(loader, desc=f"Testing {os.path.basename(save_dir)}"):
        for b in batch_list:
            name = b["name"][0] if isinstance(b["name"], (list, tuple)) else b["name"]
            case_stem = os.path.splitext(name)[0]
            aff = b["affine"]
            affine = aff[0].cpu().numpy() if torch.is_tensor(aff) else (
                aff[0] if isinstance(aff, (list, tuple, np.ndarray)) and (
                    isinstance(aff, np.ndarray) and aff.ndim == 3
                ) else aff
            )

            gt = b["gt"].unsqueeze(0).to(device).float()
            noisy = b["noisy"].unsqueeze(0).to(device).float()
            anno = b["anno"].unsqueeze(0).to(device).float()

            # Forward process + export the actual sliding-inference dynamic weights.
            den, edge_prob_last, dyn_stats = forward_sliding(
                generator,
                noisy,
                chunk=SLIDE_CHUNK,
                overlap=SLIDE_OVERLAP,
                return_edge=True,
                return_weight_stats=True,
                axes=(2, 3, 4)
            )
            den = den.clamp(0, 1)
            edge_prob_last = edge_prob_last.clamp(0, 1)

            sigma_val = float(dyn_stats["estimated_noise_mad"].mean().item())
            wd2_val = float(dyn_stats["w_d2_from_e2"].mean().item())
            we2_val = float(dyn_stats["w_e2_from_d2"].mean().item())
            wd3_val = float(dyn_stats["w_d3_from_e3"].mean().item())
            we3_val = float(dyn_stats["w_e3_from_d3"].mean().item())

            boundary_gt = boundary_from_anno(anno).clamp(0, 1)
            case_dir = os.path.join(out_cases, case_stem)
            os.makedirs(case_dir, exist_ok=True)

            den_np = den[0, 0].cpu().numpy()
            edge_np = edge_prob_last[0, 0].cpu().numpy()
            anno_np = anno[0, 0].cpu().numpy()
            bnd_np = boundary_gt[0, 0].cpu().numpy()

            save_nifti_like(affine, den_np, os.path.join(case_dir, "denoised.nii"))
            save_nifti_like(affine, edge_np, os.path.join(case_dir, "edge_prob_last.nii"))
            save_nifti_like(
                affine,
                (edge_np * 255.0).round().clip(0, 255).astype(np.uint8),
                os.path.join(case_dir, "edge_prob_last_u8.nii"),
                dtype=np.uint8
            )
            save_nifti_like(
                affine,
                (anno_np > 0.5).astype(np.uint8),
                os.path.join(case_dir, "anno_mask.nii"),
                dtype=np.uint8
            )
            save_nifti_like(
                affine,
                (bnd_np > 0.5).astype(np.uint8),
                os.path.join(case_dir, "gt_boundary.nii"),
                dtype=np.uint8
            )

            mask3 = (gt > EPS).float()
            if mask3.sum().item() >= 1.0:
                psnr_noisy = masked_psnr(noisy, gt, mask3).item()
                psnr_den = masked_psnr(den, gt, mask3).item()
                rmse_noisy = masked_rmse(noisy, gt, mask3).item()
                rmse_den = masked_rmse(den, gt, mask3).item()
            else:
                psnr_noisy = psnr_den = rmse_noisy = rmse_den = float("nan")

            anno_roi = (anno > 0.5).float()
            cnr_noisy_t, snr_noisy_t, *_ = cnr_snr_from_anno(noisy, anno_roi)
            cnr_den_t, snr_den_t, *_ = cnr_snr_from_anno(den, anno_roi)

            cnr_noisy = float(cnr_noisy_t) if torch.isfinite(cnr_noisy_t) else float("nan")
            snr_noisy = float(snr_noisy_t) if torch.isfinite(snr_noisy_t) else float("nan")
            cnr_den = float(cnr_den_t) if torch.isfinite(cnr_den_t) else float("nan")
            snr_den = float(snr_den_t) if torch.isfinite(snr_den_t) else float("nan")

            bb = _bbox_from_mask_np((gt[0, 0].cpu().numpy() > EPS).astype(np.uint8))
            if bb is None:
                ssim_noisy = ssim_den = float("nan")
            else:
                d0, d1, h0, h1, w0, w1 = bb
                gt_box = gt[:, :, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1]
                noisy_box = noisy[:, :, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1]
                den_box = den[:, :, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1]

                ssim_noisy_t = _ssim_3d_bbox_torch(noisy_box, gt_box)
                ssim_den_t = _ssim_3d_bbox_torch(den_box, gt_box)

                ssim_noisy = float(ssim_noisy_t) if torch.isfinite(ssim_noisy_t) else float("nan")
                ssim_den = float(ssim_den_t) if torch.isfinite(ssim_den_t) else float("nan")

            case_records.append({
                "name": case_stem,

                # Actual dynamic FiLM statistics used during sliding inference
                "estimated_noise_mad_mean": sigma_val,
                "w_d2_from_e2_mean": wd2_val,
                "w_e2_from_d2_mean": we2_val,
                "w_d3_from_e3_mean": wd3_val,
                "w_e3_from_d3_mean": we3_val,

                "psnr_noisy_3d_mask": psnr_noisy,
                "psnr_denoised_3d_mask": psnr_den,
                "delta_psnr": psnr_den - psnr_noisy,

                "rmse_noisy_3d_mask": rmse_noisy,
                "rmse_denoised_3d_mask": rmse_den,
                "delta_rmse": rmse_den - rmse_noisy,

                "cnr_noisy_3d_anno_ring": cnr_noisy,
                "cnr_denoised_3d_anno_ring": cnr_den,
                "delta_cnr": cnr_den - cnr_noisy,

                "snr_noisy_3d_anno_ring": snr_noisy,
                "snr_denoised_3d_anno_ring": snr_den,
                "delta_snr": snr_den - snr_noisy,

                "ssim_noisy_3d_bbox": ssim_noisy,
                "ssim_denoised_3d_bbox": ssim_den,
                "delta_ssim": ssim_den - ssim_noisy,
            })

            if bb is not None:
                gt_box_t = gt_box[0, 0]
                noisy_box_t = noisy_box[0, 0]
                den_box_t = den_box[0, 0]
                edge_box_t = edge_prob_last[0, 0, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1]
                bnd_box_t = boundary_gt[0, 0, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1]
                anno_box_t = anno[0, 0, d0:d1 + 1, h0:h1 + 1, w0:w1 + 1]

                Db, Hb, Wb = gt_box_t.shape

                # Axial
                mask_ax = (gt_box_t > EPS)
                pd.DataFrame({
                    "slice_index": (torch.arange(Db, device=device) + d0).cpu().numpy().astype(int),
                    "psnr_noisy_bbox2d": _psnr_2d_torch_masked_batch(noisy_box_t, gt_box_t, mask_ax).cpu().numpy(),
                    "psnr_denoised_bbox2d": _psnr_2d_torch_masked_batch(den_box_t, gt_box_t, mask_ax).cpu().numpy(),
                    "ssim_noisy_bbox2d": _ssim_2d_torch_batch(noisy_box_t, gt_box_t).cpu().numpy(),
                    "ssim_denoised_bbox2d": _ssim_2d_torch_batch(den_box_t, gt_box_t).cpu().numpy(),
                    "boundary_soft_dice": _soft_dice_2d_torch_batch_roi(edge_box_t, bnd_box_t, anno_box_t).cpu().numpy(),
                    "rmse_noisy_bbox2d": _rmse_2d_torch_masked_batch(noisy_box_t, gt_box_t, mask_ax).cpu().numpy(),
                    "rmse_denoised_bbox2d": _rmse_2d_torch_masked_batch(den_box_t, gt_box_t, mask_ax).cpu().numpy(),
                    "cnr_noisy_roi_ring2d": cnr_snr_2d_batch_from_roi(noisy_box_t, anno_box_t)[0].cpu().numpy(),
                    "cnr_denoised_roi_ring2d": cnr_snr_2d_batch_from_roi(den_box_t, anno_box_t)[0].cpu().numpy(),
                    "snr_noisy_roi2d": cnr_snr_2d_batch_from_roi(noisy_box_t, anno_box_t)[1].cpu().numpy(),
                    "snr_denoised_roi2d": cnr_snr_2d_batch_from_roi(den_box_t, anno_box_t)[1].cpu().numpy(),
                }).to_csv(os.path.join(case_dir, "slice_metrics_axial_z.csv"), index=False)

                # Coronal
                gt_cor = gt_box_t.permute(1, 0, 2)
                noisy_cor = noisy_box_t.permute(1, 0, 2)
                den_cor = den_box_t.permute(1, 0, 2)
                edge_cor = edge_box_t.permute(1, 0, 2)
                bnd_cor = bnd_box_t.permute(1, 0, 2)
                roi_cor = anno_box_t.permute(1, 0, 2)
                mask_cor = (gt_cor > EPS)

                pd.DataFrame({
                    "slice_index": (torch.arange(Hb, device=device) + h0).cpu().numpy().astype(int),
                    "psnr_noisy_bbox2d": _psnr_2d_torch_masked_batch(noisy_cor, gt_cor, mask_cor).cpu().numpy(),
                    "psnr_denoised_bbox2d": _psnr_2d_torch_masked_batch(den_cor, gt_cor, mask_cor).cpu().numpy(),
                    "ssim_noisy_bbox2d": _ssim_2d_torch_batch(noisy_cor, gt_cor).cpu().numpy(),
                    "ssim_denoised_bbox2d": _ssim_2d_torch_batch(den_cor, gt_cor).cpu().numpy(),
                    "boundary_soft_dice": _soft_dice_2d_torch_batch_roi(edge_cor, bnd_cor, roi_cor).cpu().numpy(),
                    "rmse_noisy_bbox2d": _rmse_2d_torch_masked_batch(noisy_cor, gt_cor, mask_cor).cpu().numpy(),
                    "rmse_denoised_bbox2d": _rmse_2d_torch_masked_batch(den_cor, gt_cor, mask_cor).cpu().numpy(),
                    "cnr_noisy_roi_ring2d": cnr_snr_2d_batch_from_roi(noisy_cor, roi_cor)[0].cpu().numpy(),
                    "cnr_denoised_roi_ring2d": cnr_snr_2d_batch_from_roi(den_cor, roi_cor)[0].cpu().numpy(),
                    "snr_noisy_roi2d": cnr_snr_2d_batch_from_roi(noisy_cor, roi_cor)[1].cpu().numpy(),
                    "snr_denoised_roi2d": cnr_snr_2d_batch_from_roi(den_cor, roi_cor)[1].cpu().numpy(),
                }).to_csv(os.path.join(case_dir, "slice_metrics_coronal_y.csv"), index=False)

                # Sagittal
                gt_sag = gt_box_t.permute(2, 0, 1)
                noisy_sag = noisy_box_t.permute(2, 0, 1)
                den_sag = den_box_t.permute(2, 0, 1)
                edge_sag = edge_box_t.permute(2, 0, 1)
                bnd_sag = bnd_box_t.permute(2, 0, 1)
                roi_sag = anno_box_t.permute(2, 0, 1)
                mask_sag = (gt_sag > EPS)

                pd.DataFrame({
                    "slice_index": (torch.arange(Wb, device=device) + w0).cpu().numpy().astype(int),
                    "psnr_noisy_bbox2d": _psnr_2d_torch_masked_batch(noisy_sag, gt_sag, mask_sag).cpu().numpy(),
                    "psnr_denoised_bbox2d": _psnr_2d_torch_masked_batch(den_sag, gt_sag, mask_sag).cpu().numpy(),
                    "ssim_noisy_bbox2d": _ssim_2d_torch_batch(noisy_sag, gt_sag).cpu().numpy(),
                    "ssim_denoised_bbox2d": _ssim_2d_torch_batch(den_sag, gt_sag).cpu().numpy(),
                    "boundary_soft_dice": _soft_dice_2d_torch_batch_roi(edge_sag, bnd_sag, roi_sag).cpu().numpy(),
                    "rmse_noisy_bbox2d": _rmse_2d_torch_masked_batch(noisy_sag, gt_sag, mask_sag).cpu().numpy(),
                    "rmse_denoised_bbox2d": _rmse_2d_torch_masked_batch(den_sag, gt_sag, mask_sag).cpu().numpy(),
                    "cnr_noisy_roi_ring2d": cnr_snr_2d_batch_from_roi(noisy_sag, roi_sag)[0].cpu().numpy(),
                    "cnr_denoised_roi_ring2d": cnr_snr_2d_batch_from_roi(den_sag, roi_sag)[0].cpu().numpy(),
                    "snr_noisy_roi2d": cnr_snr_2d_batch_from_roi(noisy_sag, roi_sag)[1].cpu().numpy(),
                    "snr_denoised_roi2d": cnr_snr_2d_batch_from_roi(den_sag, roi_sag)[1].cpu().numpy(),
                }).to_csv(os.path.join(case_dir, "slice_metrics_sagittal_x.csv"), index=False)

    # Per-case CSV
    df = pd.DataFrame(case_records)
    case_csv_path = os.path.join(save_dir, "test_case_metrics.csv")
    df.to_csv(case_csv_path, index=False)

    # Summary CSV for the current noise level
    summary_csv_path = os.path.join(save_dir, "test_summary_metrics.csv")
    if len(df) > 0:
        numeric_df = df.select_dtypes(include=[np.number])

        summary_row = {
            "n_cases": int(len(df))
        }
        for col in numeric_df.columns:
            summary_row[f"avg_{col}"] = float(numeric_df[col].mean(skipna=True))
            summary_row[f"std_{col}"] = float(numeric_df[col].std(skipna=True, ddof=0))

        pd.DataFrame([summary_row]).to_csv(summary_csv_path, index=False)
    else:
        pd.DataFrame([{"n_cases": 0}]).to_csv(summary_csv_path, index=False)

    print(f"Case metrics saved to: {case_csv_path}")
    print(f"Summary metrics saved to: {summary_csv_path}")


if RUN_TEST:
    if TEST_USE == "best":
        generator.load_state_dict(torch.load(BEST_GEN, map_location=device))
    else:
        state = torch.load(LATEST_CKPT, map_location=device)
        generator.load_state_dict(state["generator"])

    generator.eval()
    torch.cuda.empty_cache()

    for level, loader in test_loaders.items():
        print(f"\n================ Running Test for Noise Level: {level} ================")
        out_dir_for_level = os.path.join(OUT_ROOT, "test_outputs", f"noise_{level}")
        run_test_and_save(loader, out_dir_for_level)
