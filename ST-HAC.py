import os, math, json, time, random
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

@dataclass
class CFG:
    TRAIN_NPY: str = 
    TEST_NPY:  str = 
    P_IN: int  = 25

    H_ORIG: int = 100
    W_ORIG: int = 100
    C_ORIG: int = 3

    CHANNELS: list = None
    C: int = 3

    CROP_MODE: str = "center"
    CROP_SIZE: int = 20
    H: int = 20
    W: int = 20

    BATCH: int = 32
    EPOCHS: int = 300
    LR: float = 5e-4
    WD: float = 1e-3
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True

    D_MODEL: int = 128
    PERIODS: list = None
    NUM_FREQUENCIES: int = 16
    DROPOUT: float = 0.1

    LOSS_DELTA: float = 0.8
    SAVE_DIR: str = "./checkpoints"
    SEED: int = 42

    USE_PERIODIC: bool = True
    USE_NOISE: bool = True
    USE_FFT: bool = True

    USE_RESIDUAL_TO_LAST: bool = True

    USE_SPATIAL: bool = True
    N_PROTO: int = 128
    OT_EPS: float = 0.08
    OT_TAU: float = 1.4
    OT_USE_EMB_DIST: bool = True
    DLAK_K: int = 8

    USE_PARA: bool = True
    PARA_R0: float = 1.5

    USE_QLOSS: bool = True
    Q_W: float = 0.3
    USE_CPD_ATT: bool = True
    CPD_GAIN: float = 0.2

    USE_CPD_BETTER: bool = True

cfg = CFG()

if cfg.CHANNELS is None:
    cfg.CHANNELS = list(range(cfg.C_ORIG))
cfg.C = len(cfg.CHANNELS)

if cfg.PERIODS is None:
    cfg.PERIODS = [48, 168]

def set_seed(seed=42):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(x):
    return x.to(cfg.DEVICE, non_blocking=True)

def crop_data(arr: np.ndarray, crop_mode: str, crop_size: int, seed: int = None) -> np.ndarray:
    T, H_orig, W_orig, C = arr.shape
    if crop_mode == "none":
        print("No cropping applied");
        return arr
    if crop_size > min(H_orig, W_orig):
        print(f"Warning: crop_size ({crop_size}) larger than image size ({H_orig}x{W_orig}). No cropping applied.")
        return arr
    if crop_mode == "center":
        center_h, center_w = H_orig // 2, W_orig // 2
        half_crop = crop_size // 2
        start_h = center_h - half_crop
        start_w = center_w - half_crop
    elif crop_mode == "top_left":
        start_h, start_w = 0, 0
    elif crop_mode == "bottom_right":
        start_h = H_orig - crop_size
        start_w = W_orig - crop_size
    elif crop_mode == "random":
        if seed is not None:
            np.random.seed(seed)
        start_h = np.random.randint(0, H_orig - crop_size + 1)
        start_w = np.random.randint(0, W_orig - crop_size + 1)
    else:
        raise ValueError(f"Unknown crop_mode: {crop_mode}")
    start_h = max(0, min(start_h, H_orig - crop_size))
    start_w = max(0, min(start_w, W_orig - crop_size))
    end_h = start_h + crop_size
    end_w = start_w + crop_size
    cropped = arr[:, start_h:end_h, start_w:end_w, :]
    print(f"Applied {crop_mode} crop: {arr.shape} -> {cropped.shape}")
    print(f"Crop region: H[{start_h}:{end_h}], W[{start_w}:{end_w}]")
    return cropped

def select_channels(arr: np.ndarray, channels: list, channel_names: list = None) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError(f"Expected arr shape (T,H,W,C), got {arr.shape} (ndim={arr.ndim})")

    T, H, W, C_orig = arr.shape
    if channels is None or len(channels) == 0:
        raise ValueError("channels must be a non-empty list of channel indices.")

    invalid = [ch for ch in channels if (not isinstance(ch, (int, np.integer))) or ch < 0 or ch >= C_orig]
    if invalid:
        raise ValueError(f"Invalid channel indices {invalid}. Data has {C_orig} channels (0-{C_orig-1})")

    selected = arr[:, :, :, channels]

    if channel_names is None or len(channel_names) != C_orig:
        names_all = [f"Ch{i}" for i in range(C_orig)]
    else:
        names_all = list(channel_names)

    sel_names = [names_all[i] for i in channels]

    print(f"Selected channels: {channels} -> {sel_names}")
    print(f"Channel selection: {arr.shape} -> {selected.shape}")
    return selected


def sliding_windows(arr: np.ndarray, p_in: int) -> Tuple[np.ndarray, np.ndarray]:
    T = arr.shape[0]
    N = T - p_in
    X = np.stack([arr[i:i+p_in] for i in range(N)], axis=0)
    y = np.stack([arr[i+p_in] for i in range(N)], axis=0)
    return X, y

def norm_fit(x_train: np.ndarray):
    min_v = x_train.min(axis=(0,1,2), keepdims=True)
    max_v = x_train.max(axis=(0,1,2), keepdims=True)
    return min_v, max_v

def norm_apply(x: torch.Tensor, min_v: torch.Tensor, max_v: torch.Tensor, eps: float = 1e-6):
    scale = (max_v - min_v)
    scale = torch.where(scale < eps, torch.ones_like(scale), scale)
    return (x - min_v) / (scale + eps)

def denorm(x: torch.Tensor, min_v: torch.Tensor, max_v: torch.Tensor, eps: float = 1e-6):
    return min_v + x * (max_v - min_v + eps)

def grid_coords(H, W, device):
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    coords = torch.stack([yy, xx], dim=-1).reshape(-1,2).float()
    return coords

def build_knn_idx_and_relpos(H, W, k, device):
    coords = grid_coords(H, W, device)
    N = coords.size(0)
    d2 = torch.cdist(coords, coords, p=2.0)**2
    d2 = d2 + torch.eye(N, device=device)*1e9
    idx = torch.topk(-d2, k, dim=-1).indices
    rel = coords.unsqueeze(1) - coords[idx]
    rel[...,0] = 2*rel[...,0]/max(1,(H-1))
    rel[...,1] = 2*rel[...,1]/max(1,(W-1))
    geo = rel.norm(dim=-1)
    return idx, rel, geo, coords

class WinDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

def _robust_z(d, eps=1e-6):
    med = d.median(dim=1, keepdim=True).values
    mad = (d - med).abs().median(dim=1, keepdim=True).values
    return (d - med) / (mad*1.4826 + eps)

def _residual_frame(x):
    base = x.clone()
    base[:,2:] = (x[:,1:-1] + x[:,0:-2] + x[:,2:]) / 3.0
    base[:,1]  = x[:,0]
    return x - base

def _cpd_score_better(x_raw, lags=(1,2,4,6), alpha=0.6):
    x_res = _residual_frame(x_raw)
    diffs = []
    for ell in lags:
        pad = x_res[:, :ell]
        shifted = torch.cat([pad, x_res[:, :-ell]], 1)
        d = (x_res - shifted).abs().mean(dim=(2,3,4))
        diffs.append(F.softplus(_robust_z(d)))
    g = torch.stack(diffs, -1).mean(-1)
    out=[]
    for b in range(g.size(0)):
        ema=g[b,0]; buf=[ema]
        for t in range(1,g.size(1)):
            ema=alpha*ema+(1-alpha)*g[b,t]; buf.append(ema)
        out.append(torch.stack(buf))
    g=torch.stack(out,0)
    return torch.sigmoid(1.5*(g - g.mean(dim=1, keepdim=True))).detach()

def cpd_score(x_raw):
    return _cpd_score_better(x_raw)

class MultiPeriodicEmbed(nn.Module):
    def __init__(self, d_model, periods=[24, 72]):
        super().__init__()
        self.periods = periods
        n = len(periods)
        base = d_model // n
        rem  = d_model - base * n
        self._parts = [base + (1 if i < rem else 0) for i in range(n)]
        self.period_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, p),
                nn.GELU(),
                nn.Linear(p, p)
            ) for p in self._parts
        ])
    def forward(self, time_idx):
        embeds = []
        for (period, layer) in zip(self.periods, self.period_embeds):
            phase = 2 * math.pi * time_idx / period
            sin_cos = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
            embeds.append(layer(sin_cos))
        return torch.cat(embeds, dim=-1)

class FrequencyAwareEncoder(nn.Module):
    def __init__(self, d_model, num_frequencies=16):
        super().__init__()
        self.num_freq = num_frequencies
        self.freq_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(num_frequencies)
        ])
        self.combiner = nn.Sequential(
            nn.Linear(d_model * num_frequencies, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, x):
        B, P, N, d = x.shape
        x_reshaped = x.permute(0, 2, 3, 1)
        x_freq = torch.fft.rfft(x_reshaped, dim=-1)
        freq_bins = x_freq.shape[-1]
        processed_freqs = []
        for i in range(min(self.num_freq, freq_bins)):
            real_comp = x_freq[:,:,:,i].real
            imag_comp = x_freq[:,:,:,i].imag
            comp = torch.cat([real_comp, imag_comp], dim=-1)
            processed = self.freq_processors[i](comp)
            processed_freqs.append(processed)
        if len(processed_freqs) == 0:
            processed_freqs = [torch.zeros(B, N, d, device=x.device)]
        while len(processed_freqs) < self.num_freq:
            processed_freqs.append(torch.zeros_like(processed_freqs[0]))
        combined = torch.cat(processed_freqs, dim=-1)
        output = self.combiner(combined)
        return output.unsqueeze(1).expand(-1, P, -1, -1)

class EnhancedTemporalEncoder(nn.Module):
    def __init__(self, C_in, d_model, P):
        super().__init__()
        self.P = P
        self.d_model = d_model
        self.in_proj = nn.Linear(C_in, d_model)

        self.periodic_embed = MultiPeriodicEmbed(d_model//2, cfg.PERIODS)
        self.noise_embed = nn.Linear(C_in, d_model//2)

        self.freq_encoder = FrequencyAwareEncoder(d_model, cfg.NUM_FREQUENCIES)

        self.pool_param = nn.Parameter(torch.randn(P, 1))

    def forward(self, x, cpd_hint=None):
        B, P, H, W, C = x.shape
        N = H * W
        x_flat = x.reshape(B, P, N, C)
        h = self.in_proj(x_flat)

        time_idx = torch.arange(P, device=x.device).float().unsqueeze(0).expand(B, -1)
        periodic_emb = self.periodic_embed(time_idx) if cfg.USE_PERIODIC else torch.zeros(B, P, self.d_model//2, device=x.device)
        spatial_mean = x_flat.mean(dim=2)
        noise_emb = self.noise_embed(spatial_mean) if cfg.USE_NOISE else torch.zeros(B, P, self.d_model//2, device=x.device)

        time_emb = torch.cat([periodic_emb, noise_emb], dim=-1)
        h = h + time_emb.unsqueeze(2)

        if cfg.USE_FFT:
            h_freq = self.freq_encoder(h)
            h = h + 0.3 * h_freq

        base_att = torch.softmax(self.pool_param.squeeze(-1), dim=0)

        if cfg.USE_CPD_ATT:
            if cpd_hint is None:
                cpd = cpd_score(x)
            else:
                cpd = cpd_hint
                if cpd.dim() == 1:
                    cpd = cpd.unsqueeze(0).expand(B, -1)

            att = base_att[None, :] * (1.0 + cfg.CPD_GAIN * cpd)
            att = torch.softmax(att, dim=1)
        else:
            att = base_att[None, :].expand(B, -1)

        att = att[:, :, None, None]
        z = (h * att).sum(dim=1)
        time_vec = time_emb.mean(dim=1)

        return z, time_vec

class OTProtoSpatial(nn.Module):
    def __init__(self, d_model, H, W, K, eps=0.1, use_emb_dist=True, tau=1.0, dropout=0.1):
        super().__init__()
        self.K = K
        self.proto = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_p = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(),
                                 nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        self.eps = eps
        self.tau = tau
        self.use_emb_dist = use_emb_dist
        yy, xx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(-1,2)
        self.register_buffer("coords", coords)
        self.proto_xy = nn.Parameter(torch.randn(K, 2) * 0.1)

    def forward(self, z, margin: torch.Tensor = None, eps_override: float = None, tau_override: float = None):
        B, N, D = z.shape
        device = z.device
        coords = self.coords.to(device)
        q = self.proj_q(z)
        p = self.proj_p(self.proto).unsqueeze(0).expand(B,-1,-1)
        proto_xy = torch.tanh(self.proto_xy)
        geod = torch.cdist(coords.unsqueeze(0), proto_xy.unsqueeze(0), p=2.0).expand(B, -1, -1)

        if self.use_emb_dist:
            qn = F.normalize(q, dim=-1); pn = F.normalize(p, dim=-1)
            emb = 1.0 - torch.einsum("bnd,bkd->bnk", qn, pn)
            C = geod + emb
        else:
            C = geod

        eps = float(self.eps if eps_override is None else eps_override)
        tau = float(self.tau if tau_override is None else tau_override)
        logits = -C / (eps + 1e-6)
        if margin is not None:
            logits = logits + margin

        T = torch.softmax(tau * logits, dim=-1)
        mixed = torch.einsum("bnk,bkd->bnd", T, p)
        out = self.norm(z + mixed)
        out = self.norm(out + self.ffn(out))
        return out, T, geod

class PARA(nn.Module):
    def __init__(self, d_model, H, W, K=16, tv_w=1e-3, geo_w=0.25, sim_w=1.0, dropout=0.1):
        super().__init__()
        self.H, self.W, self.K = H, W, K
        self.tv_w = tv_w
        self.geo_w = geo_w
        self.sim_w = sim_w
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.out  = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)
        self.dk = d_model

        self.s_summary = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 16), nn.GELU())
        self.z_gate = nn.Sequential(nn.Linear(d_model, 16), nn.GELU())
        self.param_head = nn.Linear(32, 3)
        self.scale_gate = nn.Linear(32, 3)

        self.register_buffer("_knn_idx", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_relpos", torch.empty(0), persistent=False)
        self.register_buffer("_geod", torch.empty(0), persistent=False)

        self.drop_path = nn.Dropout(p=0.1)

    def _ensure_graph(self, device):
        if self._knn_idx.numel() == 0 or self._knn_idx.device != device:
            idx, rel, geo, _ = build_knn_idx_and_relpos(self.H, self.W, cfg.DLAK_K, device)
            self._knn_idx = idx; self._relpos = rel; self._geod = geo

    def _gather_neighbors(self, x, idx):
        B = x.shape[0]
        b = torch.arange(B, device=idx.device)[:, None, None]
        n_idx = idx[None, :, :]
        return x[b, n_idx]

    def _tv2d(self, tensor_2d):
        dy = (tensor_2d[:,1:,:] - tensor_2d[:,:-1,:]).abs().mean()
        dx = (tensor_2d[:,:,1:] - tensor_2d[:,:,:-1]).abs().mean()
        return dx + dy

    def forward(self, z, S):
        device = z.device
        self._ensure_graph(device)
        idx = self._knn_idx; rel = self._relpos; geod = self._geod

        B,N,D = z.shape
        K = idx.shape[1]

        q = self.to_q(z); k = self.to_k(z); v = self.to_v(z)
        kN = self._gather_neighbors(k, idx)
        vN = self._gather_neighbors(v, idx)
        S_n = self._gather_neighbors(S, idx)

        m1 = S.mean(dim=-1, keepdim=True)
        m2 = (S*S).mean(dim=-1, keepdim=True)
        s_feat = torch.cat([m1, m2], dim=-1)
        s_emb  = self.s_summary(s_feat)
        z_emb  = self.z_gate(z)
        ker_in = torch.cat([s_emb, z_emb], dim=-1)

        a,b,c  = self.param_head(ker_in).unbind(-1)
        La = torch.exp(a).unsqueeze(-1).unsqueeze(-1)
        Lb = b.unsqueeze(-1).unsqueeze(-1)
        Lc = torch.exp(c).unsqueeze(-1).unsqueeze(-1)

        pad = torch.zeros_like(La)
        L = torch.cat([torch.cat([La, pad], dim=-1),
                       torch.cat([Lb, Lc], dim=-1)], dim=-2)
        A = torch.matmul(L, L.transpose(-1,-2))

        rel_b = rel.unsqueeze(0).expand(B, -1, -1, -1)
        rAr   = torch.einsum('bnkd,bndc,bnkc->bnk', rel_b, A, rel_b)

        S_q = S.unsqueeze(2)
        sim = (S_q * S_n).sum(-1)

        content = (q.unsqueeze(2) * kN).sum(-1) / math.sqrt(self.dk)

        SC = [dict(r=1.5, s0=0.25, sl=0.20, cb=0.70),
              dict(r=2.5, s0=0.35, sl=0.20, cb=0.50),
              dict(r=4.0, s0=0.55, sl=0.25, cb=0.30)]

        logits_scales = []
        geod_b = geod.unsqueeze(0)
        big_neg = -1e4

        for sc in SC:
            hard_mask = (geod > sc["r"]).to(z.dtype).unsqueeze(0)
            sigma = (sc["s0"] + sc["sl"]*geod).unsqueeze(0)
            rAr_scaled = rAr / (sigma*sigma).clamp_min(1e-6)

            center_prior = torch.exp(-(geod_b**2) / (cfg.PARA_R0**2))
            center_term  = sc["cb"] * center_prior

            att_i = content - rAr_scaled - 0.25*geod_b + 1.0*sim + center_term
            att_i = att_i + hard_mask*big_neg
            logits_scales.append(att_i)

        g = torch.softmax(self.scale_gate(ker_in), dim=-1)
        att_logits = (logits_scales[0] * g[...,0:1] +
                      logits_scales[1] * g[...,1:2] +
                      logits_scales[2] * g[...,2:3])

        att = torch.softmax(att_logits, dim=-1)
        out = (att.unsqueeze(-1) * vN).sum(2)
        out = self.norm(z + self.drop_path(self.out(out)))

        A_norm = A.reshape(B, self.H, self.W, 2, 2).norm(dim=(-1,-2))
        reg = self.tv_w * self._tv2d(A_norm)
        return out, reg, {}

class SpatialEncoder(nn.Module):
    def __init__(self, d, H, W, n_proto):
        super().__init__()
        self.H, self.W = H, W
        self.ot = OTProtoSpatial(
            d_model=d, H=H, W=W, K=n_proto,
            eps=cfg.OT_EPS,
            use_emb_dist=cfg.OT_USE_EMB_DIST,
            tau=cfg.OT_TAU,
            dropout=cfg.DROPOUT
        )
        self.para = PARA(
            d_model=d, H=H, W=W, K=cfg.DLAK_K,
            tv_w=1e-3, geo_w=0.25, sim_w=1.0, dropout=cfg.DROPOUT
        ) if cfg.USE_PARA else None

    def forward(self, z_cell):
        if not cfg.USE_SPATIAL:
            return z_cell, torch.tensor(0.0, device=z_cell.device)
        h1, S1, _ = self.ot(z_cell)
        if self.para is not None:
            h_para1, reg1, _ = self.para(h1, S1)
            return h_para1, reg1
        return h1, torch.tensor(0.0, device=z_cell.device)

class STHAC(nn.Module):
    def __init__(self, H, W, C, P):
        super().__init__()
        self.temporal = EnhancedTemporalEncoder(C, cfg.D_MODEL, P)
        self.spatial  = SpatialEncoder(cfg.D_MODEL, H, W, cfg.N_PROTO)
        self.out_proj = nn.Linear(cfg.D_MODEL, C)
        self.H=H; self.W=W

    def forward(self, x, cpd_hint=None):
        B = x.size(0)
        last_in = x[:, -1]

        z_cell, time_vec = self.temporal(x, cpd_hint=cpd_hint)
        h, reg_spa = self.spatial(z_cell)
        y_delta = self.out_proj(h).view(B, self.H, self.W, -1)
        y = last_in + y_delta if cfg.USE_RESIDUAL_TO_LAST else y_delta
        return y, reg_spa


def compute_metrics(y_true, y_pred, eps=1e-8):
    B,H,W,C = y_true.shape
    yt = y_true.reshape(B,-1,C); yp = y_pred.reshape(B,-1,C)
    diff = yp - yt

    mse = (diff**2).mean(dim=1).mean(dim=0)
    rmse = torch.sqrt(mse + eps)
    mae  = diff.abs().mean(dim=1).mean(dim=0)
    smape= ( (yp-yt).abs() / (yt.abs()+yp.abs()+eps) ).mean(dim=1).mean(dim=0)

    yt_mean = yt.mean(dim=1, keepdim=True)
    ss_tot  = ((yt-yt_mean)**2).sum(dim=1).mean(dim=0)
    ss_res  = (diff**2).sum(dim=1).mean(dim=0)
    r2 = 1 - ss_res/(ss_tot+eps)

    rmse_g = torch.sqrt(((diff**2).mean()))
    mae_g  = diff.abs().mean()
    smape_g= ( (yp-yt).abs() / (yt.abs()+yp.abs()+eps) ).mean()
    r2_g   = 1 - (diff**2).sum() / (((yt-yt.mean())**2).sum()+eps)

    return {
        "per_channel": {
            "rmse": rmse.detach().cpu().tolist(),
            "mae":  mae.detach().cpu().tolist(),
            "smape": smape.detach().cpu().tolist(),
            "r2":   r2.detach().cpu().tolist(),
        },
        "global": {
            "rmse": float(rmse_g.detach().cpu()),
            "mae":  float(mae_g.detach().cpu()),
            "smape": float(smape_g.detach().cpu()),
            "r2":   float(r2_g.detach().cpu()),
        }
    }

def freq_weighted_loss(y, mu, eps=1e-8):
    yf = torch.fft.rfft2(y.permute(0, 3, 1, 2))
    mf = torch.fft.rfft2(mu.permute(0, 3, 1, 2))
    df = yf - mf
    H = y.shape[1]; W = y.shape[2]
    Hf, Wf = df.shape[2], df.shape[3]
    yy = torch.linspace(-1, 1, H, device=y.device).unsqueeze(1).expand(H, W)
    xx = torch.linspace(-1, 1, W, device=y.device).unsqueeze(0).expand(H, W)
    r = torch.sqrt(yy**2 + xx**2)[:Hf, :Wf]
    w_low = torch.exp(-3.0 * r)
    Wmap = w_low.unsqueeze(0).unsqueeze(0)
    fw = (Wmap * df.abs()).mean()
    return fw

def pinball_loss(y, yhat, tau):
    return torch.maximum(tau*(y - yhat), (tau-1)*(y - yhat)).mean()

def train_one_epoch(model, loader, min_v, max_v, optim):
    model.train()
    loss_fn = nn.HuberLoss(delta=cfg.LOSS_DELTA)
    total = 0.0

    for X,y in loader:
        X = to_device(X); y = to_device(y)

        cpd_hint = cpd_score(X) if cfg.USE_CPD_ATT else None

        Xn = norm_apply(X, min_v, max_v)
        yn = norm_apply(y, min_v, max_v)

        yp, reg_loss = model(Xn, cpd_hint=cpd_hint)

        huber = loss_fn(yp, yn)
        fw = freq_weighted_loss(yn, yp) if cfg.USE_FFT else torch.tensor(0.0, device=yp.device)
        q_loss = (pinball_loss(yn, yp, 0.2) + pinball_loss(yn, yp, 0.5) + pinball_loss(yn, yp, 0.8)) / 3.0 if cfg.USE_QLOSS else torch.tensor(0.0, device=yp.device)

        loss = huber + reg_loss + 0.1 * fw + cfg.Q_W * q_loss

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        total += float(loss.detach().cpu())

    return total/len(loader)

@torch.no_grad()
def eval_model(model, loader, min_v, max_v):
    model.eval()
    loss_fn = nn.HuberLoss(delta=cfg.LOSS_DELTA)
    total = 0.0
    all_pred, all_true = [], []

    for X, y in loader:
        X = to_device(X); y = to_device(y)

        cpd_hint = cpd_score(X) if cfg.USE_CPD_ATT else None

        Xn = norm_apply(X, min_v, max_v)
        yn = norm_apply(y, min_v, max_v)

        yp, reg_loss = model(Xn, cpd_hint=cpd_hint)

        huber = loss_fn(yp, yn)
        fw = freq_weighted_loss(yn, yp) if cfg.USE_FFT else torch.tensor(0.0, device=yp.device)
        q_loss = (pinball_loss(yn, yp, 0.2) + pinball_loss(yn, yp, 0.5) + pinball_loss(yn, yp, 0.8)) / 3.0 if cfg.USE_QLOSS else torch.tensor(0.0, device=yp.device)

        loss = huber + reg_loss + 0.1 * fw + cfg.Q_W * q_loss
        total += float(loss.detach().cpu())

        ypd = denorm(yp, min_v, max_v)
        all_pred.append(ypd.cpu())
        all_true.append(y.cpu())

    y_pred = torch.cat(all_pred, dim=0)
    y_true = torch.cat(all_true, dim=0)
    metrics = compute_metrics(y_true, y_pred)
    return total / len(loader), metrics

def main():
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    set_seed(cfg.SEED)

    print("Device:", cfg.DEVICE)
    print("STHAC (paper-clean, perf-safe, per-sample CPD)")

    print("Loading data...")
    tr_full = np.load(cfg.TRAIN_NPY)
    te_full = np.load(cfg.TEST_NPY)
    print(f"Original train shape: {tr_full.shape}")
    print(f"Original test shape:  {te_full.shape}")

    cfg.C_ORIG = tr_full.shape[3]
    assert tr_full.shape[1] == cfg.H_ORIG and tr_full.shape[2] == cfg.W_ORIG
    assert te_full.shape[1] == cfg.H_ORIG and te_full.shape[2] == cfg.W_ORIG

    if cfg.C_ORIG == 3:
        channel_names_all = ["SMS", "Call", "Internet"]
    else:
        channel_names_all = [f"Ch{i}" for i in range(cfg.C_ORIG)]

    selected_channel_names = [channel_names_all[i] for i in cfg.CHANNELS]
    print(f"{'Single' if len(cfg.CHANNELS) == 1 else 'Multi'}-channel mode -> {cfg.CHANNELS} ({selected_channel_names})")

    if cfg.CROP_MODE == "none":
        cfg.H = cfg.H_ORIG; cfg.W = cfg.W_ORIG
        print(f"No cropping - Using full dimensions: {cfg.H}x{cfg.W}")
    else:
        cfg.H = cfg.CROP_SIZE; cfg.W = cfg.CROP_SIZE
        print(f"Crop mode: {cfg.CROP_MODE} - Target dimensions: {cfg.H}x{cfg.W}")

    T_total = tr_full.shape[0]
    T_tr = int(T_total * 0.9)
    tr_train = tr_full[:T_tr]
    tr_val   = tr_full[T_tr:]
    print(f"Chrono split -> train={tr_train.shape}, val={tr_val.shape}, test(full file)={te_full.shape}")

    print("Applying channel selection...")
    tr_train = select_channels(tr_train, cfg.CHANNELS)
    tr_val   = select_channels(tr_val,   cfg.CHANNELS)
    te       = select_channels(te_full,  cfg.CHANNELS)

    print(f"Applying {cfg.CROP_MODE} cropping...")
    tr_train_p = crop_data(tr_train, cfg.CROP_MODE, cfg.CROP_SIZE, seed=cfg.SEED)
    tr_val_p   = crop_data(tr_val,   cfg.CROP_MODE, cfg.CROP_SIZE, seed=cfg.SEED)
    te_p       = crop_data(te,       cfg.CROP_MODE, cfg.CROP_SIZE, seed=cfg.SEED)

    Xtr,  ytr  = sliding_windows(tr_train_p, cfg.P_IN)
    Xval, yval = sliding_windows(tr_val_p,   cfg.P_IN)
    Xte,  yte  = sliding_windows(te_p,       cfg.P_IN)
    print(f"Sliding windows - Train: X{Xtr.shape}, y{ytr.shape}")
    print(f"Sliding windows - Val:   X{Xval.shape}, y{yval.shape}")
    print(f"Sliding windows - Test:  X{Xte.shape}, y{yte.shape}")

    min_np, max_np = norm_fit(tr_train_p)
    min_v = torch.from_numpy(min_np.astype(np.float32)).to(cfg.DEVICE)
    max_v = torch.from_numpy(max_np.astype(np.float32)).to(cfg.DEVICE)

    print("Normalization stats per channel (Min–Max, TRAIN split):")
    min_s = min_v.view(-1).detach().cpu().numpy()
    max_s = max_v.view(-1).detach().cpu().numpy()
    for i, ch_idx in enumerate(cfg.CHANNELS):
        ch_name = selected_channel_names[i]
        rng = max_s[i] - min_s[i]
        print(f"  Ch{ch_idx} ({ch_name}): Min={min_s[i]:.3f}, Max={max_s[i]:.3f}, Range={rng:.3f}")

    ds_tr  = WinDataset(Xtr,  ytr)
    ds_val = WinDataset(Xval, yval)
    ds_te  = WinDataset(Xte,  yte)

    dl_tr  = DataLoader(ds_tr,  batch_size=cfg.BATCH, shuffle=True,  num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    dl_val = DataLoader(ds_val, batch_size=cfg.BATCH, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    dl_te  = DataLoader(ds_te,  batch_size=cfg.BATCH, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)

    print(f"Creating STHAC: H={cfg.H}, W={cfg.W}, C={cfg.C}, P={cfg.P_IN}")
    model = STHAC(cfg.H, cfg.W, cfg.C, cfg.P_IN).to(cfg.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LR, weight_decay=cfg.WD
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.EPOCHS)

    crop_str = f"{cfg.CROP_MODE}_{cfg.CROP_SIZE}" if cfg.CROP_MODE != "none" else "full"
    if len(cfg.CHANNELS) == 1:
        channel_str = f"ch{cfg.CHANNELS[0]}"
    elif len(cfg.CHANNELS) == cfg.C_ORIG:
        channel_str = "allch"
    else:
        channel_str = f"ch{'_'.join(map(str, cfg.CHANNELS))}"

    model_name = f"sthac_{crop_str}_{channel_str}.pt"
    best_path = os.path.join(cfg.SAVE_DIR, model_name)

    best_rmse = 1e9
    print("\n=== Starting Training ===")
    print(f"Model will be saved as: {model_name}")

    for ep in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, dl_tr, min_v, max_v, optim)
        va_loss, va_metrics = eval_model(model, dl_val, min_v, max_v)
        scheduler.step()

        rmse_g = va_metrics["global"]["rmse"]
        r2_g   = va_metrics["global"]["r2"]

        if rmse_g < best_rmse:
            best_rmse = rmse_g
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "min": min_v.cpu().numpy(),
                "max": max_v.cpu().numpy(),
                "channels": cfg.CHANNELS,
                "channel_names": selected_channel_names
            }, best_path)

        print(f"[Ep {ep:03d}] train={tr_loss:.4f} | val={va_loss:.4f} | "
              f"RMSEg={rmse_g:.3f} | R2g={r2_g:.3f} | "
              f"lr={optim.param_groups[0]['lr']:.2e} | time={time.time()-t0:.1f}s")

    print(f"\nBest model saved to: {best_path}")
    print(f"Best RMSE (on val): {best_rmse:.4f}")

    print("\n=== Final Evaluation ===")
    ckpt = torch.load(best_path, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_metrics = eval_model(model, dl_te, min_v, max_v)

    print("\nSaving full TEST predictions and ground truth as .npy ...")
    model.eval()
    all_pred_te, all_true_te = [], []
    with torch.no_grad():
        for X_batch, y_batch in dl_te:
            X_batch = to_device(X_batch)
            y_batch = to_device(y_batch)

            cpd_hint = cpd_score(X_batch) if cfg.USE_CPD_ATT else None

            Xn_batch = norm_apply(X_batch, min_v, max_v)
            yp_batch, _ = model(Xn_batch, cpd_hint=cpd_hint)
            ypd_batch = denorm(yp_batch, min_v, max_v)

            all_pred_te.append(ypd_batch.cpu().numpy())
            all_true_te.append(y_batch.cpu().numpy())

    y_pred_test = np.concatenate(all_pred_te, axis=0)
    y_true_test = np.concatenate(all_true_te, axis=0)

    npy_pred_path = best_path.replace(".pt", "_y_pred_test.npy")
    npy_true_path = best_path.replace(".pt", "_y_true_test.npy")
    np.save(npy_pred_path, y_pred_test)
    np.save(npy_true_path, y_true_test)
    print(f"Saved TEST prediction array to: {npy_pred_path}")
    print(f"Saved TEST ground-truth array to: {npy_true_path}")

    global_metrics = test_metrics.get("global", {})
    per_channel = test_metrics.get("per_channel", None)

    print("\n=== Final Metrics (Test set) ===")
    print("Global metrics:")
    for k in ["rmse", "mae", "smape", "r2"]:
        if k in global_metrics:
            print(f"  {k.upper()}: {global_metrics[k]:.4f}")

    print("\nPer-channel metrics:")
    if per_channel is not None and all(m in per_channel for m in ["rmse", "mae", "smape", "r2"]):
        for i, (ch_idx, ch_name) in enumerate(zip(cfg.CHANNELS, selected_channel_names)):
            rmse_i  = per_channel["rmse"][i]
            mae_i   = per_channel["mae"][i]
            smape_i = per_channel["smape"][i]
            r2_i    = per_channel["r2"][i]
            print(f"  [{ch_idx}] {ch_name}: RMSE={rmse_i:.4f}, MAE={mae_i:.4f}, SMAPE={smape_i:.4f}, R2={r2_i:.4f}")
    else:
        for ch_idx, ch_name in zip(cfg.CHANNELS, selected_channel_names):
            print(f"  [{ch_idx}] {ch_name}: RMSE={global_metrics.get('rmse', float('nan')):.4f}, "
                  f"MAE={global_metrics.get('mae', float('nan')):.4f}, "
                  f"SMAPE={global_metrics.get('smape', float('nan')):.4f}, "
                  f"R2={global_metrics.get('r2', float('nan')):.4f}")

    results = {
        "config": cfg.__dict__,
        "channels": cfg.CHANNELS,
        "channel_names": selected_channel_names,
        "best_rmse_val": best_rmse,
        "final_metrics": test_metrics
    }
    results_path = best_path.replace('.pt', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_path}")

if __name__ == "__main__":
    cfg.CHANNELS = [0]
    cfg.C = len(cfg.CHANNELS)
    main()
