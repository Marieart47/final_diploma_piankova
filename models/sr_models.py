"""Архитектуры для Super Resolution: SRCNN, ESPCN, SwinIR (lightweight)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SRCNN ──────────────────────────────────────────────────────────────────────

class SRCNN(nn.Module):
    """
    Super-Resolution CNN (Dong et al., ECCV 2014).
    Принимает LR-изображение, бикубически увеличивает его до нужного
    разрешения, затем применяет три свёрточных слоя.
    """

    def __init__(self, scale: int = 4, num_channels: int = 3):
        super().__init__()
        self.scale = scale
        self.upsample = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return self.conv3(x)


# ── ESPCN ──────────────────────────────────────────────────────────────────────

class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel CNN (Shi et al., CVPR 2016).
    Обрабатывает LR напрямую, субпиксельная свёртка выполняет апсемплинг
    в конце без промежуточного увеличения разрешения.
    """

    def __init__(self, scale: int = 4, num_channels: int = 3):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, num_channels * scale * scale, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        return self.pixel_shuffle(self.conv4(x))


# ── SwinIR (lightweight) ────────────────────────────────────────────────────────
#
# Упрощённая реализация по мотивам Liang et al., ICCV 2021.
# Параметры по умолчанию соответствуют варианту SwinIR-light
# (embed_dim=60, 4 RSTB-блока, 6 голов внимания, window_size=8).

def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) → (num_windows*B, ws, ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """(num_windows*B, ws, ws, C) → (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class _WindowAttention(nn.Module):
    """Оконный self-attention с относительными позиционными смещениями."""

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        # Таблица относительных смещений: (2*ws-1)^2 × num_heads
        self.rel_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

        coords = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
        )  # (2, ws, ws)
        coords_flat = coords.flatten(1)  # (2, ws^2)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, ws^2, ws^2)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_idx", rel.sum(-1))  # (ws^2, ws^2)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.rel_bias_table[self.rel_idx.view(-1)]
        bias = bias.view(self.window_size ** 2, self.window_size ** 2, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class _SwinBlock(nn.Module):
    """Один блок Swin Transformer с (опциональным) циклическим сдвигом."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 2.):
        super().__init__()
        self.shift_size = shift_size
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    @staticmethod
    def _compute_mask(H: int, W: int, ws: int, shift: int,
                      device: torch.device) -> torch.Tensor:
        mask = torch.zeros(1, H, W, 1, device=device)
        slices_h = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        slices_w = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for sh in slices_h:
            for sw in slices_w:
                mask[:, sh, sw, :] = cnt
                cnt += 1
        mw = _window_partition(mask, ws).view(-1, ws * ws)
        attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x

        x2d = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x2d = torch.roll(x2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = self._compute_mask(H, W, self.window_size, self.shift_size, x.device)
        else:
            mask = None

        wins = _window_partition(x2d, self.window_size).view(-1, self.window_size ** 2, C)
        wins = self.attn(wins, mask=mask)
        wins = wins.view(-1, self.window_size, self.window_size, C)
        x2d = _window_reverse(wins, self.window_size, H, W)

        if self.shift_size > 0:
            x2d = torch.roll(x2d, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x2d.view(B, L, C)
        x = x + self.mlp(self.norm2(x))
        return x


class _RSTB(nn.Module):
    """Residual Swin Transformer Block: несколько Swin-блоков + свёртка + skip."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: int = 8, mlp_ratio: float = 2.):
        super().__init__()
        self.blocks = nn.ModuleList([
            _SwinBlock(
                dim, num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        residual = x
        for blk in self.blocks:
            x = blk(x, H, W)
        B, L, C = x.shape
        x2d = x.transpose(1, 2).view(B, C, H, W)
        x2d = self.conv(x2d)
        return x2d.flatten(2).transpose(1, 2) + residual


class SwinIR(nn.Module):
    """
    Lightweight SwinIR для Super Resolution (Liang et al., ICCV 2021).
    Вариант SwinIR-light: embed_dim=60, 4 RSTB, window_size=8.
    """

    def __init__(
        self,
        scale: int = 4,
        num_channels: int = 3,
        embed_dim: int = 60,
        depths: tuple = (6, 6, 6, 6),
        num_heads: tuple = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.scale = scale
        self.window_size = window_size

        self.conv_first = nn.Conv2d(num_channels, embed_dim, 3, padding=1)

        self.layers = nn.ModuleList([
            _RSTB(embed_dim, depth=d, num_heads=h,
                  window_size=window_size, mlp_ratio=mlp_ratio)
            for d, h in zip(depths, num_heads)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        # Субпиксельный апсемплинг
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )
        self.conv_last = nn.Conv2d(embed_dim, num_channels, 3, padding=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _pad_to_window(self, x: torch.Tensor):
        """Дополняет до кратного window_size, возвращает оригинальные H, W."""
        _, _, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, H_orig, W_orig = self._pad_to_window(x)
        _, _, H, W = x.shape

        feat = self.conv_first(x)
        shortcut = feat

        seq = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        for layer in self.layers:
            seq = layer(seq, H, W)
        seq = self.norm(seq)

        feat = seq.transpose(1, 2).view(-1, feat.shape[1], H, W)
        feat = self.conv_after_body(feat) + shortcut

        out = self.conv_last(self.upsample(feat))
        # Убираем padding
        return out[:, :, :H_orig * self.scale, :W_orig * self.scale]


# ── Factory ────────────────────────────────────────────────────────────────────

def build_sr_model(arch: str, scale: int = 4) -> nn.Module:
    arch = arch.lower()
    if arch == "srcnn":
        return SRCNN(scale=scale)
    if arch == "espcn":
        return ESPCN(scale=scale)
    if arch == "swinir":
        return SwinIR(scale=scale)
    raise ValueError(f"Unknown SR arch: {arch!r}. Choose from srcnn, espcn, swinir.")
