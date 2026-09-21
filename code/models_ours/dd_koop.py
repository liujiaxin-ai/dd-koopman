"""Observation-conditioned delay-Doppler operator for multi-step CSI
prediction — the paper's reported architecture.

The prediction is

    pred = base + g1 * corr_mix + g2 * (corr_graph + corr_backcast) + g3 * corr_split

where ``base`` is the spectral extrapolation of the observed delay profile
(the exact on-grid answer, so the model starts at a certified predictor) and
the three gated corrections are learned residuals initialised to zero. With
``use_graph=false`` (the reported configuration, `tdd_caplong.yaml`) the
graph and backcast channels are disabled and the model reduces to ``base``
plus the two gated corrections of Eq. (2) in the paper.
Components are adapted from CSI-4CAST (denoiser, reweighting, delay
decomposition), ChannelMamba (shared dynamics block), MamKO (selective
Koopman scan), SEED (signed latent graph), STEMGNN (Chebyshev spectral
block, backcast split) and FreDF (spectral objective); see the paper's
related-work section for the citations.

Registered predictor classes: DD_KOOP_TDD, DD_KOOP_FDD.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.cp.config.config import ExperimentConfig
from src.cp.models.common.base import BaseCSIModel
from src.cp.loss.loss import NMSELoss

class SharedDynamicsBlock(nn.Module):

    def __init__(self, channels: int, d_conv: int = 4, dropout: float = 0.0):
        super().__init__()

        self.norm = nn.LayerNorm(2 * channels)
        self.conv = nn.Conv1d(2 * channels, 2 * channels, kernel_size=d_conv, groups=2 * channels, padding=d_conv - 1)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pair = torch.view_as_real(x)
        flat = pair.reshape(*pair.shape[:2], -1)
        h = self.norm(flat).transpose(1, 2)
        h = self.conv(h)[:, :, : flat.shape[1]].transpose(1, 2)
        h = self.drop(self.act(h))
        mixed = flat + h
        return torch.view_as_complex(mixed.reshape(*pair.shape).contiguous())

class SignedLatentGraph(nn.Module):

    def __init__(
        self,
        histogram_dim: int,
        pred_len: int,
        num_taps: int,
        hidden: int = 64,
        order: int = 4,
        tap_encoding: bool = False,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.order = order
        self.use_tap_encoding = tap_encoding
        if tap_encoding:

            self.tap_encoding = SinusoidalTapEncoding(num_taps, hidden)
        self.embed = nn.Linear(histogram_dim, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.cheb_weight = nn.Parameter(torch.randn(order) * 0.1)

        self.glu_real_left = nn.Linear(hidden, hidden)
        self.glu_real_right = nn.Linear(hidden, hidden)
        self.glu_imag_left = nn.Linear(hidden, hidden)
        self.glu_imag_right = nn.Linear(hidden, hidden)
        self.forecast = nn.Linear(2 * hidden, pred_len * 2)
        self.backcast = nn.Linear(2 * hidden, histogram_dim)
        self.stability_loss = torch.tensor(0.0)

    def _adjacency(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.query(tokens)
        k = self.key(tokens)
        logits = torch.einsum("bld,bmd->blm", q, k) / math.sqrt(tokens.shape[-1])
        logits = mask_topk(logits, ratio=0.5)
        adj = torch.tanh(logits)

        adj = adj / adj.abs().sum(dim=-1, keepdim=True).clamp_min(1e-3)
        self.stability_loss = torch.var(adj, dim=2, unbiased=False).mean()
        return adj

    def forward(self, doppler: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, hist_len, taps = doppler.shape
        features = torch.view_as_real(doppler).permute(0, 2, 1, 3).reshape(batch, taps, -1)
        tokens = self.embed(features)
        if self.use_tap_encoding:
            tokens = self.tap_encoding(tokens)

        adj = self._adjacency(tokens)
        degree = adj.abs().sum(dim=-1)
        d_inv = torch.diag_embed(1.0 / (degree.sqrt() + 1e-3))
        identity = torch.eye(taps, device=adj.device, dtype=adj.dtype).expand(batch, taps, taps)
        laplacian = identity - d_inv @ adj @ d_inv

        scale = laplacian.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        laplacian = laplacian / scale

        basis = SelectiveKoopmanScan._chebyshev_basis(laplacian, self.order)
        coefficients = torch.einsum("k,kblm,bmd->bld", self.cheb_weight, basis, tokens)

        spectrum = torch.fft.fft(coefficients, dim=-1)
        real = self.glu_real_left(spectrum.real) * torch.sigmoid(self.glu_real_right(spectrum.real))
        imag = self.glu_imag_left(spectrum.imag) * torch.sigmoid(self.glu_imag_right(spectrum.imag))
        spectral = torch.fft.ifft(torch.complex(real, imag), dim=-1).real
        fused = torch.cat([real, spectral], dim=-1)

        correction = self.forecast(fused).view(batch, taps, self.pred_len, 2)
        correction = torch.view_as_complex(correction.contiguous()).permute(0, 2, 1)

        backcast = self.backcast(fused).view(batch, taps, hist_len, 2)
        backcast = torch.view_as_complex(backcast.contiguous()).permute(0, 2, 1)
        return correction, backcast

class SpectralObjective(nn.Module):

    def __init__(self, pred_len: int, auxi_lambda: float = 0.5, mode: str = "complex"):
        super().__init__()
        self.auxi_lambda = auxi_lambda
        self.mode = mode
        self.rec = NMSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_c = torch.view_as_complex(pred.reshape(*pred.shape[:-1], -1, 2).contiguous())
        target_c = torch.view_as_complex(target.reshape(*target.shape[:-1], -1, 2).contiguous())
        residual = torch.fft.fft(pred_c, dim=1) - torch.fft.fft(target_c, dim=1)
        if self.mode == "complex-phase":
            auxi = residual.angle().abs().mean()
        else:
            auxi = residual.abs().pow(2).mean()
        return self.rec(pred, target) + self.auxi_lambda * auxi, auxi

def mask_topk(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    if ratio >= 1.0:
        return x
    flat = x.abs().flatten()
    keep = max(1, int(flat.numel() * ratio))
    threshold = flat.topk(keep, largest=True).values.min()
    return x * (x.abs() >= threshold)

class SelectiveKoopmanScan(nn.Module):

    def __init__(
        self,
        hist_len: int,
        pred_len: int,
        num_taps: int,
        state_dim: int = 8,
        d_conv: int = 4,
    ):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.num_taps = num_taps
        self.state_dim = state_dim
        features = 2 * hist_len

        self.z_project = nn.Linear(features, state_dim)

        self.context_conv = nn.Conv1d(2, 2, kernel_size=d_conv, groups=2, padding=d_conv - 1)
        self.generator = nn.Linear(features, pred_len * (1 + state_dim + 2 * state_dim))
        self.dt_project = nn.Linear(1, state_dim)

        self.raw_A = nn.Parameter(torch.randn(state_dim) * 0.1 - 2.5)

    @staticmethod
    def _chebyshev_basis(laplacian: torch.Tensor, order: int = 4) -> torch.Tensor:
        batch, n, _ = laplacian.shape
        eye = torch.eye(n, device=laplacian.device, dtype=laplacian.dtype).expand(batch, n, n)
        terms = [eye, laplacian]
        for _ in range(2, order):
            terms.append(2 * torch.matmul(laplacian, terms[-1]) - terms[-2])
        return torch.stack(terms, dim=0)

    def forward(self, doppler: torch.Tensor) -> torch.Tensor:
        batch, hist_len, taps = doppler.shape
        flat = torch.view_as_real(doppler).permute(0, 2, 1, 3).reshape(batch * taps, hist_len, 2)

        state = torch.tanh(self.z_project(flat.reshape(batch * taps, -1)))
        sequence = self.context_conv(flat.transpose(1, 2))[:, :, :hist_len]
        sequence = F.silu(sequence).transpose(1, 2).reshape(batch * taps, -1)
        generated = self.generator(sequence).view(
            batch * taps, self.pred_len, 1 + self.state_dim + 2 * self.state_dim
        )

        A = -(F.softplus(self.raw_A) + 1e-6)
        outputs = []
        for step in range(self.pred_len):
            raw_dt = generated[:, step, :1]
            B_term = generated[:, step, 1 : 1 + self.state_dim]
            C_term = generated[:, step, 1 + self.state_dim :].view(batch * taps, self.state_dim, 2)
            dt = F.softplus(self.dt_project(raw_dt))
            z = dt * A
            dA = torch.exp(z)

            dB = torch.where(z.abs() > 1e-6, torch.expm1(z) / A, dt) * B_term
            state = dA * state + dB

            outputs.append(torch.einsum("bn,bnp->bp", state, C_term))

        stacked = torch.stack(outputs, dim=1)
        stacked = stacked.view(batch, taps, self.pred_len, 2)
        return torch.view_as_complex(stacked.contiguous()).permute(0, 2, 1)

class ResidualDenoiser(nn.Module):

    def __init__(self, num_filters: int = 3, kernel: int = 3, post: bool = True):
        super().__init__()
        widths = [2 ** (i + 1) for i in range(num_filters)]
        pad = (kernel - 1) // 2

        self.encoder = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(widths[i], widths[i + 1], kernel, padding=pad),
                    nn.BatchNorm2d(widths[i + 1]),
                )
            )
        self.decoder = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(widths[i], widths[i - 1], kernel, padding=pad),
                    nn.BatchNorm2d(widths[i - 1]),
                )
            )
        self.post = post
        if post:
            self.post_processor = nn.Conv1d(16, 16, kernel, padding=pad)
            self.post_bn = nn.BatchNorm1d(16)

    def forward(self, x: torch.Tensor, hist_len: int) -> torch.Tensor:
        batch = x.shape[0]
        noisy = x
        h = x.reshape(batch, hist_len, -1, 2).permute(0, 3, 1, 2)
        for layer in self.encoder:
            h = F.relu(layer(h))
        for layer in self.decoder:
            h = F.relu(layer(h))
        h = h.permute(0, 2, 3, 1).reshape(batch, hist_len, -1)
        if self.post:
            h = self.post_bn(self.post_processor(h))
        return noisy - h

class InstanceNormalization(nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(2 * num_features))
            self.bias = nn.Parameter(torch.zeros(2 * num_features))
        self._mean = None
        self._std = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        out = (x - self._mean) / self._std
        if self.affine:
            out = out * self.weight + self.bias
        return out

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None:
            return x
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps * self.eps)
        return x * self._std + self._mean

class LearnableDelayDecomposition(nn.Module):

    def __init__(self, num_taps: int, init_scale: float = 8.0):
        super().__init__()
        taps = torch.arange(num_taps, dtype=torch.float32)
        prior = 1.0 / (1.0 + (taps / init_scale) ** 2)
        prior = prior.clamp(1e-3, 1 - 1e-3)
        self.mask = nn.Parameter(torch.log(prior / (1 - prior)))

    def forward(self, delay: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.mask)
        structured = gate * delay
        return structured, delay - structured

class AdaptiveReweighting(nn.Module):

    def __init__(self, dim: int, hidden: int = 128, operation: str = "multiply"):
        super().__init__()
        if operation not in ("multiply", "add"):
            raise ValueError(f"unsupported ARL operation: {operation}")
        self.operation = operation

        self.dim = dim
        width = 2 * dim
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pair = torch.view_as_real(x)
        flat = pair.reshape(*pair.shape[:2], -1)
        weight = self.mlp(flat)
        out = weight * flat if self.operation == "multiply" else weight + flat
        out = self.norm(out)
        return torch.view_as_complex(out.reshape(*pair.shape).contiguous())

class SinusoidalTapEncoding(nn.Module):

    def __init__(self, num_taps: int, dim: int):
        super().__init__()
        position = torch.arange(num_taps, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        encoding = torch.zeros(num_taps, dim)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.encoding[:, : tokens.shape[1], : tokens.shape[2]]

class SpectralKoopmanOperator(nn.Module):

    def __init__(
        self,
        hist_len: int = 16,
        pred_len: int = 4,
        num_taps: int = 64,
        hidden: int = 64,
        input_conditioned_poles: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.num_taps = num_taps
        self.input_conditioned_poles = input_conditioned_poles

        bins = torch.arange(hist_len, dtype=torch.float64)
        last = hist_len - 1

        initial = torch.exp(1j * 2 * math.pi * bins * last / hist_len) / hist_len
        init = initial.unsqueeze(0).repeat(pred_len, 1)
        self.gain_real = nn.Parameter(init.real.float())
        self.gain_imag = nn.Parameter(init.imag.float())

        self.raw_radius = nn.Parameter(torch.full((hist_len,), 2.5))

        self.raw_delta = nn.Parameter(torch.zeros(hist_len))

        self.context_norm = nn.LayerNorm(hist_len)
        self.modulator = nn.Sequential(
            nn.Linear(hist_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * pred_len * hist_len),
        )
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

        self.pole_modulator = nn.Sequential(
            nn.Linear(hist_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, hist_len),
        )
        nn.init.zeros_(self.pole_modulator[-1].weight)
        nn.init.zeros_(self.pole_modulator[-1].bias)

    def poles(self) -> torch.Tensor:
        bins = torch.arange(self.hist_len, device=self.raw_radius.device, dtype=torch.float32)
        delta = 0.5 * torch.tanh(self.raw_delta)
        radius = 0.5 * (1.0 + torch.tanh(self.raw_radius))
        angle = 2 * math.pi * (bins + delta) / self.hist_len
        return torch.polar(radius, angle)

    def pole_angles(self, doppler: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bins = torch.arange(self.hist_len, device=doppler.device, dtype=torch.float32)
        base = 2 * math.pi * (bins + 0.5 * torch.tanh(self.raw_delta)) / self.hist_len
        radius = 0.5 * (1.0 + torch.tanh(self.raw_radius))
        if not self.input_conditioned_poles:
            return base, radius
        power = torch.log(doppler.abs().pow(2).mean(dim=-1) + 1e-8)
        offset = 0.5 * torch.tanh(self.pole_modulator(self.context_norm(power)))
        return base.unsqueeze(0) + offset * (2 * math.pi / self.hist_len), radius

    def forward(self, doppler: torch.Tensor) -> torch.Tensor:
        batch, hist, _ = doppler.shape
        horizon = torch.arange(1, self.pred_len + 1, device=doppler.device, dtype=torch.float32)
        angle, radius = self.pole_angles(doppler)

        magnitude = radius.unsqueeze(-2) ** horizon.reshape(1, -1, 1)
        advance = torch.polar(magnitude, angle.unsqueeze(-2) * horizon.reshape(1, -1, 1))
        base = torch.complex(self.gain_real, self.gain_imag).unsqueeze(0) * advance

        power = torch.log(doppler.abs().pow(2).mean(dim=-1) + 1e-8)
        modulation = self.modulator(self.context_norm(power)).view(batch, self.pred_len, hist, 2)
        bounded = 1.0 + torch.tanh(torch.view_as_complex(modulation.contiguous()))
        gain = base.unsqueeze(-1) * bounded.unsqueeze(-1)
        return torch.einsum("bhl,bphl->bpl", doppler, gain)

    @torch.no_grad()
    def diagnostics(self) -> dict:
        poles = self.poles().detach().cpu()
        return {
            "radius_min": float(poles.abs().min()),
            "radius_max": float(poles.abs().max()),
            "angle_min": float(poles.angle().min()),
            "angle_max": float(poles.angle().max()),
        }

    @torch.no_grad()
    def warm_start(self, weight: torch.Tensor) -> None:
        poles = self.poles().to(weight.device)
        horizon = torch.arange(1, self.pred_len + 1, device=weight.device, dtype=torch.float32)
        advance = poles.unsqueeze(0) ** horizon.unsqueeze(1)
        gain = weight / advance
        self.gain_real.copy_(gain.real)
        self.gain_imag.copy_(gain.imag)

class RadiusSpectralKoopmanOperator(SpectralKoopmanOperator):

    def __init__(
        self,
        hist_len: int = 16,
        pred_len: int = 4,
        num_taps: int = 64,
        hidden: int = 64,
        *args,
        **kwargs,
    ):
        kwargs.pop("input_conditioned_poles", None)
        super().__init__(
            hist_len=hist_len,
            pred_len=pred_len,
            num_taps=num_taps,
            hidden=hidden,
            input_conditioned_poles=True,
            **kwargs,
        )

        self.pole_modulator = nn.Sequential(
            nn.Linear(hist_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * hist_len),
        )
        nn.init.zeros_(self.pole_modulator[-1].weight)
        nn.init.zeros_(self.pole_modulator[-1].bias)

    def pole_angles(self, doppler: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bins = torch.arange(self.hist_len, device=doppler.device, dtype=torch.float32)
        base = 2 * math.pi * (bins + 0.5 * torch.tanh(self.raw_delta)) / self.hist_len
        power = torch.log(doppler.abs().pow(2).mean(dim=-1) + 1e-8)
        generated = self.pole_modulator(self.context_norm(power))
        angle_offset = 0.5 * torch.tanh(generated[:, : self.hist_len])
        radius_offset = torch.tanh(generated[:, self.hist_len :])
        angle = base.unsqueeze(0) + angle_offset * (2 * math.pi / self.hist_len)
        radius = 0.5 * (1.0 + torch.tanh(self.raw_radius.unsqueeze(0) + radius_offset))
        return angle, radius

def least_squares_spectrum_map(
    hist: torch.Tensor, pred: torch.Tensor, taps: int, ridge: float = 1e-6
) -> torch.Tensor:
    slots = hist.shape[2]
    horizon = pred.shape[2]
    observed = torch.fft.fft(torch.fft.ifft(hist, dim=-1)[..., :taps], dim=2)
    target = torch.fft.ifft(pred, dim=-1)[..., :taps]
    design = observed.permute(0, 1, 3, 2).reshape(-1, slots)
    future = target.permute(0, 1, 3, 2).reshape(-1, horizon)

    gram = design.conj().T @ design
    rhs = design.conj().T @ future
    scale = torch.diagonal(gram).real.sum() / slots
    gram = gram + (ridge * scale + 1e-12) * torch.eye(
        slots, dtype=design.dtype, device=design.device
    )
    return torch.linalg.solve(gram, rhs).T.contiguous()

class DDKoopNet(nn.Module):

    def __init__(
        self,
        hist_len: int = 16,
        pred_len: int = 4,
        num_subcarriers: int = 300,
        num_delay_taps: int = 64,
        state_dim: int = 8,
        hidden: int = 64,
        order: int = 4,
        arl_hidden: int = 128,
        arl_operation: str = "multiply",
        denoise_filters: int = 3,
        use_revin: bool = False,
        use_delay_decomposition: bool = False,
        use_tap_encoding: bool = False,
        use_denoiser: bool = True,
        use_graph: bool = True,
        input_conditioned_poles: bool = True,
        use_radius_conditioning: bool = True,
        freeze_corrections: bool = False,
        learned_basis: bool = False,
        decomposition_scale: float = 8.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.num_subcarriers = num_subcarriers
        self.num_delay_taps = num_delay_taps
        features = 2 * hist_len

        self.use_learned_basis = learned_basis
        if learned_basis:
            self.basis_h = nn.Parameter(torch.zeros(hist_len, hist_len))

        self.use_revin = use_revin
        if use_revin:
            self.instance_norm = InstanceNormalization(num_subcarriers)

        self.use_denoiser = use_denoiser
        if use_denoiser:
            self.denoiser = ResidualDenoiser(num_filters=denoise_filters)
            with torch.no_grad():

                self.denoiser.post_processor.weight.zero_()
                self.denoiser.post_processor.bias.zero_()

        operator = RadiusSpectralKoopmanOperator if use_radius_conditioning else SpectralKoopmanOperator
        self.operator = operator(
            hist_len=hist_len,
            pred_len=pred_len,
            num_taps=num_delay_taps,
            input_conditioned_poles=input_conditioned_poles,
        )

        self.arl = AdaptiveReweighting(num_delay_taps, hidden=arl_hidden, operation=arl_operation)

        self.arl.norm = nn.Identity()
        with torch.no_grad():
            self.arl.mlp[-1].weight.zero_()
            self.arl.mlp[-1].bias.fill_(1.0)

        self.shared_mixer = SharedDynamicsBlock(num_delay_taps)
        with torch.no_grad():
            self.shared_mixer.conv.weight.zero_()
            self.shared_mixer.conv.bias.zero_()

        self.use_graph = use_graph
        if use_graph:
            self.residual = SignedLatentGraph(
                features, pred_len, num_delay_taps, hidden=hidden, order=order,
                tap_encoding=use_tap_encoding,
            )
            with torch.no_grad():
                self.residual.forecast.weight.zero_()
                self.residual.forecast.bias.zero_()
                self.residual.backcast.weight.zero_()
                self.residual.backcast.bias.zero_()

        self.use_delay_decomposition = use_delay_decomposition
        if use_delay_decomposition:
            self.decomposition = LearnableDelayDecomposition(num_delay_taps, decomposition_scale)

        self.gate_mix = nn.Parameter(torch.zeros(()))
        self.gate_graph = nn.Parameter(torch.zeros(()))
        self.gate_backcast = nn.Parameter(torch.zeros(()))
        self.gate_split = nn.Parameter(torch.zeros(()))
        if freeze_corrections:

            for gate in (self.gate_mix, self.gate_graph, self.gate_backcast, self.gate_split):
                gate.requires_grad_(False)

    def _lift(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, features = x.shape
        freq = torch.view_as_complex(x.reshape(batch, time_steps, features // 2, 2).contiguous())
        taps = min(self.num_delay_taps, freq.shape[-1])
        return torch.fft.ifft(freq, dim=-1)[..., :taps]

    def basis_matrix(self) -> torch.Tensor:
        lower = torch.tril(self.basis_h, diagonal=-1)
        upper = lower.conj().transpose(-1, -2)
        hermitian = lower + upper + torch.diag(torch.diagonal(self.basis_h))
        return torch.matrix_exp(1j * hermitian.to(torch.complex64))

    def spectrum_of(self, delay: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft(delay, dim=1)
        if self.use_learned_basis:
            spectrum = torch.einsum("tm,bml->btl", self.basis_matrix(), spectrum)
        return spectrum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, features = x.shape
        if self.use_revin:
            x = self.instance_norm.normalize(x)
        if self.use_denoiser:
            x = self.denoiser(x, time_steps)

        delay = self._lift(x)
        spectrum = self.spectrum_of(delay)
        base = self.operator(spectrum)
        predicted = base

        mixed = self.shared_mixer(self.arl(delay))
        predicted = predicted + self.gate_mix * (self.operator(self.spectrum_of(mixed)) - base)

        if self.use_graph:
            correction, backcast = self.residual(spectrum)
            backcast_term = self.operator(self.spectrum_of(delay - backcast)) - base
            predicted = predicted + self.gate_graph * correction + self.gate_backcast * backcast_term

        if self.use_delay_decomposition:
            _structured, residual_part = self.decomposition(delay)
            predicted = predicted + self.gate_split * (
                self.operator(self.spectrum_of(residual_part)) - base
            )

        taps = predicted.shape[-1]
        if taps < self.num_subcarriers:
            predicted = torch.cat(
                [predicted, predicted.new_zeros(batch, self.pred_len, self.num_subcarriers - taps)],
                dim=-1,
            )
        h_pred = torch.fft.fft(predicted, dim=-1)[..., : self.num_subcarriers]
        pred = torch.view_as_real(h_pred).reshape(batch, self.pred_len, self.num_subcarriers * 2)
        return self.instance_norm.denormalize(pred) if self.use_revin else pred

class _DDKoopLightning(BaseCSIModel):

    MODEL_NAME = "DDK8"

    def __init__(self, config: ExperimentConfig, *args, **kwargs):
        super().__init__(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
        )
        self.name = self.MODEL_NAME
        self.is_separate_antennas = config.model.is_separate_antennas
        self.save_hyperparameters({"model": config.model})
        self.model = DDKoopNet(**config.model.params)
        self.objective = SpectralObjective(
            pred_len=config.model.params.get("pred_len", 4),
            auxi_lambda=config.model.params.get("auxi_lambda", 0.5),
            mode=config.model.params.get("auxi_mode", "complex"),
        )
        self.metric = NMSELoss()

        self.sample_weight_power = float(
            config.model.params.get("sample_weight_power", 0.0)
        )

    def __str__(self):
        return self.name

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        hist, target = batch
        pred = self(hist)
        loss, auxi = self.objective(pred, target)
        if self.sample_weight_power > 0.0:

            error = (pred - target).pow(2).flatten(1).sum(-1)
            power = target.pow(2).flatten(1).sum(-1) + 1e-12
            per_sample = error / power
            weight = per_sample.detach().clamp(min=1e-4).pow(-self.sample_weight_power)
            weight = weight / weight.mean()
            auxiliary = loss - self.objective.rec(pred, target)
            loss = (weight * per_sample).mean() + auxiliary
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_auxi", auxi, on_step=False, on_epoch=True, logger=True)
        stability = getattr(getattr(self.model, "residual", None), "stability_loss", None)
        if stability is not None:
            loss = loss + 1e-3 * stability
        return loss

    def validation_step(self, batch, batch_idx):
        hist, target = batch
        pred = self(hist)
        loss = self.metric(pred, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

class DD_KOOP_TDD(_DDKoopLightning):

    MODEL_NAME = "DDK8"

class DD_KOOP_FDD(_DDKoopLightning):

    MODEL_NAME = "DDK8"

def _register() -> None:
    from src.cp.models import PREDICTORS

    PREDICTORS.DD_KOOP_TDD = DD_KOOP_TDD
    PREDICTORS.DD_KOOP_FDD = DD_KOOP_FDD

_register()
