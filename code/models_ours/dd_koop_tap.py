"""Tap-truncation variant of the delay-Doppler operator.

Extends ``dd_koop.py`` with a tap transformer and Doppler shrinkage on the
tap axis; produces the DD_KOOP_TAP runs behind the T300C64 row of the
released robustness table. Registered predictor classes: DD_KOOP_TAP_TDD,
DD_KOOP_TAP_FDD.
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

class TapTransformer(nn.Module):

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.0,
        ffn_width: int | None = None,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ffn_width or 4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(tokens))

class DopplerShrinkage(nn.Module):

    def __init__(self, hist_len: int, floor: float = 1e-3) -> None:
        super().__init__()
        self.hist_len = hist_len
        self.floor = floor
        self.log_psd = nn.Parameter(torch.zeros(hist_len))

    def psd(self) -> torch.Tensor:
        return torch.softmax(self.log_psd, dim=0)

    def lag_profile(self) -> torch.Tensor:
        psd = self.psd()
        bins = torch.arange(self.hist_len, device=psd.device, dtype=psd.dtype)
        angle = 2.0 * math.pi * torch.outer(bins, bins) / self.hist_len
        return (torch.cos(angle) * psd[None, :]).sum(dim=1)

    def forward(self, delay: torch.Tensor) -> torch.Tensor:
        batch, length, taps = delay.shape
        prior = self.lag_profile()
        cov = torch.stack(
            [
                (delay[:, lag:] * torch.conj(delay[:, : length - lag])).mean(dim=1).real
                for lag in range(length)
            ],
            dim=1,
        )
        d11 = (prior * prior).sum()
        d12 = prior[0]
        d22 = torch.ones_like(d11)
        b1 = (prior[None, :, None] * cov).sum(dim=1)
        b2 = cov[:, 0, :]
        determinant = d11 * d22 - d12 * d12 + 1e-12
        scale = (b1 * d22 - b2 * d12) / determinant
        noise = (b2 * d11 - b1 * d12) / determinant
        total = cov[:, 0, :]

        noise = torch.maximum(noise, self.floor * total)
        scale = torch.minimum(torch.clamp(scale, min=0.0), total)
        psd = self.psd()
        signal = scale[:, None, :] * psd[None, :, None]
        gain = signal / (signal + noise[:, None, :] + 1e-12)
        spectrum = torch.fft.fft(delay, dim=1)
        return torch.fft.ifft(spectrum * gain, dim=1)

class DDKoopTapNet(nn.Module):

    def __init__(
        self,
        hist_len: int = 16,
        pred_len: int = 4,
        num_subcarriers: int = 300,
        num_delay_taps: int = 64,
        corr_taps: int = 0,
        d_model: int = 160,
        depth: int = 4,
        heads: int = 8,
        ffn_scale: float = 2.0,
        dropout: float = 0.1,
        use_denoiser: bool = True,
        denoise_filters: int = 3,
        use_revin: bool = True,
        use_delay_decomposition: bool = True,
        use_tap_encoding: bool = True,
        use_radius_conditioning: bool = True,
        input_conditioned_poles: bool = True,
        correction_scale: float = 1.0,
        decomposition_scale: float = 8.0,
        arl_hidden: int = 128,
        arl_operation: str = "multiply",
        gated_correction: bool = False,
        persistence_blend: bool = False,
        blend_source: str = "last",
        blend_coherence_init: bool = False,
        doppler_shrink: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.num_subcarriers = num_subcarriers
        self.num_delay_taps = num_delay_taps

        self.corr_taps = corr_taps if corr_taps > 0 else num_delay_taps
        self.correction_scale = correction_scale
        self.use_doppler_shrink = doppler_shrink
        if doppler_shrink:
            self.shrink = DopplerShrinkage(hist_len)
        features = 2 * hist_len

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

        self.arl = AdaptiveReweighting(self.corr_taps, hidden=arl_hidden, operation=arl_operation)
        self.arl.norm = nn.Identity()
        with torch.no_grad():
            self.arl.mlp[-1].weight.zero_()
            self.arl.mlp[-1].bias.fill_(1.0)
        self.shared_mixer = SharedDynamicsBlock(self.corr_taps)
        with torch.no_grad():
            self.shared_mixer.conv.weight.zero_()
            self.shared_mixer.conv.bias.zero_()
        self.gate_mix = nn.Parameter(torch.zeros(()))

        self.embed = nn.Linear(features, d_model)
        self.tap_encoding = SinusoidalTapEncoding(num_delay_taps, d_model) if use_tap_encoding else None
        self.encoder = TapTransformer(
            d_model, depth, heads, dropout, ffn_width=int(ffn_scale * d_model)
        )
        self.head = nn.Linear(d_model, pred_len * 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.use_delay_decomposition = use_delay_decomposition
        if use_delay_decomposition:
            self.decomposition = LearnableDelayDecomposition(self.corr_taps, decomposition_scale)
            self.split_head = nn.Linear(features, pred_len * 2)
            nn.init.zeros_(self.split_head.weight)
            nn.init.zeros_(self.split_head.bias)
        self.gate_split = nn.Parameter(torch.zeros(()))

        self.use_correction_gate = gated_correction
        if gated_correction:
            self.gate_head = nn.Sequential(
                nn.Linear(hist_len, hist_len),
                nn.GELU(),
                nn.Linear(hist_len, 1),
            )
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.zeros_(self.gate_head[-1].bias)

        self.use_persistence_blend = persistence_blend

        self.blend_source = blend_source
        if persistence_blend:

            self.use_coherence_gate = blend_coherence_init
            in_features = hist_len + (1 if blend_coherence_init else 0)
            self.blend_head = nn.Sequential(
                nn.Linear(in_features, hist_len),
                nn.GELU(),
                nn.Linear(hist_len, 1),
            )
            nn.init.zeros_(self.blend_head[-1].weight)
            nn.init.zeros_(self.blend_head[-1].bias)
            if blend_coherence_init:
                nn.init.zeros_(self.blend_head[0].weight)
                nn.init.zeros_(self.blend_head[0].bias)
        self.coherence_gain = 12.0
        self.coherence_center = 0.9

    @staticmethod
    def _lift(x: torch.Tensor, taps: int) -> torch.Tensor:
        batch, time_steps, features = x.shape
        freq = torch.view_as_complex(x.reshape(batch, time_steps, features // 2, 2).contiguous())
        return torch.fft.ifft(freq, dim=-1)[..., : min(taps, freq.shape[-1])]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.use_revin:
            x = self.instance_norm.normalize(x)
        if self.use_denoiser:
            x = self.denoiser(x, x.shape[1])

        delay = self._lift(x, self.num_delay_taps)
        if self.use_doppler_shrink:
            delay = self.shrink(delay)
        spectrum = torch.fft.fft(delay, dim=1)
        base = self.operator(spectrum)
        if self.use_persistence_blend:
            profile = torch.log(spectrum.abs().pow(2).mean(dim=-1) + 1e-8)
            if self.use_coherence_gate:

                power = delay.abs().pow(2).mean(dim=(1, 2))
                lag_one = (delay[:, 1:, :] * torch.conj(delay[:, :-1, :])).mean(dim=(1, 2))
                coherence = (lag_one.abs() / (power + 1e-8)).unsqueeze(1)
                logit = (
                    self.blend_head(torch.cat([profile, coherence], dim=1))
                    + self.coherence_gain * (coherence - self.coherence_center)
                )
                weight = torch.sigmoid(logit).view(batch, 1, 1)
            else:
                weight = torch.sigmoid(self.blend_head(profile)).view(batch, 1, 1)
            if self.blend_source == "mean":
                reference = delay.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)
            else:
                reference = delay[:, -1:, :].expand(-1, self.pred_len, -1)
            base = base + weight * (reference - base)
        predicted = base

        narrow = self.corr_taps < base.shape[-1]
        delay_c = delay[..., : self.corr_taps] if narrow else delay
        spectrum_c = spectrum[..., : self.corr_taps] if narrow else spectrum
        correction_sum = torch.zeros_like(base)

        mixed = self.shared_mixer(self.arl(delay_c))
        correction_sum[..., : self.corr_taps] += self.gate_mix * (
            self.operator(torch.fft.fft(mixed, dim=1)) - base[..., : self.corr_taps]
        )

        tokens = torch.view_as_real(spectrum_c).permute(0, 2, 1, 3).reshape(
            batch, self.corr_taps, -1
        )
        tokens = self.embed(tokens)
        if self.tap_encoding is not None:
            tokens = self.tap_encoding(tokens)
        tokens = self.encoder(tokens)
        correction = self.head(tokens).reshape(batch, self.corr_taps, self.pred_len, 2)
        correction = torch.view_as_complex(correction.contiguous()).permute(0, 2, 1)
        if self.use_correction_gate:

            profile = torch.log(spectrum_c.abs().pow(2).mean(dim=-1) + 1e-8)
            gate = torch.sigmoid(self.gate_head(profile)).view(batch, 1, 1)
            correction = correction * gate
        correction_sum[..., : self.corr_taps] += self.correction_scale * correction

        if self.use_delay_decomposition:
            _structured, residual = self.decomposition(delay_c)
            residual_tokens = torch.view_as_real(residual).permute(0, 2, 1, 3)
            residual_tokens = residual_tokens.reshape(batch, residual.shape[-1], -1)
            split = self.split_head(residual_tokens).reshape(batch, residual.shape[-1], self.pred_len, 2)
            split = torch.view_as_complex(split.contiguous()).permute(0, 2, 1)
            correction_sum[..., : self.corr_taps] += self.gate_split * split

        predicted = base + correction_sum
        taps = predicted.shape[-1]
        if taps < self.num_subcarriers:
            predicted = torch.cat(
                [predicted, predicted.new_zeros(batch, self.pred_len, self.num_subcarriers - taps)],
                dim=-1,
            )
        h_pred = torch.fft.fft(predicted, dim=-1)[..., : self.num_subcarriers]
        pred = torch.view_as_real(h_pred).reshape(batch, self.pred_len, self.num_subcarriers * 2)
        return self.instance_norm.denormalize(pred) if self.use_revin else pred

class _DDKoopTapLightning(BaseCSIModel):

    MODEL_NAME = "DDK10"

    def __init__(self, config: ExperimentConfig, *args, **kwargs):
        super().__init__(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
        )
        self.name = self.MODEL_NAME
        self.is_separate_antennas = config.model.is_separate_antennas
        self.save_hyperparameters({"model": config.model})
        self.model = DDKoopTapNet(**config.model.params)
        self.objective = SpectralObjective(
            pred_len=config.model.params.get("pred_len", 4),
            auxi_lambda=config.model.params.get("auxi_lambda", 0.5),
            mode=config.model.params.get("auxi_mode", "complex"),
        )

        self.kd_lambda = float(config.model.params.get("kd_lambda", 0.0))
        self.metric = NMSELoss()

    def __str__(self):
        return self.name

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        hist, target = batch[0], batch[1]
        teacher = batch[2] if len(batch) > 2 else None
        pred = self(hist)
        loss, auxi = self.objective(pred, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_auxi", auxi, on_step=False, on_epoch=True, logger=True)
        if teacher is not None and self.kd_lambda > 0:
            kd = torch.nn.functional.mse_loss(pred, teacher)
            loss = loss + self.kd_lambda * kd
            self.log("train_kd", kd, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):

        hist, target = batch[0], batch[1]
        pred = self(hist)
        loss = self.metric(pred, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

class DD_KOOP_TAP_TDD(_DDKoopTapLightning):

    MODEL_NAME = "DDK10"

class DD_KOOP_TAP_FDD(_DDKoopTapLightning):

    MODEL_NAME = "DDK10"

def _register() -> None:
    from src.cp.models import PREDICTORS

    PREDICTORS.DD_KOOP_TAP_TDD = DD_KOOP_TAP_TDD
    PREDICTORS.DD_KOOP_TAP_FDD = DD_KOOP_TAP_FDD

_register()

__all__ = ["DDKoopTapNet", "DD_KOOP_TAP_FDD", "DD_KOOP_TAP_TDD", "least_squares_spectrum_map"]
