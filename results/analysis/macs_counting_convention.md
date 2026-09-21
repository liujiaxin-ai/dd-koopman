# MACs counting convention (X3)

Source: `profile_local.py` (recovered from host 2, identical logic to the
original remote profiler) and direct measurements of `torch.utils.flop_counter`
on this machine.

## What is measured

```
counter = torch.utils.flop_counter.FlopCounterMode(display=False)
with counter:
    network(x)                      # x has shape [32, 16, 600]
macs_per_sample = counter.get_total_flops() / 2.0 / batch   # batch = 32
```

- **The unit is one antenna history of 16×300 complex samples.** The profiler
  feeds a whole scene (32 antennas) as a batch and divides by 32, so
  "MACs per sample" = MACs per antenna history, exactly the column label in
  Table 2.
- **One multiply-add counts as 2 FLOPs**, hence the `/ 2` that converts
  FLOPs to MACs. Verified: a `4×64 @ 64×64` real matmul reports 32,768 FLOPs,
  i.e. `4·64·64·2`.
- **Only matmul and convolution family ops are counted.** Measured on the
  reported model (`DD_KOOP_TDD` in the shipped code; historical harness alias `DD_KOOP8`, seed 42, 173,190 parameters):

  | op | FLOPs | share |
  |---|---|---|
  | `aten.convolution` | 253,593,600 | 60.8% |
  | `aten.addmm` | 159,645,696 | 38.3% |
  | `aten.bmm` | 3,686,400 | 0.9% |
  | **total** | **416,925,696** | 100% |

  `416,925,696 / 2 / 32 = 6,514,464` MACs per sample — reproduces the Table 2
  value 6.51 M exactly, so the table's convention is this counter.

## What is *not* counted

- **DFT/IDFT transforms.** `torch.fft.fft` on a complex tensor reports 0 FLOPs
  in this counter (measured on this machine: `fft total flops: 0`). The
  delay-domain lift and the inverse transform in our model are therefore
  outside the reported number.
- **Elementwise products**, real or complex (`complex64` and `float32`
  elementwise multiply of 1000 elements both report 0 FLOPs). The diagonal
  mode-gain multiplications and the gated corrections are outside it as well.

## Consequence for the paper

The MACs column is a *matmul-and-convolution* count, not total arithmetic. It
is applied identically to every row (the same profiler produced the published
row's 804.9 M), so the comparison is internally consistent, but a reader could
reasonably assume "MACs" means all arithmetic. The latency column
(2.24 ms vs 11.15 ms, measured wall-clock with `cuda.synchronize`) does not
depend on this convention and carries the efficiency claim on its own.

Suggested footnote wording: "MACs are PyTorch `FlopCounterMode` FLOPs divided by
two, per antenna history of 16×300; matmul and convolution ops only — DFT/IDFT
and elementwise products are outside the counter. Latency is wall-clock for one
32-antenna scene."
