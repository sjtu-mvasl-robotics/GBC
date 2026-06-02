# Block-causal clip-compression latent (T-series) — design spec

**Date:** 2026-06-02
**Status:** implemented + unit-tested locally; smoke pending; **not yet submitted to DLC**
**Baseline:** D4 (`checkpoints/d4_baseline`, per-frame, val l2m=0.0823); E1 = causal per-frame (running).

## Motivation

Make the **latent unit a motion clip, not a frame**: a clip of `k` frames carries a
pattern (stride, reach, turn) a single frame's latent cannot. This is SONIC's
temporal-compaction idea, adapted to our three constraints that SONIC (a non-causal
whole-motion *tracker*) does not share:

1. **Diffusion**: the latent must feed a motion FM/diffusion predictor — fewer tokens
   = shorter autoregressive rollout, less error accumulation. (The upside.)
2. **Retargeting**: must decode to per-frame robot DOFs.
3. **Real-time**: streaming. **Compression ratio = latency** — a token can't be emitted
   until its block's frames arrive.

**FPS-invariance is explicitly NOT a motivation.** The data pipeline already resamples
every AMASS clip to a canonical 20 fps upstream (`amass_loader.py` `interpolate_angle_axis`
= SLERP-equivalent + `interpolate_trans`; `amass_sequence10_dataset.py target_fps=20.0`).
Source-rate variance is gone before the encoder runs.

## The budget framing (why this is worth an ablation)

For a fixed latent budget there are two axes:
- **Per-frame width** (`slot_z`) — the E2 sweep (48/24/12), already running.
- **Temporal count** (tokens/sec) — this T-series.

Falsifiable hypothesis ("a clip holds a pattern a frame doesn't"): **few fat clip tokens
reconstruct full motion as well as, or better than, many thin per-frame tokens at matched
budget — and are easier for an FM to predict.**

## Architecture

Block-causal: **bidirectional within a block** (whole clip pattern visible; bounded
lookahead = `k` frames) but **causal across blocks** (no future block leaks into a past
one; streaming-safe; FM rolls out block-by-block).

```
SMPL-H clip (T) → graph encoder (per-frame, per-slot, GraphFuseBlock)   (B,T,S,D)
   → block-causal RoPE temporal (mask = kron(tril(N),ones(k,k)))        (B,T,S,D)
   → BlockCompressor (non-overlapping stride-k conv)                    (B,K,S,D)   K=T/k
   → mu/logvar                                              z_clip       (B,K,S,slot_z)  ← VAE bottleneck (KL here)
   → BlockCausalUpsampler (perceiver cross-attn, block-causal, RoPE)    (B,T,S,slot_z)
   → existing per-frame DOF-graph decoder / human_decoder / residual    per-frame, unchanged
```

Decisions (approved 2026-06-02):
- **Compression** = block-causal RoPE *attention* + stride-k conv pool (not a plain conv kernel).
- **Decompression** = `BlockCausalUpsampler` (perceiver cross-attn, anti-checkerboard).
- **PE** = two-level RoPE: frame positions in the encoder; query=frame / key=block-center in the upsampler.
- **Decoder context = none (option a)**: the K clip tokens are the SOLE information
  channel. D4's `masked_graph_hidden` per-frame context would bypass the bottleneck and
  void the compression test. (Option b — route context through the bottleneck — is the
  follow-up if recon needs it.)
- **Port, not import**: the rsl_rl block-causal/RoPE/compressor blocks were reimplemented
  in the data-prep package (adapted to per-slot 4-D), so the retargeter takes no hard
  dependency on the RL trainer. See [[reference-block-causal-toolkit]].

## Implementation

New module `GBC/GBC/utils/data_preparation/block_causal_compress.py`:
`build_block_causal_mask`, `RotaryPE`/`apply_rope`, `BlockCausalRoPELayer`,
`BlockCausalRoPEEncoder` (standalone), `BlockCompressor`, `BlockCausalUpsampler`,
`pick_num_heads`.

`graph_retarget_latent.py`:
- `GraphTemporalEncoder` gains `block_size`: when `>1`, the per-slot temporal path uses
  `BlockCausalRoPELayer`s + the block-causal mask (no additive sinusoidal PE; RoPE inside
  attention). Graph fusion (per-frame) is preserved, so whole-encoder block-causality holds.
  Reports `is_causal=True`, `total_lookahead_frames=k-1`.
- `UnifiedLatentRetargeter` routes `temporal_module="block_causal_compress"`: builds the
  block-causal encoder + `BlockCompressor(d_model,k)` + `BlockCausalUpsampler(slot_z,…,k)`.
  `encode_latent` compresses before mu/logvar (latent = (B,K,S,slot_z)); `forward`
  upsamples sampled z to per-frame and passes `context=None`.

`train_graph_latent.py`: `--temporal-module block_causal_compress`,
`TEMPORAL_MODULE_ID=7`, `--block-size`, constructor wiring, and a `train()` guard that
`--window % --block-size == 0`.

## Verification

- `GBC/tests/test_block_causal_compress.py` — 8/8 pass: mask shape; encoder within-block
  bidirectional + across-block causal (<1e-5 past-stability) + past-propagates-forward;
  compressor shape; upsampler block-causal + past-propagates; **GraphTemporalEncoder(block_size=k)
  integration** block-causal. (Perturbations are NOISE, not constant — pre-norm LayerNorm
  erases a constant offset and would falsely pass.)
- E1 causal tests still pass (no regression from the `GraphTemporalEncoder` change).
- Local smoke `scripts_phaseA/smoke_t_blockcausal_local.sh` (k=5, window 60 → 12 tokens):
  **PASS (exit 0).** `temporal_module_id=7`, `is_causal=true`, `total_lookahead_frames=4`
  (=k-1); `weighted_rot_term` 0.744→0.542→0.495→0.415 over 60 steps; val l2m 0.140→0.130.
  End-to-end compress→bottleneck→upsample→decode trains with no shape/KL errors.
  (Smoke is 16 files / 2 epochs — quality is not meaningful, only that it learns.)

## Validation plan (DLC, after smoke + review)

T-series ladder on the D4 recipe, causal throughout:
| Cell | block_size | tokens/window (T=60) | tests |
|---|---|---|---|
| T0 ref | 1 (=E1 causal) | 60 | per-frame control |
| T2 | 2 | 30 | mild compression |
| T4 | 4 | 15 | |
| T5 | 5 | 12 | ~0.25 s clip |

Metrics: full-motion recon l2m vs block_size; latent budget `(T/k)·S·slot_z`; the payoff
metric — a small FM's per-step prediction error on clip tokens vs per-frame z at matched
budget. T0 reuses E1 as the control.

## Out of scope / follow-ups
- Option (b) bottleneck-routed decoder context.
- FSQ quantization of clip tokens (discrete token vocabulary for a VLA/LM planner).
- The FM-trackability experiment is its own plan (depends on a chosen block_size).
