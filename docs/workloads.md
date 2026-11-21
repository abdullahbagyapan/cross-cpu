# Workloads, Fixed Units & SDC Verification

`CrossCPU` defines **21 canonical workloads** across 7 categories. All measurements report **fixed, non-auto-scaling SI units** (`10^9`):

- **`GOP/s` & `GOP`:** $10^9$ integer or atomic operations.
- **`GFLOP/s` & `GFLOP`:** $10^9$ floating-point operations.
- **`GB/s` & `GB`:** SI $10^9$ bytes transferred (write + verified readback).

## Zero Fake Emulation Policy

Workloads requiring hardware datatypes, dot-product pipelines, matrix tile engines, or cryptographic instructions (`compute-vnni`, `compute-fp16`, `compute-bf16`, `compute-matrix-fp32`, `compute-matrix-bf16`, `compute-matrix-int8`, `crypto-aes`, `crypto-sha256`) **never** fall back to scalar C loops or FP32 upcasting. When unsupported by the host CPU/OS or forced `--isa`, they report `status="unsupported"` with an explicit reason.

## Workload Matrix

| ID | Workload Name | Category | Units | Verification & Hardware Requirements |
| --- | --- | --- | --- | --- |
| `0` | `compute-int64` | `integer` | `GOP/s`, `GOP` | 64-bit unsigned modular linear recurrence ($x_{k+1} = A x_k + C \bmod 2^{64}$) verified bit-for-bit against $O(\log K)$ closed-form reference across all vector lanes. |
| `1` | `compute-int32` | `integer` | `GOP/s`, `GOP` | 32-bit unsigned modular recurrence verified bit-for-bit against exact analytical reference. |
| `2` | `compute-int16` | `integer` | `GOP/s`, `GOP` | 16-bit unsigned modular recurrence verified bit-for-bit against exact analytical reference. |
| `3` | `compute-int8` | `integer` | `GOP/s`, `GOP` | 8-bit unsigned modular recurrence verified bit-for-bit against exact analytical reference. |
| `4` | `compute-vnni` | `integer` | `GOP/s`, `GOP` | Hardware 8-bit integer dot-product / matrix accumulate (x86 `AVX-VNNI`/`AVX512-VNNI` `vpdpbusd` or ARM `DotProd` `sdot` / `I8MM` `smmla` / SVE `sdot`) with bit-exact accumulator verification. |
| `5` | `compute-fp64` | `floating-point` | `GFLOP/s`, `GFLOP` | IEEE-754 FP64 bounded contractive FMA recurrence ($y_{k+1} = 0.75 y_k + b_i$) with paired sign-symmetry ($+y_0/+b_i$ vs $-y_0/-b_i \implies y^+ + y^- = 0$), `isfinite()` `NaN`/`Inf` rejection, and tight tolerance against fixed point $4 b_i$. |
| `6` | `compute-fp32` | `floating-point` | `GFLOP/s`, `GFLOP` | IEEE-754 FP32 bounded contractive FMA recurrence with paired sign-symmetry, `NaN`/`Inf` rejection, and fixed-point verification. |
| `7` | `compute-fp16` | `floating-point` | `GFLOP/s`, `GFLOP` | Native hardware IEEE-754 FP16 FMA recurrence (ARMv8.2-FP16 `vfmaq_f16` / SVE `svmla_f16_x` or x86 `AVX-512FP16` `_mm512_fmadd_ph`). |
| `8` | `compute-bf16` | `floating-point` | `GFLOP/s`, `GFLOP` | Native hardware BFloat16 dot-product (ARMv8.6-BF16 `vbfdotq_f32` or x86 `AVX-512BF16` `_mm512_dpbf16_ps`) verified against exact binary BF16 inputs. |
| `9` | `compute-matrix-fp32` | `matrix` | `GFLOP/s`, `GFLOP` | Hardware FP32 outer-product matrix tile engine (ARM SME `svmopa_za32_f32_m`), counting $2 M N K$ FLOPs and verifying every ZA tile element analytically. |
| `10` | `compute-matrix-bf16` | `matrix` | `GFLOP/s`, `GFLOP` | Hardware BF16 matrix tile engine (Intel `AMX-BF16` `_tile_dpbf16ps` or ARM `SME` `svmopa_za32_bf16_m`), counting $2 M N K$ FLOPs and verifying every tile element. |
| `11` | `compute-matrix-int8` | `matrix` | `GOP/s`, `GOP` | Hardware INT8 matrix tile engine (Intel `AMX-INT8` `_tile_dpbssd` or ARM `SME` `svmopa_za32_s8_m`), counting $2 M N K$ ops with bit-exact tile verification. |
| `12` | `crypto-aes` | `crypto` | `GB/s`, `GB` | Hardware AES-128 10-round encryption (x86 `AES-NI` or ARM `FEAT_AES`) verified on every block against NIST FIPS 197 Appendix C.1 KAT (`69c4e0d86a7b0430d8cdb78070b4c55a`) with loop-carried dependency barriers. |
| `13` | `crypto-sha256` | `crypto` | `GB/s`, `GB` | Hardware SHA-256 64-round compression (x86 `SHA-NI` or ARM `FEAT_SHA256`) verified on every block against NIST FIPS 180-4 `"abc"` KAT digest. |
| `14` | `cache-l1` | `cache` | `GB/s`, `GB` | Per-thread buffer sized to ~50% of L1d cache; SIMD/scalar write + bit-exact readback verification ($2 \times \text{buffer\_bytes}$ per pass). |
| `15` | `cache-l2` | `cache` | `GB/s`, `GB` | Per-thread buffer sized to ~50% of per-core L2 share; write + bit-exact readback verification. |
| `16` | `cache-l3` | `cache` | `GB/s`, `GB` | Per-thread buffer sized so participating threads occupy ~50% of shared L3/LLC (reports `unsupported` on hosts whose topology has no L3 tier). |
| `17` | `bandwidth-ram` | `memory` | `GB/s`, `GB` | Aggregate working set $\ge 4\times \text{LLC}$ (capped safely against free RAM) using non-temporal streaming stores (`_mm512_stream_si512`, `_mm256_stream_si256`, `_mm_stream_si128`, ARM64 `stnp`, SVE `svstnt1_u64`) or cache flush/eviction (`scalar`/`rvv`/`vsx`/`vx`) + memory fence + full readback verification. |
| `18` | `memory-pattern` | `memory` | `GB/s`, `GB` | Adversarial DRAM bit patterns (`0x00..`, `0xFF..`, `0x55..`, `0xAA..`, `0x0F..`, `0xF0..`, walking-1, walking-0, checkerboard) with hardware cache line flush (`clflushopt`/`dc civac`/eviction sweep) before readback. |
| `19` | `memory-sweep` | `memory` | `GB/s`, `GB` | SplitMix64 address-hashed DRAM sweep (`splitmix64(salt ^ (idx * GOLDEN_RATIO))`) + hardware cache flush before readback to catch address-line aliasing and stuck DRAM bits. |
| `20` | `coherency-atomic` | `coherency` | `GOP/s`, `GOP` | Multi-core C11 `<stdatomic.h>` `atomic_fetch_add_explicit` contention across 8 cache-line-padded (`128B` stride) shared slots plus per-thread CAS loop, verifying exact global increment sum post-join. |
