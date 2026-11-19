# Architecture-Separated ISA Backends

`--isa auto` resolves the highest-priority ISA backend that is both available on the host (silicon + OS save-state + compiled-in) and natively implements the requested workload. `--isa <name>` forces that exact ISA backend (exiting with `INIT_ERROR (2)` if unavailable on the host).

---

## 1. x86_64 (`native/arch/x86_64/`)

Probed via `CPUID` leaves `0`, `1`, and `7` (subleaves `0` and `1`), `XGETBV` (`XCR0` OS register save-state validation), and Linux `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` in `probe_x86.c`. Cache line flushing uses `clflushopt` (with `mfence` barriers), falling back to `clflush` or `cross_cpu_cache_flush_generic()`.

| ISA Name | Priority | Translation Unit & Probed Flags | Runtime Detection Gate (Silicon + OS Save-State) |
| --- | --- | --- | --- |
| `amx` | `120` | `isa_amx.c` (`-mamx-tile -mamx-bf16 -mamx-int8`) | `CPUID.7.0:EDX` AMX-TILE (`bit 24`) + AMX-BF16 (`bit 22`) / AMX-INT8 (`bit 25`) + `XCR0` bits `17..18` (`XTILECFG`, `XTILEDATA`) + Linux `arch_prctl(ARCH_REQ_XCOMP_PERM, 18) == 0` |
| `avx512` | `100` | `isa_avx512.c` (`-mavx512f -mavx512bw -mavx512vl` + target `avx512vnni`, `avx512bf16`, `avx512fp16`) | `CPUID.7.0:EBX` AVX-512F/BW/VL (+ `ECX` VNNI, `EDX` FP16, `7.1:EAX` BF16) + `XCR0` bits `1,2,5,6,7` (`0xE6`) |
| `avx2` | `80` | `isa_avx2.c` (`-mavx2 -mfma` + target `avxvnni`) | `CPUID.1:ECX` AVX + FMA + `OSXSAVE` + `CPUID.7.0:EBX` AVX2 (+ `7.1:EAX` AVX-VNNI) + `XCR0` bits `1,2` (`0x6`) |
| `sse4` | `60` | `isa_sse4.c` (`-msse4.2` + target `aes`, `sha`) | `CPUID.1:ECX` SSE4.2 (`bit 20`) (+ `CPUID.1:ECX` AES-NI `bit 25`, `CPUID.7.0:EBX` SHA-NI `bit 29`) |
| `scalar` | `20` | `native/arch/common/isa_scalar.c` (baseline C11) | Always available |

---

## 2. ARM64 / `aarch64` (`native/arch/arm64/`)

Probed via `getauxval(AT_HWCAP)` and `getauxval(AT_HWCAP2)` on Linux and `sysctlbyname("hw.optional.arm.*")` on macOS in `probe_arm64.c`. Cache line flushing uses user-space `dc civac` + `dsb ish` stepping by the discovered L1 data cache line size (`128B` on Apple Silicon, `64B` on standard ARM64).

| ISA Name | Priority | Translation Unit & Probed Flags | Runtime Detection Gate (Linux & macOS) |
| --- | --- | --- | --- |
| `sme` | `120` | `isa_sme.c` (`-march=armv9-a+sme`, `<arm_sme.h>`) | Linux `AT_HWCAP2` `HWCAP2_SME` (`bit 23`) / macOS `sysctlbyname("hw.optional.arm.FEAT_SME")` (M4+) |
| `sve` | `100` | `isa_sve.c` (`-march=armv8.2-a+sve`, `<arm_sve.h>`) | Linux `AT_HWCAP` `HWCAP_SVE` (`bit 22`, plus `AT_HWCAP2` `HWCAP2_SVE2`); not exposed by macOS XNU |
| `neon` | `80` | `isa_neon.c` (baseline ARMv8-A + target `fp16`, `bf16`, `dotprod`, `i8mm`, `aes`, `sha2`) | Baseline ASIMD/NEON + per-extension capability gates for `FP16`, `BF16`, `DotProd`, `I8MM`, `FEAT_AES`, `FEAT_SHA256`, and non-temporal `stnp` |
| `scalar` | `20` | `native/arch/common/isa_scalar.c` (baseline C11) | Always available |

---

## 3. Other Architectures (`native/arch/other/`)

Dedicated sub-modules ensure distro build farms on non-x86/non-ARM architectures compile cleanly, vectorize where supported, and pass all offline verification tests.

| Target Arch | Directory | ISA Name | Priority | Runtime Detection Gate & Fallback |
| --- | --- | --- | --- | --- |
| `riscv64` | `native/arch/other/riscv64/` | `rvv` | `80` | Linux `getauxval(AT_HWCAP)` V-extension (`1 << ('V' - 'A')`) and `riscv_hwprobe` (`RISCV_HWPROBE_IMA_V` + `Zvk*`); compiled when toolchain supports `-march=rv64gcv` (`<riscv_vector.h>`) |
| `ppc64le` / `ppc64` | `native/arch/other/ppc64/` | `vsx` | `80` | Linux `getauxval(AT_HWCAP)` `PPC_FEATURE_HAS_VSX` (`0x80`) and `AT_HWCAP2` POWER9/POWER10/MMA bits; compiled with `-mvsx` (`<altivec.h>`) |
| `s390x` | `native/arch/other/s390x/` | `vx` | `80` | Linux `getauxval(AT_HWCAP)` `HWCAP_S390_VX` (`2048`) and `HWCAP_S390_VXE` (`8192`); big-endian safe; compiled with `-mzvector` (`<vecintrin.h>`) |
| Any other POSIX CPU | `native/arch/other/generic/` | `scalar` | `20` | `probe_generic.c` registers portable C11 `scalar` (`isa_scalar.c`) and `cache_flush_generic.c` with zero build failures |
