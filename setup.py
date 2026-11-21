# SPDX-License-Identifier: MIT
"""Setuptools build script compiling CrossCPU's isolated per-arch C11 native shared library."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

CORE_SOURCES = [
    "native/core/worker_pool.c",
    "native/core/dispatch.c",
]

COMMON_SOURCES = [
    "native/arch/common/isa_scalar.c",
    "native/arch/common/cache_flush_generic.c",
]


def detect_arch_family(machine: str | None = None) -> str:
    """Map host/target machine architecture string to a CrossCPU native/arch/ family."""
    raw = (machine or os.environ.get("CROSS_CPU_ARCH") or platform.machine()).strip().lower()
    if raw in ("x86_64", "amd64"):
        return "x86_64"
    if raw in ("arm64", "aarch64"):
        return "arm64"
    if raw == "riscv64":
        return "riscv64"
    if raw in ("ppc64", "ppc64le", "powerpc64", "powerpc64le"):
        return "ppc64"
    if raw == "s390x":
        return "s390x"
    return "generic"


def probe_compiler_snippet(cc_cmd: list[str], flags: list[str], snippet: str) -> bool:
    """Return True if the C compiler compiles ``snippet`` cleanly with ``flags``."""
    with tempfile.TemporaryDirectory(prefix="cross-cpu-probe-") as tmp:
        src_path = Path(tmp) / "probe.c"
        obj_path = Path(tmp) / "probe.o"
        src_path.write_text(snippet, encoding="utf-8")
        cmd = [*cc_cmd, "-std=c11", "-O2", "-fPIC", *flags, "-c", str(src_path), "-o", str(obj_path)]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return proc.returncode == 0
        except OSError:
            return False


def build_arch_compile_plan(
    arch_family: str,
    cc_cmd: list[str],
) -> list[tuple[str, list[str]]]:
    """
    Build the exact ``[(source_file, per_file_extra_flags), ...]`` plan.

    Core, common, and probe files NEVER receive ISA-specific flags or ``-march=native``.
    """
    plan: list[tuple[str, list[str]]] = [
        (src, []) for src in (*CORE_SOURCES, *COMMON_SOURCES)
    ]

    if arch_family == "x86_64":
        plan.append(("native/arch/x86_64/probe_x86.c", []))

        sse4_flags: list[str] = []
        if probe_compiler_snippet(
            cc_cmd, ["-msse4.2"], "#include <immintrin.h>\nint f(void){return 0;}\n"
        ):
            sse4_flags.extend(["-msse4.2", "-DCROSS_CPU_HAVE_SSE4"])
            if probe_compiler_snippet(
                cc_cmd,
                ["-msse4.2"],
                '#include <immintrin.h>\n__attribute__((target("aes"))) '
                "__m128i f(__m128i a){return _mm_aesenc_si128(a,a);}\n",
            ):
                sse4_flags.append("-DCROSS_CPU_HAVE_X86_AES")
            if probe_compiler_snippet(
                cc_cmd,
                ["-msse4.2"],
                '#include <immintrin.h>\n__attribute__((target("sha,ssse3,sse4.1"))) '
                "__m128i f(__m128i a){return _mm_sha256msg1_epu32(a,a);}\n",
            ):
                sse4_flags.append("-DCROSS_CPU_HAVE_X86_SHA")
        plan.append(("native/arch/x86_64/isa_sse4.c", sse4_flags))

        avx2_flags: list[str] = []
        if probe_compiler_snippet(
            cc_cmd,
            ["-mavx2", "-mfma"],
            "#include <immintrin.h>\n__m256 f(__m256 a){return _mm256_fmadd_ps(a,a,a);}\n",
        ):
            avx2_flags.extend(["-mavx2", "-mfma", "-DCROSS_CPU_HAVE_AVX2"])
            if probe_compiler_snippet(
                cc_cmd,
                ["-mavx2", "-mfma"],
                '#include <immintrin.h>\n__attribute__((target("avx2,avxvnni"))) '
                "__m256i f(__m256i a){return _mm256_dpbusd_avx_epi32(a,a,a);}\n",
            ):
                avx2_flags.append("-DCROSS_CPU_HAVE_X86_AVXVNNI")
        plan.append(("native/arch/x86_64/isa_avx2.c", avx2_flags))

        avx512_flags: list[str] = []
        base512 = ["-mavx512f", "-mavx512bw", "-mavx512vl"]
        if probe_compiler_snippet(
            cc_cmd,
            base512,
            "#include <immintrin.h>\n__m512 f(__m512 a){return _mm512_fmadd_ps(a,a,a);}\n",
        ):
            avx512_flags.extend([*base512, "-DCROSS_CPU_HAVE_AVX512"])
            if probe_compiler_snippet(
                cc_cmd,
                base512,
                '#include <immintrin.h>\n__attribute__((target("avx512f,avx512vnni"))) '
                "__m512i f(__m512i a){return _mm512_dpbusd_epi32(a,a,a);}\n",
            ):
                avx512_flags.append("-DCROSS_CPU_HAVE_X86_AVX512VNNI")
            if probe_compiler_snippet(
                cc_cmd,
                base512,
                '#include <immintrin.h>\n__attribute__((target("avx512f,avx512bf16"))) '
                "__m512 f(__m512 a, __m512bh b){return _mm512_dpbf16_ps(a,b,b);}\n",
            ):
                avx512_flags.append("-DCROSS_CPU_HAVE_X86_AVX512BF16")
            if probe_compiler_snippet(
                cc_cmd,
                base512,
                '#include <immintrin.h>\n__attribute__((target("avx512f,avx512fp16"))) '
                "__m512h f(__m512h a){return _mm512_fmadd_ph(a,a,a);}\n",
            ):
                avx512_flags.append("-DCROSS_CPU_HAVE_X86_AVX512FP16")
        plan.append(("native/arch/x86_64/isa_avx512.c", avx512_flags))

        amx_flags: list[str] = []
        has_amx_bf16 = probe_compiler_snippet(
            cc_cmd,
            ["-mamx-tile", "-mamx-bf16"],
            "#include <immintrin.h>\nvoid f(void){_tile_dpbf16ps(0,1,2);}\n",
        )
        has_amx_int8 = probe_compiler_snippet(
            cc_cmd,
            ["-mamx-tile", "-mamx-int8"],
            "#include <immintrin.h>\nvoid f(void){_tile_dpbssd(0,1,2);}\n",
        )
        if has_amx_bf16 or has_amx_int8:
            amx_flags.extend(["-mamx-tile", "-DCROSS_CPU_HAVE_AMX"])
            if has_amx_bf16:
                amx_flags.extend(["-mamx-bf16", "-DCROSS_CPU_HAVE_AMX_BF16"])
            if has_amx_int8:
                amx_flags.extend(["-mamx-int8", "-DCROSS_CPU_HAVE_AMX_INT8"])
        plan.append(("native/arch/x86_64/isa_amx.c", amx_flags))

    elif arch_family == "arm64":
        plan.append(("native/arch/arm64/probe_arm64.c", []))

        neon_flags: list[str] = []
        arm_probes = [
            (
                "-DCROSS_CPU_HAVE_ARM_FP16",
                '#include <arm_neon.h>\n__attribute__((target("fp16"))) '
                "float16x8_t f(float16x8_t a){return vfmaq_f16(a,a,a);}\n",
            ),
            (
                "-DCROSS_CPU_HAVE_ARM_BF16",
                '#include <arm_neon.h>\n__attribute__((target("bf16"))) '
                "float32x4_t f(float32x4_t a, bfloat16x8_t b){return vbfdotq_f32(a,b,b);}\n",
            ),
            (
                "-DCROSS_CPU_HAVE_ARM_DOTPROD",
                '#include <arm_neon.h>\n__attribute__((target("dotprod"))) '
                "int32x4_t f(int32x4_t a, int8x16_t b){return vdotq_s32(a,b,b);}\n",
            ),
            (
                "-DCROSS_CPU_HAVE_ARM_I8MM",
                '#include <arm_neon.h>\n__attribute__((target("i8mm"))) '
                "int32x4_t f(int32x4_t a, int8x16_t b){return vmmlaq_s32(a,b,b);}\n",
            ),
            (
                "-DCROSS_CPU_HAVE_ARM_AES",
                '#include <arm_neon.h>\n__attribute__((target("aes"))) '
                "uint8x16_t f(uint8x16_t a){return vaesmcq_u8(vaeseq_u8(a,a));}\n",
            ),
            (
                "-DCROSS_CPU_HAVE_ARM_SHA2",
                '#include <arm_neon.h>\n__attribute__((target("sha2"))) '
                "uint32x4_t f(uint32x4_t a){return vsha256hq_u32(a,a,a);}\n",
            ),
        ]
        for macro, snippet in arm_probes:
            if probe_compiler_snippet(cc_cmd, [], snippet):
                neon_flags.append(macro)
        plan.append(("native/arch/arm64/isa_neon.c", neon_flags))

        sve_flags: list[str] = []
        if probe_compiler_snippet(
            cc_cmd,
            ["-march=armv8.2-a+sve"],
            "#include <arm_sve.h>\nfloat f(void){return svaddv_f32(svptrue_b32(),svdup_f32(1.0f));}\n",
        ):
            sve_flags.extend(["-march=armv8.2-a+sve", "-DCROSS_CPU_HAVE_SVE"])
        plan.append(("native/arch/arm64/isa_sve.c", sve_flags))

        sme_flags: list[str] = []
        if probe_compiler_snippet(
            cc_cmd,
            ["-march=armv9-a+sme"],
            '#include <arm_sme.h>\n__arm_new("za") void f(void) __arm_streaming {svzero_za();}\n',
        ):
            sme_flags.extend(["-march=armv9-a+sme", "-DCROSS_CPU_HAVE_SME"])
        plan.append(("native/arch/arm64/isa_sme.c", sme_flags))

    elif arch_family == "riscv64":
        plan.append(("native/arch/other/riscv64/probe_riscv64.c", []))
        rvv_flags: list[str] = []
        if probe_compiler_snippet(
            cc_cmd,
            ["-march=rv64gcv"],
            "#include <riscv_vector.h>\nsize_t f(void){return __riscv_vsetvl_e64m1(4);}\n",
        ):
            rvv_flags.extend(["-march=rv64gcv", "-DCROSS_CPU_HAVE_RVV"])
        plan.append(("native/arch/other/riscv64/isa_rvv.c", rvv_flags))

    elif arch_family == "ppc64":
        plan.append(("native/arch/other/ppc64/probe_ppc64.c", []))
        vsx_flags: list[str] = []
        if probe_compiler_snippet(
            cc_cmd,
            ["-mvsx"],
            "#include <altivec.h>\nint f(void){return 0;}\n",
        ):
            vsx_flags.extend(["-mvsx", "-DCROSS_CPU_HAVE_VSX"])
        plan.append(("native/arch/other/ppc64/isa_vsx.c", vsx_flags))

    elif arch_family == "s390x":
        plan.append(("native/arch/other/s390x/probe_s390x.c", []))
        vx_flags: list[str] = []
        if probe_compiler_snippet(
            cc_cmd,
            ["-mzvector", "-march=z13"],
            "#include <vecintrin.h>\nint f(void){return 0;}\n",
        ):
            vx_flags.extend(["-mzvector", "-march=z13", "-DCROSS_CPU_HAVE_VX"])
            if probe_compiler_snippet(
                cc_cmd,
                ["-mzvector", "-march=z13"],
                '#include <vecintrin.h>\n__attribute__((target("arch=z14"))) '
                "__vector float f(__vector float a){return vec_madd(a,a,a);}\n",
            ):
                vx_flags.append("-DCROSS_CPU_HAVE_VXE")
        plan.append(("native/arch/other/s390x/isa_vx.c", vx_flags))

    else:
        plan.append(("native/arch/other/generic/probe_generic.c", []))

    return plan


def _extract_darwin_arches(cmd: list[str]) -> list[str]:
    arches: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "-arch" and i + 1 < len(cmd):
            val = cmd[i + 1]
            if val not in arches:
                arches.append(val)
            i += 2
        else:
            i += 1
    return arches


def _strip_darwin_arch_flags(cmd: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "-arch" and i + 1 < len(cmd):
            i += 2
        else:
            out.append(cmd[i])
            i += 1
    return out


class CrossCPUBuildExt(build_ext):
    """Custom build_ext compiling isolated per-ISA C11 translation units for the target arch."""

    def get_export_symbols(self, ext: Extension) -> list[str]:
        # Pure C11 ctypes shared library; do not require PyInit__native export
        return []

    def _compile_single_slice(
        self,
        *,
        arch_family: str,
        cc_cmd: list[str],
        output_path: str,
        temp_dir: str,
        slice_extra_flags: list[str],
    ) -> None:
        compile_plan = build_arch_compile_plan(arch_family, [*cc_cmd, *slice_extra_flags])
        base_flags = ["-std=c11", "-O2", "-Wall", "-Wextra", "-fPIC", *slice_extra_flags]

        objects: list[str] = []
        for src, isa_flags in compile_plan:
            objs = self.compiler.compile(
                [src],
                output_dir=temp_dir,
                include_dirs=["native/include"],
                extra_postargs=[*base_flags, *isa_flags],
            )
            objects.extend(objs)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.compiler.link_shared_object(
            objects,
            output_path,
            libraries=["pthread", "m"],
            extra_preargs=list(slice_extra_flags),
            target_lang="c",
        )

    def build_extension(self, ext: Extension) -> None:
        raw_cc = os.environ.get("CC")
        cc_cmd = shlex.split(raw_cc) if raw_cc else list(self.compiler.compiler[:1])
        ext_path = self.get_ext_fullpath(ext.name)

        if sys.platform == "darwin":
            arch_env = shlex.split(os.environ.get("ARCHFLAGS", ""))
            arches = _extract_darwin_arches(arch_env) or _extract_darwin_arches(
                list(self.compiler.compiler_so)
            )
            if len(arches) > 1 and not os.environ.get("CROSS_CPU_ARCH"):
                orig_compiler_so = list(self.compiler.compiler_so)
                orig_linker_so = list(self.compiler.linker_so)
                try:
                    self.compiler.set_executable(
                        "compiler_so", _strip_darwin_arch_flags(orig_compiler_so)
                    )
                    self.compiler.set_executable(
                        "linker_so", _strip_darwin_arch_flags(orig_linker_so)
                    )
                    lipo_bin = shutil.which("lipo")
                    if lipo_bin:
                        thin_outputs: list[str] = []
                        for mac_arch in arches:
                            slice_dir = os.path.join(self.build_temp, mac_arch)
                            thin_out = os.path.join(self.build_temp, f"_native.{mac_arch}.so")
                            self._compile_single_slice(
                                arch_family=detect_arch_family(mac_arch),
                                cc_cmd=cc_cmd,
                                output_path=thin_out,
                                temp_dir=slice_dir,
                                slice_extra_flags=["-arch", mac_arch],
                            )
                            thin_outputs.append(thin_out)
                        Path(ext_path).parent.mkdir(parents=True, exist_ok=True)
                        subprocess.run(
                            [lipo_bin, "-create", *thin_outputs, "-output", ext_path],
                            check=True,
                        )
                        return
                    # Fallback if lipo unavailable: build host architecture slice only
                    host_arch = platform.machine()
                    self._compile_single_slice(
                        arch_family=detect_arch_family(host_arch),
                        cc_cmd=cc_cmd,
                        output_path=ext_path,
                        temp_dir=self.build_temp,
                        slice_extra_flags=["-arch", host_arch],
                    )
                    return
                finally:
                    self.compiler.set_executable("compiler_so", orig_compiler_so)
                    self.compiler.set_executable("linker_so", orig_linker_so)

        self._compile_single_slice(
            arch_family=detect_arch_family(),
            cc_cmd=cc_cmd,
            output_path=ext_path,
            temp_dir=self.build_temp,
            slice_extra_flags=[],
        )


setup(
    ext_modules=[
        Extension(
            name="cross_cpu._native",
            sources=[*CORE_SOURCES, *COMMON_SOURCES],
            include_dirs=["native/include"],
        )
    ],
    cmdclass={"build_ext": CrossCPUBuildExt},
)
