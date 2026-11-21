# SPDX-License-Identifier: MIT
# Offline build, test, and FHS-compliant installation Makefile for CrossCPU.

CC        ?= cc
PYTHON    ?= python3
PREFIX    ?= /usr/local
DESTDIR   ?=

UNAME_S   := $(shell uname -s)
UNAME_M   := $(shell uname -m)
ARCH      ?= $(UNAME_M)

ifeq ($(UNAME_S),Darwin)
  LIB_EXT     := .dylib
  SHLIB_FLAGS := -dynamiclib
  LDFLAGS_OS  := -lpthread -lm
else
  LIB_EXT     := .so
  SHLIB_FLAGS := -shared
  LDFLAGS_OS  := -lpthread -lm
endif

BASE_CFLAGS := -std=c11 -O2 -Wall -Wextra -fPIC -Inative/include $(CFLAGS)
BUILD_DIR   := build/native
TARGET_LIB  := src/_native$(LIB_EXT)

# Normalize target architecture into native/arch/<target>
ifneq (,$(filter x86_64 amd64,$(ARCH)))
  ARCH_FAMILY := x86_64
else ifneq (,$(filter arm64 aarch64,$(ARCH)))
  ARCH_FAMILY := arm64
else ifneq (,$(filter riscv64,$(ARCH)))
  ARCH_FAMILY := riscv64
else ifneq (,$(filter ppc64 ppc64le powerpc64 powerpc64le,$(ARCH)))
  ARCH_FAMILY := ppc64
else ifneq (,$(filter s390x,$(ARCH)))
  ARCH_FAMILY := s390x
else
  ARCH_FAMILY := generic
endif

CORE_SRCS   := native/core/worker_pool.c native/core/dispatch.c
COMMON_SRCS := native/arch/common/isa_scalar.c native/arch/common/cache_flush_generic.c

# Escaped '#' for C preprocessor directives inside GNU Make $(shell ...) probes
HASH := \#

CORE_OBJS   := $(patsubst %.c,$(BUILD_DIR)/%.o,$(CORE_SRCS))
COMMON_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(COMMON_SRCS))

ifeq ($(ARCH_FAMILY),x86_64)
  ARCH_SRCS := \
    native/arch/x86_64/probe_x86.c \
    native/arch/x86_64/isa_sse4.c \
    native/arch/x86_64/isa_avx2.c \
    native/arch/x86_64/isa_avx512.c \
    native/arch/x86_64/isa_amx.c

  HAVE_SSE4        := $(shell printf '$(HASH)include <immintrin.h>\nint f(void){return 0;}\n' | $(CC) -std=c11 -O2 -fPIC -msse4.2 -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_X86_AES     := $(shell printf '$(HASH)include <immintrin.h>\n__attribute__((target("aes"))) __m128i f(__m128i a){return _mm_aesenc_si128(a,a);}\n' | $(CC) -std=c11 -O2 -fPIC -msse4.2 -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_X86_SHA     := $(shell printf '$(HASH)include <immintrin.h>\n__attribute__((target("sha,ssse3,sse4.1"))) __m128i f(__m128i a){return _mm_sha256msg1_epu32(a,a);}\n' | $(CC) -std=c11 -O2 -fPIC -msse4.2 -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_AVX2        := $(shell printf '$(HASH)include <immintrin.h>\n__m256 f(__m256 a){return _mm256_fmadd_ps(a,a,a);}\n' | $(CC) -std=c11 -O2 -fPIC -mavx2 -mfma -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_X86_AVXVNNI := $(shell printf '$(HASH)include <immintrin.h>\n__attribute__((target("avx2,avxvnni"))) __m256i f(__m256i a){return _mm256_dpbusd_avx_epi32(a,a,a);}\n' | $(CC) -std=c11 -O2 -fPIC -mavx2 -mfma -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_AVX512      := $(shell printf '$(HASH)include <immintrin.h>\n__m512 f(__m512 a){return _mm512_fmadd_ps(a,a,a);}\n' | $(CC) -std=c11 -O2 -fPIC -mavx512f -mavx512bw -mavx512vl -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_512_VNNI    := $(shell printf '$(HASH)include <immintrin.h>\n__attribute__((target("avx512f,avx512vnni"))) __m512i f(__m512i a){return _mm512_dpbusd_epi32(a,a,a);}\n' | $(CC) -std=c11 -O2 -fPIC -mavx512f -mavx512bw -mavx512vl -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_512_BF16    := $(shell printf '$(HASH)include <immintrin.h>\n__attribute__((target("avx512f,avx512bf16"))) __m512 f(__m512 a,__m512bh b){return _mm512_dpbf16_ps(a,b,b);}\n' | $(CC) -std=c11 -O2 -fPIC -mavx512f -mavx512bw -mavx512vl -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_512_FP16    := $(shell printf '$(HASH)include <immintrin.h>\n__attribute__((target("avx512f,avx512fp16"))) __m512h f(__m512h a){return _mm512_fmadd_ph(a,a,a);}\n' | $(CC) -std=c11 -O2 -fPIC -mavx512f -mavx512bw -mavx512vl -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_AMX_BF16    := $(shell printf '$(HASH)include <immintrin.h>\nvoid f(void){_tile_dpbf16ps(0,1,2);}\n' | $(CC) -std=c11 -O2 -fPIC -mamx-tile -mamx-bf16 -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_AMX_INT8    := $(shell printf '$(HASH)include <immintrin.h>\nvoid f(void){_tile_dpbssd(0,1,2);}\n' | $(CC) -std=c11 -O2 -fPIC -mamx-tile -mamx-int8 -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)

  SSE4_FLAGS   := $(if $(filter 1,$(HAVE_SSE4)),-msse4.2 -DCROSS_CPU_HAVE_SSE4) \
                  $(if $(filter 1,$(HAVE_X86_AES)),-DCROSS_CPU_HAVE_X86_AES) \
                  $(if $(filter 1,$(HAVE_X86_SHA)),-DCROSS_CPU_HAVE_X86_SHA)
  AVX2_FLAGS   := $(if $(filter 1,$(HAVE_AVX2)),-mavx2 -mfma -DCROSS_CPU_HAVE_AVX2) \
                  $(if $(filter 1,$(HAVE_X86_AVXVNNI)),-DCROSS_CPU_HAVE_X86_AVXVNNI)
  AVX512_FLAGS := $(if $(filter 1,$(HAVE_AVX512)),-mavx512f -mavx512bw -mavx512vl -DCROSS_CPU_HAVE_AVX512) \
                  $(if $(filter 1,$(HAVE_512_VNNI)),-DCROSS_CPU_HAVE_X86_AVX512VNNI) \
                  $(if $(filter 1,$(HAVE_512_BF16)),-DCROSS_CPU_HAVE_X86_AVX512BF16) \
                  $(if $(filter 1,$(HAVE_512_FP16)),-DCROSS_CPU_HAVE_X86_AVX512FP16)
  AMX_FLAGS    := $(if $(filter 1,$(HAVE_AMX_BF16)),-mamx-tile -mamx-bf16 -DCROSS_CPU_HAVE_AMX -DCROSS_CPU_HAVE_AMX_BF16) \
                  $(if $(filter 1,$(HAVE_AMX_INT8)),-mamx-tile -mamx-int8 -DCROSS_CPU_HAVE_AMX -DCROSS_CPU_HAVE_AMX_INT8)

else ifeq ($(ARCH_FAMILY),arm64)
  ARCH_SRCS := \
    native/arch/arm64/probe_arm64.c \
    native/arch/arm64/isa_neon.c \
    native/arch/arm64/isa_sve.c \
    native/arch/arm64/isa_sme.c

  HAVE_ARM_FP16    := $(shell printf '$(HASH)include <arm_neon.h>\n__attribute__((target("fp16"))) float16x8_t f(float16x8_t a){return vfmaq_f16(a,a,a);}\n' | $(CC) -std=c11 -O2 -fPIC -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_ARM_BF16    := $(shell printf '$(HASH)include <arm_neon.h>\n__attribute__((target("bf16"))) float32x4_t f(float32x4_t a,bfloat16x8_t b){return vbfdotq_f32(a,b,b);}\n' | $(CC) -std=c11 -O2 -fPIC -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_ARM_DOTPROD := $(shell printf '$(HASH)include <arm_neon.h>\n__attribute__((target("dotprod"))) int32x4_t f(int32x4_t a,int8x16_t b){return vdotq_s32(a,b,b);}\n' | $(CC) -std=c11 -O2 -fPIC -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_ARM_I8MM    := $(shell printf '$(HASH)include <arm_neon.h>\n__attribute__((target("i8mm"))) int32x4_t f(int32x4_t a,int8x16_t b){return vmmlaq_s32(a,b,b);}\n' | $(CC) -std=c11 -O2 -fPIC -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_ARM_AES     := $(shell printf '$(HASH)include <arm_neon.h>\n__attribute__((target("aes"))) uint8x16_t f(uint8x16_t a){return vaesmcq_u8(vaeseq_u8(a,a));}\n' | $(CC) -std=c11 -O2 -fPIC -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_ARM_SHA2    := $(shell printf '$(HASH)include <arm_neon.h>\n__attribute__((target("sha2"))) uint32x4_t f(uint32x4_t a){return vsha256hq_u32(a,a,a);}\n' | $(CC) -std=c11 -O2 -fPIC -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_SVE         := $(shell printf '$(HASH)include <arm_sve.h>\nfloat f(void){return svaddv_f32(svptrue_b32(),svdup_f32(1.0f));}\n' | $(CC) -std=c11 -O2 -fPIC -march=armv8.2-a+sve -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_SME         := $(shell printf '$(HASH)include <arm_sme.h>\n__arm_new("za") void f(void) __arm_streaming {svzero_za();}\n' | $(CC) -std=c11 -O2 -fPIC -march=armv9-a+sme -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)

  NEON_FLAGS := $(if $(filter 1,$(HAVE_ARM_FP16)),-DCROSS_CPU_HAVE_ARM_FP16) \
                $(if $(filter 1,$(HAVE_ARM_BF16)),-DCROSS_CPU_HAVE_ARM_BF16) \
                $(if $(filter 1,$(HAVE_ARM_DOTPROD)),-DCROSS_CPU_HAVE_ARM_DOTPROD) \
                $(if $(filter 1,$(HAVE_ARM_I8MM)),-DCROSS_CPU_HAVE_ARM_I8MM) \
                $(if $(filter 1,$(HAVE_ARM_AES)),-DCROSS_CPU_HAVE_ARM_AES) \
                $(if $(filter 1,$(HAVE_ARM_SHA2)),-DCROSS_CPU_HAVE_ARM_SHA2)
  SVE_FLAGS  := $(if $(filter 1,$(HAVE_SVE)),-march=armv8.2-a+sve -DCROSS_CPU_HAVE_SVE)
  SME_FLAGS  := $(if $(filter 1,$(HAVE_SME)),-march=armv9-a+sme -DCROSS_CPU_HAVE_SME)

else ifeq ($(ARCH_FAMILY),riscv64)
  ARCH_SRCS := \
    native/arch/other/riscv64/probe_riscv64.c \
    native/arch/other/riscv64/isa_rvv.c
  HAVE_RVV  := $(shell printf '$(HASH)include <riscv_vector.h>\nsize_t f(void){return __riscv_vsetvl_e64m1(4);}\n' | $(CC) -std=c11 -O2 -fPIC -march=rv64gcv -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  RVV_FLAGS := $(if $(filter 1,$(HAVE_RVV)),-march=rv64gcv -DCROSS_CPU_HAVE_RVV)

else ifeq ($(ARCH_FAMILY),ppc64)
  ARCH_SRCS := \
    native/arch/other/ppc64/probe_ppc64.c \
    native/arch/other/ppc64/isa_vsx.c
  HAVE_VSX  := $(shell printf '$(HASH)include <altivec.h>\nint f(void){return 0;}\n' | $(CC) -std=c11 -O2 -fPIC -mvsx -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  VSX_FLAGS := $(if $(filter 1,$(HAVE_VSX)),-mvsx -DCROSS_CPU_HAVE_VSX)

else ifeq ($(ARCH_FAMILY),s390x)
  ARCH_SRCS := \
    native/arch/other/s390x/probe_s390x.c \
    native/arch/other/s390x/isa_vx.c
  HAVE_VX   := $(shell printf '$(HASH)include <vecintrin.h>\nint f(void){return 0;}\n' | $(CC) -std=c11 -O2 -fPIC -mzvector -march=z13 -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  HAVE_VXE  := $(shell printf '$(HASH)include <vecintrin.h>\n__vector float f(__vector float a){return vec_madd(a,a,a);}\n' | $(CC) -std=c11 -O2 -fPIC -mzvector -march=z14 -x c -c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
  VX_FLAGS  := $(if $(filter 1,$(HAVE_VXE)),-mzvector -march=z14 -DCROSS_CPU_HAVE_VX -DCROSS_CPU_HAVE_VXE,$(if $(filter 1,$(HAVE_VX)),-mzvector -march=z13 -DCROSS_CPU_HAVE_VX))

else
  ARCH_SRCS := native/arch/other/generic/probe_generic.c
endif

ARCH_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(ARCH_SRCS))
ALL_OBJS  := $(CORE_OBJS) $(COMMON_OBJS) $(ARCH_OBJS)

.PHONY: all build test install uninstall clean

all: build

build: $(TARGET_LIB)

$(TARGET_LIB): $(ALL_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(SHLIB_FLAGS) -o $@ $(ALL_OBJS) $(LDFLAGS) $(LDFLAGS_OS)
	@echo "Built $(TARGET_LIB) for arch=$(ARCH_FAMILY) ($(UNAME_S))"

# Per-ISA isolated translation unit rules (baseline flags for core/common/probe)
$(BUILD_DIR)/native/arch/x86_64/isa_sse4.o: native/arch/x86_64/isa_sse4.c
	@mkdir -p $(dir $@)
	$(CC) $(BASE_CFLAGS) $(SSE4_FLAGS) -c $< -o $@

$(BUILD_DIR)/native/arch/x86_64/isa_avx2.o: native/arch/x86_64/isa_avx2.c
	@mkdir -p $(dir $@)
	$(CC) $(BASE_CFLAGS) $(AVX2_FLAGS) -c $< -o $@

$(BUILD_DIR)/native/arch/x86_64/isa_avx512.o: native/arch/x86_64/isa_avx512.c
	@mkdir -p $(dir $@)
	$(CC) $(BASE_CFLAGS) $(AVX512_FLAGS) -c $< -o $@

$(BUILD_DIR)/native/arch/x86_64/isa_amx.o: native/arch/x86_64/isa_amx.c
	@mkdir -p $(dir $@)
	$(CC) $(BASE_CFLAGS) $(AMX_FLAGS) -c $< -o $@

$(BUILD_DIR)/native/arch/arm64/isa_neon.o: native/arch/arm64/isa_neon.c
	@mkdir -p $(dir $@)
	$(CC) $(BASE_CFLAGS) $(NEON_FLAGS) -c $< -o $@

$(BUILD_DIR)/native/arch/arm64/isa_sve.o: native/arch/arm64/isa_sve.c
	@mkdir -p $(dir $@)
	$(CC) $(BASE_CFLAGS) $(SVE_FLAGS) -c $< -o $@

$(BUILD_DIR)/native/arch/arm64/isa_sme.o: native/arch/arm64/isa_sme.c
	@mkdir -p $(dir $@)
	$(CC) $(BASE_CFLAGS) $(SME_FLAGS) -c $< -o $@

$(BUILD_DIR)/native/arch/other/riscv64/isa_rvv.o: native/arch/other/riscv64/isa_rvv.c
	@mkdir -p $(dir $@)
	$(CC) $(BASE_CFLAGS) $(RVV_FLAGS) -c $< -o $@

$(BUILD_DIR)/native/arch/other/ppc64/isa_vsx.o: native/arch/other/ppc64/isa_vsx.c
	@mkdir -p $(dir $@)
	$(CC) $(BASE_CFLAGS) $(VSX_FLAGS) -c $< -o $@

$(BUILD_DIR)/native/arch/other/s390x/isa_vx.o: native/arch/other/s390x/isa_vx.c
	@mkdir -p $(dir $@)
	$(CC) $(BASE_CFLAGS) $(VX_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(BASE_CFLAGS) -c $< -o $@

test: build
	$(PYTHON) -m compileall -q src tests
	$(PYTHON) -m unittest discover -s tests -v

install: build
	install -d $(DESTDIR)$(PREFIX)/lib/cross-cpu/cross_cpu
	install -d $(DESTDIR)$(PREFIX)/bin
	install -d $(DESTDIR)$(PREFIX)/share/man/man1
	install -d $(DESTDIR)$(PREFIX)/share/bash-completion/completions
	install -d $(DESTDIR)$(PREFIX)/share/zsh/site-functions
	install -d $(DESTDIR)$(PREFIX)/share/fish/vendor_completions.d
	install -m 644 src/*.py $(DESTDIR)$(PREFIX)/lib/cross-cpu/cross_cpu/
	install -m 755 $(TARGET_LIB) $(DESTDIR)$(PREFIX)/lib/cross-cpu/cross_cpu/
	printf '#!/usr/bin/env python3\n# SPDX-License-Identifier: MIT\nimport sys\nsys.path.insert(0, "%s/lib/cross-cpu")\nfrom cross_cpu.cli import main\nraise SystemExit(main())\n' "$(PREFIX)" > $(DESTDIR)$(PREFIX)/bin/cross-cpu
	chmod 755 $(DESTDIR)$(PREFIX)/bin/cross-cpu
	@if [ -f docs/cross-cpu.1 ]; then \
	  install -m 644 docs/cross-cpu.1 $(DESTDIR)$(PREFIX)/share/man/man1/cross-cpu.1; \
	fi
	install -m 644 completions/cross-cpu.bash $(DESTDIR)$(PREFIX)/share/bash-completion/completions/cross-cpu
	install -m 644 completions/_cross-cpu $(DESTDIR)$(PREFIX)/share/zsh/site-functions/_cross-cpu
	install -m 644 completions/cross-cpu.fish $(DESTDIR)$(PREFIX)/share/fish/vendor_completions.d/cross-cpu.fish

uninstall:
	rm -rf $(DESTDIR)$(PREFIX)/lib/cross-cpu
	rm -f $(DESTDIR)$(PREFIX)/bin/cross-cpu
	rm -f $(DESTDIR)$(PREFIX)/share/man/man1/cross-cpu.1
	rm -f $(DESTDIR)$(PREFIX)/share/bash-completion/completions/cross-cpu
	rm -f $(DESTDIR)$(PREFIX)/share/zsh/site-functions/_cross-cpu
	rm -f $(DESTDIR)$(PREFIX)/share/fish/vendor_completions.d/cross-cpu.fish

clean:
	rm -rf build src/_native.so src/_native.dylib src/_native.*.so src/_native.*.dylib
