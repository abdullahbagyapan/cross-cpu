#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# CrossCPU Docker-Only Multi-Architecture Emulation Test Harness
# ==============================================================
# TESTING-ONLY script (NOT part of the CrossCPU CLI tool or packaged distributions).
#
# Hard design rules:
#   - Docker is the ONLY dependency (zero host Python/C/make/QEMU/Rosetta assumptions).
#   - Targets every architecture backend in native/arch/:
#       x86_64 (linux/amd64), arm64 (linux/arm64), riscv64 (linux/riscv64),
#       ppc64le (linux/ppc64le), s390x (linux/s390x, big-endian), and generic.
#   - Exercises QEMU user-mode per-ISA gating (x86_64 + ARM64), multi-distro
#     libc/compiler hardening (Alpine/Debian/Fedora/Arch with GCC + Clang + wheel),
#     and adversarial container constraints (offline, read-only, UID 65534, cpuset).
#   - Mounts the repository strictly read-only (/repo:ro) and writes all per-step
#     logs + summary.md into an isolated log directory.
#   - Compatible with Bash 3.2+ and modern Linux Bash.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="crosscpu-emul-$$-$(date +%Y%m%d%H%M%S)"

# Defaults (no host OS/arch detection anywhere)
DURATION="150ms"
DISTRO_PLATFORM="linux/amd64"
LOG_ROOT=""
SKIP_BINFMT="0"
FAIL_FAST="0"
DRY_RUN="0"
VERBOSE="0"
LIST_ONLY="0"
BINFMT_DONE="0"
DOCKER_BIN=""
DOCKER_NCPU="1"
DOCKER_ARCH=""

SUITES=()
TARGETS=()
SELECTED_TARGETS=()
RESULTS_TARGET=()
RESULTS_STATUS=()
RESULTS_ELAPSED=()
RESULTS_DETAIL=()

ALL_TARGETS=(
  arch-amd64
  arch-arm64
  arch-riscv64
  arch-ppc64le
  arch-s390x
  arch-generic
  isa-x86_64
  isa-arm64
  distro-alpine
  distro-debian
  distro-fedora
  distro-arch
  adversarial
)

# ---------------------------------------------------------------------------
# Color & Formatting Helpers
# ---------------------------------------------------------------------------
USE_COLOR="0"
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  USE_COLOR="1"
fi

c_green()  { if [[ "${USE_COLOR}" == "1" ]]; then printf '\033[32;1m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }
c_red()    { if [[ "${USE_COLOR}" == "1" ]]; then printf '\033[31;1m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }
c_yellow() { if [[ "${USE_COLOR}" == "1" ]]; then printf '\033[33;1m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }
c_cyan()   { if [[ "${USE_COLOR}" == "1" ]]; then printf '\033[36;1m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }

decode_rc() {
  local code="$1"
  case "${code}" in
    124) echo "124 (TIMEOUT)" ;;
    132) echo "132 (SIGILL)" ;;
    134) echo "134 (SIGABRT)" ;;
    136) echo "136 (SIGFPE)" ;;
    137) echo "137 (SIGKILL/OOM)" ;;
    139) echo "139 (SIGSEGV)" ;;
    143) echo "143 (SIGTERM)" ;;
    *)
      if [[ "${code}" =~ ^[0-9]+$ ]] && (( code > 128 && code < 192 )); then
        echo "${code} (SIG$((code - 128)))"
      else
        echo "${code}"
      fi
      ;;
  esac
}

# ---------------------------------------------------------------------------
# Signal / Orphan Container Cleanup
# ---------------------------------------------------------------------------
cleanup_containers() {
  if [[ -n "${DOCKER_BIN}" ]]; then
    local ids=""
    ids="$("${DOCKER_BIN}" ps -q --filter "label=cross-cpu-emulate=${RUN_ID}" 2>/dev/null || true)"
    if [[ -n "${ids}" ]]; then
      printf '\nStopping active emulation container(s) for run %s...\n' "${RUN_ID}" >&2
      # shellcheck disable=SC2086
      "${DOCKER_BIN}" rm -f ${ids} >/dev/null 2>&1 || true
    fi
    "${DOCKER_BIN}" volume rm -f "${RUN_ID}-wheels" >/dev/null 2>&1 || true
  fi
}

on_interrupt() {
  cleanup_containers
  printf '\nInterrupted by signal. Partial logs (if any): %s\n' "${LOG_ROOT:-none}" >&2
  exit 130
}

trap on_interrupt INT TERM

# ---------------------------------------------------------------------------
# Target Metadata
# ---------------------------------------------------------------------------
target_suite() {
  case "$1" in
    arch-amd64|arch-arm64|arch-riscv64|arch-ppc64le|arch-s390x|arch-generic) echo "arch" ;;
    isa-x86_64|isa-arm64) echo "isa" ;;
    distro-alpine|distro-debian|distro-fedora|distro-arch) echo "distros" ;;
    adversarial) echo "adversarial" ;;
    *) return 1 ;;
  esac
}

target_desc() {
  case "$1" in
    arch-amd64)    echo "linux/amd64 (x86_64, little-endian) build, 39 unit tests, info/run JSON" ;;
    arch-arm64)    echo "linux/arm64 (arm64, little-endian) build, 39 unit tests, info/run JSON" ;;
    arch-riscv64)  echo "linux/riscv64 (riscv64, little-endian) build, 39 unit tests, info/run JSON" ;;
    arch-ppc64le)  echo "linux/ppc64le (ppc64le, little-endian) build, 39 unit tests, info/run JSON" ;;
    arch-s390x)    echo "linux/s390x (s390x, big-endian) build, 39 unit tests, info/run JSON" ;;
    arch-generic)  echo "ARCH=generic portable POSIX scalar-only backend build, 39 tests, zero-emulation gates" ;;
    isa-x86_64)    echo "linux/amd64 + qemu-x86_64 tiers (qemu64/Nehalem/Haswell/max) + 6 negative exit-2 gates" ;;
    isa-arm64)     echo "linux/arm64 + qemu-aarch64 tiers (cortex-a53/neon_only/sve_on) + 5 negative exit-2 gates" ;;
    distro-alpine) echo "alpine:latest (musl) GCC + Clang builds, 39 unit tests, offline wheel check" ;;
    distro-debian) echo "debian:trixie-slim hardened CFLAGS GCC + Clang builds, 39 unit tests, offline wheel" ;;
    distro-fedora) echo "fedora:latest (redhat-rpm-config) GCC + Clang builds, 39 unit tests, offline wheel" ;;
    distro-arch)   echo "archlinux:latest (pacman --disable-sandbox) GCC + Clang builds, 39 tests, wheel" ;;
    adversarial)   echo "python:3.12-slim offline, --read-only, UID 65534, non-contiguous cpuset container" ;;
    *) return 1 ;;
  esac
}

is_valid_target() {
  local cand="$1"
  local t
  for t in "${ALL_TARGETS[@]}"; do
    if [[ "${t}" == "${cand}" ]]; then
      return 0
    fi
  done
  return 1
}

add_target_unique() {
  local cand="$1"
  local existing
  if [[ "${#SELECTED_TARGETS[@]}" -gt 0 ]]; then
    for existing in "${SELECTED_TARGETS[@]}"; do
      if [[ "${existing}" == "${cand}" ]]; then
        return 0
      fi
    done
  fi
  SELECTED_TARGETS+=("${cand}")
}

# ---------------------------------------------------------------------------
# CLI Help & Listing
# ---------------------------------------------------------------------------
usage() {
  cat <<'EOF'
CrossCPU Docker-Only Multi-Architecture Emulation Test Harness (testing only)

Usage:
  ./scripts/emulate.sh [OPTIONS]

Options:
  --list                     List all emulation suites and targets
  --suite SUITE[,SUITE...]   Select suite(s): arch, isa, distros, adversarial, all
                             (default when no --suite/--target is passed: all)
  --target NAME[,NAME...]    Select specific target(s) (repeatable or comma-separated)
  --distro-platform PLAT     Docker --platform for distros and adversarial targets
                             [default: linux/amd64]
  --duration TIME            Timed CLI workload duration inside containers [default: 150ms]
  --log-dir DIR              Directory for per-step logs and summary.md
                             [default: ~/.cache/cross-cpu-emulation/<timestamp>]
  --skip-binfmt              Skip automatic tonistiigi/binfmt multi-arch QEMU registration
  --fail-fast                Stop immediately on the first failed target
  --dry-run                  Print selected targets and Docker settings without running
  -v, --verbose              Print container step output live in addition to log files
  -h, --help                 Show this help message

Suites & Targets:
  arch        : arch-amd64, arch-arm64, arch-riscv64, arch-ppc64le, arch-s390x, arch-generic
  isa         : isa-x86_64, isa-arm64
  distros     : distro-alpine, distro-debian, distro-fedora, distro-arch
  adversarial : adversarial
  all         : every target above

Examples:
  ./scripts/emulate.sh --list
  ./scripts/emulate.sh --suite arch
  ./scripts/emulate.sh --suite isa
  ./scripts/emulate.sh --target arch-s390x,arch-riscv64,arch-generic
  ./scripts/emulate.sh --suite distros --distro-platform linux/arm64
  ./scripts/emulate.sh --suite all
EOF
}

print_list() {
  printf '%s\n' "CrossCPU Docker-Only Emulation Targets (testing harness; Docker is the only dependency)"
  printf '%s\n' "======================================================================================"
  printf '%-16s %-13s %s\n' "TARGET" "SUITE" "DESCRIPTION"
  printf '%s\n' "--------------------------------------------------------------------------------------"
  local t
  for t in "${ALL_TARGETS[@]}"; do
    printf '%-16s %-13s %s\n' "${t}" "$(target_suite "${t}")" "$(target_desc "${t}")"
  done
}

# ---------------------------------------------------------------------------
# Docker Pre-Flight & binfmt_misc Setup (Zero Host Assumptions)
# ---------------------------------------------------------------------------
locate_and_verify_docker() {
  local extra_dir
  for extra_dir in "/opt/homebrew/bin" "/usr/local/bin"; do
    if [[ -d "${extra_dir}" && ":${PATH}:" != *":${extra_dir}:"* ]]; then
      PATH="${PATH}:${extra_dir}"
    fi
  done
  export PATH

  DOCKER_BIN="$(command -v docker 2>/dev/null || true)"
  if [[ -z "${DOCKER_BIN}" ]]; then
    printf 'Error: docker CLI not found on PATH.\n' >&2
    printf 'Docker is the sole dependency of this emulation test harness.\n' >&2
    return 2
  fi

  local info_out=""
  local info_err=""
  info_err="$(mktemp)"
  if ! info_out="$("${DOCKER_BIN}" info --format '{{.NCPU}}|{{.Architecture}}|{{.ServerVersion}}' 2>"${info_err}")"; then
    printf 'Error: unable to reach the Docker daemon via %s.\n' "${DOCKER_BIN}" >&2
    cat "${info_err}" >&2
    rm -f "${info_err}"
    return 2
  fi
  rm -f "${info_err}"

  local ncpu_raw="${info_out%%|*}"
  local rest="${info_out#*|}"
  DOCKER_ARCH="${rest%%|*}"
  local server_ver="${rest#*|}"

  if [[ "${ncpu_raw}" =~ ^[0-9]+$ ]] && (( ncpu_raw >= 1 )); then
    DOCKER_NCPU="${ncpu_raw}"
  else
    DOCKER_NCPU="1"
  fi

  printf 'Docker daemon ready: %s (server=%s, engine_arch=%s, engine_ncpu=%s)\n' \
    "${DOCKER_BIN}" "${server_ver}" "${DOCKER_ARCH}" "${DOCKER_NCPU}"
}

ensure_binfmt() {
  if [[ "${SKIP_BINFMT}" == "1" || "${BINFMT_DONE}" == "1" ]]; then
    return 0
  fi
  BINFMT_DONE="1"

  local bf_dir="${LOG_ROOT}/00_binfmt"
  mkdir -p "${bf_dir}"
  local probe_out="${bf_dir}/probe_binfmt.stdout.log"
  local probe_err="${bf_dir}/probe_binfmt.stderr.log"
  local probe_rc=0

  printf '==> Checking multi-arch binfmt_misc handlers (tonistiigi/binfmt)...\n'
  "${DOCKER_BIN}" run --privileged --rm \
    --label "cross-cpu-emulate=${RUN_ID}" \
    tonistiigi/binfmt >"${probe_out}" 2>"${probe_err}" || probe_rc=$?

  local missing=()
  if [[ "${probe_rc}" -eq 0 ]]; then
    local status_txt
    status_txt="$(cat "${probe_out}")"
    # Preserve native/Rosetta amd64 handling; only install architectures not already handled.
    if [[ "${status_txt}" != *'"linux/amd64"'* && "${status_txt}" != *'"rosetta"'* && "${status_txt}" != *'"qemu-x86_64"'* ]]; then
      missing+=("amd64")
    fi
    if [[ "${status_txt}" != *'"linux/arm64"'* && "${status_txt}" != *'"qemu-aarch64"'* ]]; then
      missing+=("arm64")
    fi
    if [[ "${status_txt}" != *'"linux/riscv64"'* && "${status_txt}" != *'"qemu-riscv64"'* ]]; then
      missing+=("riscv64")
    fi
    if [[ "${status_txt}" != *'"linux/ppc64le"'* && "${status_txt}" != *'"qemu-ppc64le"'* ]]; then
      missing+=("ppc64le")
    fi
    if [[ "${status_txt}" != *'"linux/s390x"'* && "${status_txt}" != *'"qemu-s390x"'* ]]; then
      missing+=("s390x")
    fi
  else
    missing=("amd64" "arm64" "riscv64" "ppc64le" "s390x")
  fi

  if [[ "${#missing[@]}" -eq 0 ]]; then
    printf '  [%s] all required binfmt_misc architectures already registered\n\n' "$(c_green PASS)"
    return 0
  fi

  local install_csv=""
  local arch_item
  for arch_item in "${missing[@]}"; do
    if [[ -z "${install_csv}" ]]; then
      install_csv="${arch_item}"
    else
      install_csv="${install_csv},${arch_item}"
    fi
  done

  local out="${bf_dir}/install_binfmt.stdout.log"
  local err="${bf_dir}/install_binfmt.stderr.log"
  printf '%s\n' "${DOCKER_BIN} run --privileged --rm --label cross-cpu-emulate=${RUN_ID} tonistiigi/binfmt --install ${install_csv}" \
    > "${bf_dir}/install_binfmt.cmd"
  printf '0\n' > "${bf_dir}/install_binfmt.expected_rc"

  printf '  Installing missing binfmt handler(s): %s\n' "${install_csv}"
  local rc=0
  "${DOCKER_BIN}" run --privileged --rm \
    --label "cross-cpu-emulate=${RUN_ID}" \
    tonistiigi/binfmt --install "${install_csv}" >"${out}" 2>"${err}" || rc=$?
  printf '%s\n' "${rc}" > "${bf_dir}/install_binfmt.rc"

  if [[ "${rc}" -eq 0 ]]; then
    printf '  [%s] binfmt_misc registered for %s\n\n' "$(c_green PASS)" "${install_csv}"
    if [[ "${VERBOSE}" == "1" && -s "${out}" ]]; then
      cat "${out}"
    fi
  else
    printf '  [%s] binfmt_misc registration returned rc=%s (continuing; if your daemon already has binfmt or uses rootless Docker, pass --skip-binfmt)\n' \
      "$(c_yellow WARN)" "${rc}" >&2
    if [[ -s "${err}" ]]; then
      tail -n 20 "${err}" >&2
    fi
    printf '\n'
  fi
}

# ---------------------------------------------------------------------------
# Shared In-Container Bash Logging & Diagnostic Prelude
# ---------------------------------------------------------------------------
emit_container_bash_prelude() {
  cat <<'EOF'
trap 'chmod -R a+rX /logs 2>/dev/null || true' EXIT

decode_rc() {
  local code="$1"
  case "${code}" in
    124) echo "124 (TIMEOUT)" ;;
    132) echo "132 (SIGILL)" ;;
    134) echo "134 (SIGABRT)" ;;
    136) echo "136 (SIGFPE)" ;;
    137) echo "137 (SIGKILL/OOM)" ;;
    139) echo "139 (SIGSEGV)" ;;
    143) echo "143 (SIGTERM)" ;;
    *)
      if (( code > 128 && code < 192 )); then
        echo "${code} (SIG$((code - 128)))"
      else
        echo "${code}"
      fi
      ;;
  esac
}

run_logged_expect() {
  local expected="$1"; shift
  local name="$1"; shift
  local out="/logs/${name}.stdout.log"
  local err="/logs/${name}.stderr.log"
  local rc=0
  printf '%s\n' "$*" > "/logs/${name}.cmd"
  printf '%s\n' "${expected}" > "/logs/${name}.expected_rc"
  "$@" >"${out}" 2>"${err}" || rc=$?
  printf '%s\n' "${rc}" > "/logs/${name}.rc"

  if [[ "${rc}" -eq "${expected}" ]]; then
    printf '  [PASS] %s (rc=%s)\n' "${name}" "${rc}"
    return 0
  fi

  local rc_desc
  rc_desc="$(decode_rc "${rc}")"
  printf '%s\n' "================================================================"
  printf '  [FAIL] %s\n' "${name}"
  printf '  Command : %s\n' "$*"
  printf '  Exit    : %s (expected %s)\n' "${rc_desc}" "${expected}"
  printf '%s\n' "  --- stdout (${name}.stdout.log) ---"
  if [[ -s "${out}" ]]; then cat "${out}"; else printf '%s\n' "  (empty)"; fi
  printf '%s\n' "  --- stderr (${name}.stderr.log) ---"
  if [[ -s "${err}" ]]; then cat "${err}"; else printf '%s\n' "  (empty)"; fi
  printf '%s\n' "================================================================"
  return 1
}

run_logged() {
  local name="$1"; shift
  run_logged_expect 0 "${name}" "$@"
}

run_cli_json_expect() {
  local expected="$1"; shift
  local name="$1"; shift
  if run_logged_expect "${expected}" "${name}" "$@"; then
    return 0
  fi
  if [[ ! -s "/logs/${name}.stdout.log" && ! -s "/logs/${name}.stderr.log" ]]; then
    local diag_cmd=()
    local arg
    for arg in "$@"; do
      if [[ "${arg}" != "--json" ]]; then
        diag_cmd+=("${arg}")
      fi
    done
    diag_cmd+=("-v")
    printf '%s\n' "  >>> Silent --json failure detected; diagnostic retry with -v for ${name}:"
    run_logged_expect 0 "${name}_diag_verbose" "${diag_cmd[@]}" || true
  fi
  return 1
}

assert_empty_stderr() {
  local name="$1"
  if [[ -s "/logs/${name}.stderr.log" ]]; then
    printf '  [FAIL] %s polluted stderr in --json mode:\n' "${name}" >&2
    cat "/logs/${name}.stderr.log" >&2
    return 1
  fi
}
EOF
}

# ---------------------------------------------------------------------------
# Target Runners (100% Docker, Zero Host Assumptions)
# ---------------------------------------------------------------------------
run_docker_script() {
  local target="$1"
  local log_dir="$2"
  local script_file="$3"
  shift 3
  # Remaining "$@" is the full `docker run ...` argument list (without `docker`)

  local container_out="${log_dir}/_container.stdout.log"
  local container_err="${log_dir}/_container.stderr.log"
  printf '%s\n' "${DOCKER_BIN} $*" > "${log_dir}/_container.cmd"
  printf '0\n' > "${log_dir}/_container.expected_rc"

  local rc=0
  if [[ "${VERBOSE}" == "1" ]]; then
    "${DOCKER_BIN}" "$@" < "${script_file}" > "${container_out}" 2> "${container_err}" || rc=$?
    cat "${container_out}"
    if [[ -s "${container_err}" ]]; then
      printf '%s\n' "--- container stderr (${target}) ---" >&2
      cat "${container_err}" >&2
    fi
  else
    "${DOCKER_BIN}" "$@" < "${script_file}" > "${container_out}" 2> "${container_err}" || rc=$?
    cat "${container_out}"
    if [[ "${rc}" -ne 0 && -s "${container_err}" ]]; then
      printf '%s\n' "--- container stderr (${target}) ---" >&2
      cat "${container_err}" >&2
    fi
  fi

  printf '%s\n' "${rc}" > "${log_dir}/_container.rc"
  return "${rc}"
}

run_target_arch() {
  local target="$1"
  local log_dir="$2"
  local plat=""
  local expected_arch=""
  local expected_endian=""
  local make_arch_arg=""

  case "${target}" in
    arch-amd64)   plat="linux/amd64";   expected_arch="x86_64";  expected_endian="little" ;;
    arch-arm64)   plat="linux/arm64";   expected_arch="arm64";   expected_endian="little" ;;
    arch-riscv64) plat="linux/riscv64"; expected_arch="riscv64"; expected_endian="little" ;;
    arch-ppc64le) plat="linux/ppc64le"; expected_arch="ppc64le"; expected_endian="little" ;;
    arch-s390x)   plat="linux/s390x";   expected_arch="s390x";   expected_endian="big" ;;
    arch-generic) plat="linux/amd64";   expected_arch="generic"; expected_endian="little"; make_arch_arg="ARCH=generic" ;;
    *) printf 'Internal error: unknown arch target %s\n' "${target}" >&2; return 1 ;;
  esac

  local script_file
  script_file="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 python3-setuptools gcc make libc6-dev'

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info src/_native.*

run_logged "01_env" uname -a
run_logged "02_cc_version" gcc --version
run_logged "03_compileall" python3 -m compileall -q src tests

if [[ -n "${MAKE_ARCH_ARG}" ]]; then
  run_logged "04_make_build" make build "${MAKE_ARCH_ARG}"
else
  run_logged "04_make_build" make build
fi

run_logged "05_unittest" python3 -m unittest discover -s tests -v
run_logged "06_info_verbose" python3 -m src info -v

run_cli_json_expect 0 "07_info_json" python3 -m src info --json
assert_empty_stderr "07_info_json"
run_logged "07_info_assert" python3 -c '
import json, os, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["command"] == "info", data
assert data["native"]["arch_name"] == os.environ["EXPECTED_ARCH"], data["native"]
assert data["native"]["endianness"] == os.environ["EXPECTED_ENDIAN"], data["native"]
if os.environ["EXPECTED_ARCH"] == "generic":
    names = [b["name"] for b in data["native"]["backends"]]
    assert names == ["scalar"], f"generic arch must expose scalar-only backend, got {names}"
print("OK info:", data["native"]["arch_name"], data["native"]["endianness"], data["native"]["probe_summary"])
' /logs/07_info_json.stdout.log

run_cli_json_expect 0 "08_run_json" python3 -m src run \
  --test compute-fp32,compute-int64,memory-pattern,coherency-atomic \
  --duration "${RUN_DURATION}" --json
assert_empty_stderr "08_run_json"
run_logged "08_run_assert" python3 -c '
import json, os, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["host"]["arch"] == os.environ["EXPECTED_ARCH"], data["host"]
assert data["host"]["endianness"] == os.environ["EXPECTED_ENDIAN"], data["host"]
assert data["summary"]["exit_code"] == 0, data["summary"]
assert data["summary"]["passed"] == 4, data["summary"]
assert data["summary"]["failed"] == 0, data["summary"]
for r in data["results"]:
    assert r["status"] == "passed", r
    assert r["error_count"] == 0, r
    assert r["raw_ops"] > 0, r
    if os.environ["EXPECTED_ARCH"] == "generic":
        assert r["isa"] == "scalar", r
print("OK run:", [(r["name"], r["isa"], round(r["rate"], 3)) for r in data["results"]])
' /logs/08_run_json.stdout.log

run_cli_json_expect 0 "09_run_all_json" python3 -m src run --test all --duration 60ms --json
assert_empty_stderr "09_run_all_json"
run_logged "09_run_all_assert" python3 -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["summary"]["exit_code"] == 0, data["summary"]
assert data["summary"]["failed"] == 0, data["summary"]
assert data["summary"]["init_error"] == 0, data["summary"]
assert data["summary"]["passed"] >= 12, data["summary"]
print("OK --test all:", data["summary"])
' /logs/09_run_all_json.stdout.log

if [[ "${EXPECTED_ARCH}" == "generic" ]]; then
  run_cli_json_expect 2 "10_neg_generic_crypto_aes" python3 -m src run --test crypto-aes --duration 60ms --json
  assert_empty_stderr "10_neg_generic_crypto_aes"
  run_logged "10_neg_generic_crypto_aes_assert" python3 -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["summary"]["exit_code"] == 2, data["summary"]
assert data["results"][0]["status"] == "unsupported", data["results"]
print("OK generic zero-emulation gate:", data["results"][0]["reason"])
' /logs/10_neg_generic_crypto_aes.stdout.log
fi
EOF
  } > "${script_file}"

  local rc=0
  run_docker_script "${target}" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-cpu-emulate=${RUN_ID}" \
    --platform "${plat}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "EXPECTED_ARCH=${expected_arch}" \
    -e "EXPECTED_ENDIAN=${expected_endian}" \
    -e "MAKE_ARCH_ARG=${make_arch_arg}" \
    -e "RUN_DURATION=${DURATION}" \
    debian:trixie-slim bash -euo pipefail || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

run_target_isa_x86_64() {
  local log_dir="$1"
  local script_file
  script_file="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 gcc make libc6-dev qemu-user'

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info src/_native.*
PY="/usr/bin/python3"

run_logged "00_qemu_version" qemu-x86_64 --version
run_logged "01_make_build" make build

for spec in "qemu64:scalar" "Nehalem:sse4" "Haswell:avx2" "max:avx2,avx512"; do
  cpu="${spec%%:*}"
  expected_isas="${spec#*:}"
  slug="$(printf '%s' "${cpu}" | tr '[:upper:]' '[:lower:]')"

  run_logged "02_${slug}_info_verbose" qemu-x86_64 -cpu "${cpu}" "${PY}" -m src info -v
  run_cli_json_expect 0 "02_${slug}_info_json" qemu-x86_64 -cpu "${cpu}" "${PY}" -m src info --json
  run_logged "02_${slug}_info_assert" "${PY}" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["command"] == "info", data
assert data["native"]["arch_name"] == "x86_64", data["native"]
assert data["native"]["endianness"] == "little", data["native"]
print("OK info:", data["native"]["probe_summary"])
' "/logs/02_${slug}_info_json.stdout.log"

  run_cli_json_expect 0 "03_${slug}_run_json" qemu-x86_64 -cpu "${cpu}" "${PY}" -m src run \
    --test compute-fp32,compute-int64,memory-pattern,coherency-atomic \
    --duration "${RUN_DURATION}" --json
  run_logged "03_${slug}_run_assert" "${PY}" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
allowed = set(sys.argv[2].split(","))
assert data["summary"]["exit_code"] == 0, data["summary"]
assert data["summary"]["passed"] == 4, data["summary"]
assert data["summary"]["failed"] == 0, data["summary"]
by_name = {r["name"]: r for r in data["results"]}
for r in data["results"]:
    assert r["status"] == "passed", r
    assert r["error_count"] == 0, r
    assert r["raw_ops"] > 0, r
fp32_isa = by_name["compute-fp32"]["isa"]
assert fp32_isa in allowed, f"expected compute-fp32 ISA in {sorted(allowed)}, got {fp32_isa!r}"
print(f"OK run: compute-fp32 isa={fp32_isa}")
' "/logs/03_${slug}_run_json.stdout.log" "${expected_isas}"
done

run_cli_json_expect 0 "03_max_run_all_json" qemu-x86_64 -cpu "max" "${PY}" -m src run \
  --test all --duration 80ms --json
run_logged "03_max_run_all_assert" "${PY}" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["summary"]["exit_code"] == 0, data["summary"]
assert data["summary"]["failed"] == 0, data["summary"]
assert data["summary"]["init_error"] == 0, data["summary"]
assert data["summary"]["passed"] >= 12, data["summary"]
print("OK --test all:", data["summary"])
' "/logs/03_max_run_all_json.stdout.log"

assert_neg_json() {
  local log_name="$1"
  local expected_status="$2"
  run_logged "${log_name}_assert" "${PY}" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
exp_status = sys.argv[2]
assert data["summary"]["exit_code"] == 2, data["summary"]
assert len(data["results"]) == 1, data["results"]
assert data["results"][0]["status"] == exp_status, data["results"][0]
assert data["results"][0]["reason"], data["results"][0]
print("OK negative gate:", data["results"][0]["status"], "-", data["results"][0]["reason"])
' "/logs/${log_name}.stdout.log" "${expected_status}"
}

run_cli_json_expect 2 "04_neg_qemu64_sse4" qemu-x86_64 -cpu "qemu64" "${PY}" -m src run --isa sse4 --test compute-fp32 --duration 100ms --json
assert_neg_json "04_neg_qemu64_sse4" "init_error"
run_cli_json_expect 2 "04_neg_qemu64_avx2" qemu-x86_64 -cpu "qemu64" "${PY}" -m src run --isa avx2 --test compute-fp32 --duration 100ms --json
assert_neg_json "04_neg_qemu64_avx2" "init_error"
run_cli_json_expect 2 "04_neg_qemu64_crypto_aes" qemu-x86_64 -cpu "qemu64" "${PY}" -m src run --test crypto-aes --duration 100ms --json
assert_neg_json "04_neg_qemu64_crypto_aes" "unsupported"
run_cli_json_expect 2 "04_neg_nehalem_avx2" qemu-x86_64 -cpu "Nehalem" "${PY}" -m src run --isa avx2 --test compute-fp32 --duration 100ms --json
assert_neg_json "04_neg_nehalem_avx2" "init_error"
run_cli_json_expect 2 "04_neg_haswell_avx512" qemu-x86_64 -cpu "Haswell" "${PY}" -m src run --isa avx512 --test compute-fp32 --duration 100ms --json
assert_neg_json "04_neg_haswell_avx512" "init_error"
run_cli_json_expect 2 "04_neg_haswell_amx" qemu-x86_64 -cpu "Haswell" "${PY}" -m src run --isa amx --test compute-matrix-bf16 --duration 100ms --json
assert_neg_json "04_neg_haswell_amx" "init_error"
EOF
  } > "${script_file}"

  local rc=0
  run_docker_script "isa-x86_64" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-cpu-emulate=${RUN_ID}" \
    --platform "linux/amd64" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "RUN_DURATION=${DURATION}" \
    debian:trixie-slim bash -euo pipefail || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

run_target_isa_arm64() {
  local log_dir="$1"
  local script_file
  script_file="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 gcc make libc6-dev qemu-user'

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info src/_native.*
PY="/usr/bin/python3"

run_logged "00_qemu_version" qemu-aarch64 --version
run_logged "01_make_build" make build

for spec in \
  "cortex_a53:cortex-a53:neon" \
  "neon_only:max,sve=off,sme=off:neon" \
  "sve_on:max,sve=on,sme=off:sve_if_compiled"; do
  slug="${spec%%:*}"
  rest="${spec#*:}"
  cpu="${rest%%:*}"
  expected_mode="${rest#*:}"

  run_logged "02_${slug}_info_verbose" qemu-aarch64 -cpu "${cpu}" "${PY}" -m src info -v
  run_cli_json_expect 0 "02_${slug}_info_json" qemu-aarch64 -cpu "${cpu}" "${PY}" -m src info --json
  run_logged "02_${slug}_info_assert" "${PY}" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["command"] == "info", data
assert data["native"]["arch_name"] == "arm64", data["native"]
assert data["native"]["endianness"] == "little", data["native"]
print("OK info:", data["native"]["probe_summary"])
' "/logs/02_${slug}_info_json.stdout.log"

  run_cli_json_expect 0 "03_${slug}_run_json" qemu-aarch64 -cpu "${cpu}" "${PY}" -m src run \
    --test compute-fp32,compute-int64,memory-pattern,coherency-atomic \
    --duration "${RUN_DURATION}" --json
  run_logged "03_${slug}_run_assert" "${PY}" -c '
import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
run_data = json.load(open(sys.argv[2], encoding="utf-8"))
mode = sys.argv[3]
assert run_data["summary"]["exit_code"] == 0, run_data["summary"]
assert run_data["summary"]["passed"] == 4, run_data["summary"]
assert run_data["summary"]["failed"] == 0, run_data["summary"]
for r in run_data["results"]:
    assert r["status"] == "passed", r
    assert r["error_count"] == 0, r
    assert r["raw_ops"] > 0, r
backends = {b["name"]: b for b in info["native"]["backends"]}
expected_isa = ("sve" if backends.get("sve", {}).get("compiled_in") else "neon") if mode == "sve_if_compiled" else mode
by_name = {r["name"]: r for r in run_data["results"]}
got_isa = by_name["compute-fp32"]["isa"]
assert got_isa == expected_isa, f"expected compute-fp32 isa={expected_isa!r}, got {got_isa!r}"
print(f"OK run ({mode}): compute-fp32 isa={got_isa}")
' "/logs/02_${slug}_info_json.stdout.log" "/logs/03_${slug}_run_json.stdout.log" "${expected_mode}"
done

assert_neg_json() {
  local log_name="$1"
  local expected_status="$2"
  run_logged "${log_name}_assert" "${PY}" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
exp_status = sys.argv[2]
assert data["summary"]["exit_code"] == 2, data["summary"]
assert len(data["results"]) == 1, data["results"]
assert data["results"][0]["status"] == exp_status, data["results"][0]
assert data["results"][0]["reason"], data["results"][0]
print("OK negative gate:", data["results"][0]["status"], "-", data["results"][0]["reason"])
' "/logs/${log_name}.stdout.log" "${expected_status}"
}

run_cli_json_expect 2 "04_neg_cortex_a53_fp16" qemu-aarch64 -cpu "cortex-a53" "${PY}" -m src run --test compute-fp16 --duration 100ms --json
assert_neg_json "04_neg_cortex_a53_fp16" "unsupported"
run_cli_json_expect 2 "04_neg_cortex_a53_sve" qemu-aarch64 -cpu "cortex-a53" "${PY}" -m src run --isa sve --test compute-fp32 --duration 100ms --json
assert_neg_json "04_neg_cortex_a53_sve" "init_error"
run_cli_json_expect 2 "04_neg_neon_only_sve" qemu-aarch64 -cpu "max,sve=off,sme=off" "${PY}" -m src run --isa sve --test compute-fp32 --duration 100ms --json
assert_neg_json "04_neg_neon_only_sve" "init_error"
run_cli_json_expect 2 "04_neg_neon_only_sme" qemu-aarch64 -cpu "max,sve=off,sme=off" "${PY}" -m src run --isa sme --test compute-matrix-fp32 --duration 100ms --json
assert_neg_json "04_neg_neon_only_sme" "init_error"
run_cli_json_expect 2 "04_neg_sve_on_sme" qemu-aarch64 -cpu "max,sve=on,sme=off" "${PY}" -m src run --isa sme --test compute-matrix-fp32 --duration 100ms --json
assert_neg_json "04_neg_sve_on_sme" "init_error"
EOF
  } > "${script_file}"

  local rc=0
  run_docker_script "isa-arm64" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-cpu-emulate=${RUN_ID}" \
    --platform "linux/arm64" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "RUN_DURATION=${DURATION}" \
    debian:trixie-slim bash -euo pipefail || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

run_target_distro() {
  local target="$1"
  local log_dir="$2"
  local image=""
  local install_cmd=""
  local cflags=""

  case "${target}" in
    distro-alpine)
      image="alpine:latest"
      install_cmd="apk add --no-cache python3 python3-dev py3-setuptools py3-wheel py3-build gcc clang musl-dev make bash"
      cflags="-O2 -Wall -Wextra"
      ;;
    distro-debian)
      image="debian:trixie-slim"
      install_cmd="apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 python3-dev python3-setuptools python3-wheel python3-build gcc clang make libc6-dev"
      cflags="-O2 -D_FORTIFY_SOURCE=3 -fstack-protector-strong -Werror=format-security"
      ;;
    distro-fedora)
      image="fedora:latest"
      install_cmd="dnf install -y -q python3 python3-devel python3-setuptools python3-wheel python3-build gcc clang make redhat-rpm-config"
      cflags="-O2 -D_FORTIFY_SOURCE=3 -fstack-protector-strong"
      ;;
    distro-arch)
      image="archlinux:latest"
      install_cmd="pacman --disable-sandbox -Syu --noconfirm --needed python python-setuptools python-wheel python-build gcc clang make"
      cflags="-O2"
      ;;
    *)
      printf 'Internal error: unknown distro target %s\n' "${target}" >&2
      return 1
      ;;
  esac

  local script_file
  script_file="$(mktemp)"
  cat <<'EOF' > "${script_file}"
trap 'chmod -R a+rX /logs 2>/dev/null || true' EXIT

run_logged() {
  _name="$1"; shift
  _out="/logs/${_name}.stdout.log"
  _err="/logs/${_name}.stderr.log"
  _rc=0
  printf '%s\n' "$*" > "/logs/${_name}.cmd"
  printf '0\n' > "/logs/${_name}.expected_rc"
  "$@" >"${_out}" 2>"${_err}" || _rc=$?
  printf '%s\n' "${_rc}" > "/logs/${_name}.rc"

  if [ "${_rc}" -eq 0 ]; then
    printf '  [PASS] %s (rc=0)\n' "${_name}"
    return 0
  fi

  printf '%s\n' "================================================================"
  printf '  [FAIL] %s\n' "${_name}"
  printf '  Command : %s\n' "$*"
  printf '  Exit    : %s (expected 0)\n' "${_rc}"
  printf '%s\n' "  --- stdout (${_name}.stdout.log) ---"
  if [ -s "${_out}" ]; then cat "${_out}"; else printf '%s\n' "  (empty)"; fi
  printf '%s\n' "  --- stderr (${_name}.stderr.log) ---"
  if [ -s "${_err}" ]; then cat "${_err}"; else printf '%s\n' "  (empty)"; fi
  printf '%s\n' "================================================================"
  return "${_rc}"
}

run_logged "00_install_deps" sh -eu -c "${INSTALL_CMD}"

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info src/_native.*

run_logged "01_gcc_version" gcc --version
run_logged "02_build_gcc" env CFLAGS="${EXTRA_CFLAGS}" make build CC=gcc
run_logged "03_unittest_gcc" python3 -m unittest discover -s tests -v

make clean || true
run_logged "04_clang_version" clang --version
run_logged "05_build_clang" env CFLAGS="${EXTRA_CFLAGS}" make build CC=clang
run_logged "06_unittest_clang" python3 -m unittest discover -s tests -v

make clean || true
run_logged "07_wheel_offline" python3 -m build --wheel --no-isolation --outdir /tmp/dist .
run_logged "08_verify_wheel" python3 -c '
import glob, zipfile
wheels = glob.glob("/tmp/dist/cross_cpu-*.whl")
assert len(wheels) == 1, f"Expected 1 wheel, found: {wheels}"
with zipfile.ZipFile(wheels[0]) as zf:
    names = zf.namelist()
assert any("cross_cpu/_native" in n and n.endswith(".so") for n in names), names
print("OK wheel:", wheels[0], "entries:", len(names))
'
EOF

  local rc=0
  run_docker_script "${target}" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-cpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "INSTALL_CMD=${install_cmd}" \
    -e "EXTRA_CFLAGS=${cflags}" \
    "${image}" sh -eu || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

run_target_adversarial() {
  local log_dir="$1"
  local cpuset="0"
  if (( DOCKER_NCPU >= 3 )); then
    cpuset="0,2"
  elif (( DOCKER_NCPU == 2 )); then
    cpuset="1"
  fi

  local wheel_vol="${RUN_ID}-wheels"
  "${DOCKER_BIN}" volume rm -f "${wheel_vol}" >/dev/null 2>&1 || true
  "${DOCKER_BIN}" volume create "${wheel_vol}" >/dev/null

  local build_script
  build_script="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
trap 'chmod -R a+rX /wheels /logs 2>/dev/null || true' EXIT
export DEBIAN_FRONTEND=noninteractive
run_logged "00_builder_apt" bash -euo pipefail -c \
  'apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends gcc libc6-dev'
run_logged "01_builder_pip_tools" python3 -m pip install --quiet --no-cache-dir setuptools wheel build

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info src/_native.*
run_logged "02_builder_wheel" python3 -m build --wheel --no-isolation --outdir /wheels .
EOF
  } > "${build_script}"

  local rc=0
  run_docker_script "adversarial-builder" "${log_dir}" "${build_script}" \
    run -i --rm \
    --label "cross-cpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${wheel_vol}:/wheels:rw" \
    -v "${log_dir}:/logs:rw" \
    python:3.12-slim bash -euo pipefail || rc=$?
  rm -f "${build_script}"

  if [[ "${rc}" -ne 0 ]]; then
    "${DOCKER_BIN}" volume rm -f "${wheel_vol}" >/dev/null 2>&1 || true
    return "${rc}"
  fi

  local adv_script
  adv_script="$(mktemp)"
  cat <<'EOF' > "${adv_script}"
run_inner() {
  local name="$1"; shift
  local out="/tmp/${name}.stdout.log"
  local err="/tmp/${name}.stderr.log"
  local irc=0
  "$@" >"${out}" 2>"${err}" || irc=$?
  if [[ "${irc}" -eq 0 ]]; then
    printf '  [PASS] %s (rc=0)\n' "${name}"
    return 0
  fi
  printf '%s\n' "================================================================"
  printf '  [FAIL] %s\n' "${name}"
  printf '  Command : %s\n' "$*"
  printf '  Exit    : %s (expected 0)\n' "${irc}"
  printf '%s\n' "  --- stdout (${name}) ---"
  if [[ -s "${out}" ]]; then cat "${out}"; else printf '%s\n' "  (empty)"; fi
  printf '%s\n' "  --- stderr (${name}) ---"
  if [[ -s "${err}" ]]; then cat "${err}"; else printf '%s\n' "  (empty)"; fi
  printf '%s\n' "================================================================"
  return "${irc}"
}

assert_empty_stderr() {
  local name="$1"
  if [[ -s "/tmp/${name}.stderr.log" ]]; then
    printf '  [FAIL] %s polluted stderr in --json mode:\n' "${name}" >&2
    cat "/tmp/${name}.stderr.log" >&2
    return 1
  fi
}

run_inner "01_venv_create" python3 -m venv /tmp/venv
run_inner "02_pip_install_offline" /tmp/venv/bin/pip install --no-index --no-deps /wheels/*.whl
cd /tmp

run_inner "03_version" /tmp/venv/bin/cross-cpu --version
run_inner "04_info_verbose" /tmp/venv/bin/cross-cpu info -v

run_inner "05_list_json" /tmp/venv/bin/cross-cpu list --json
assert_empty_stderr "05_list_json"
run_inner "05_list_json_check" python3 -m json.tool /tmp/05_list_json.stdout.log

run_inner "06_info_json" /tmp/venv/bin/cross-cpu info --json
assert_empty_stderr "06_info_json"
run_inner "06_info_json_check" python3 -m json.tool /tmp/06_info_json.stdout.log

run_inner "07_run_cpuset_json" /tmp/venv/bin/cross-cpu run \
  --cores "${CPUSET}" \
  --test compute-fp32,compute-int64,cache-l1,memory-pattern,coherency-atomic \
  --duration "${RUN_DURATION}" --json
assert_empty_stderr "07_run_cpuset_json"
run_inner "07_run_cpuset_assert" python3 -c '
import json, os, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
expected_ids = sorted(int(x) for x in os.environ["CPUSET"].split(","))
assert data["config"]["selected_core_ids"] == expected_ids, data["config"]
assert data["summary"]["exit_code"] == 0, data["summary"]
assert data["summary"]["passed"] == 5, data["summary"]
assert data["summary"]["failed"] == 0, data["summary"]
print("OK cpuset run:", expected_ids, data["summary"])
' /tmp/07_run_cpuset_json.stdout.log

run_inner "08_run_all_json" /tmp/venv/bin/cross-cpu run \
  --cores all \
  --test compute-fp32,compute-int64,cache-l1,memory-pattern,coherency-atomic \
  --duration "${RUN_DURATION}" --json
assert_empty_stderr "08_run_all_json"
run_inner "08_run_all_assert" python3 -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data["summary"]["exit_code"] == 0, data["summary"]
assert data["summary"]["passed"] == 5, data["summary"]
assert data["summary"]["failed"] == 0, data["summary"]
print("OK --cores all run:", data["summary"])
' /tmp/08_run_all_json.stdout.log
EOF

  local adv_out="${log_dir}/03_adversarial_container.stdout.log"
  local adv_err="${log_dir}/03_adversarial_container.stderr.log"
  printf '%s\n' "${DOCKER_BIN} run --platform ${DISTRO_PLATFORM} --network=none --read-only --user 65534:65534 --cpuset-cpus=${cpuset} python:3.12-slim" \
    > "${log_dir}/03_adversarial_container.cmd"
  printf '0\n' > "${log_dir}/03_adversarial_container.expected_rc"

  rc=0
  "${DOCKER_BIN}" run -i --rm \
    --label "cross-cpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    --network=none \
    --read-only \
    --tmpfs /tmp:rw,exec,mode=1777,size=256m \
    --user 65534:65534 \
    --cpuset-cpus="${cpuset}" \
    -e HOME=/tmp \
    -e XDG_CACHE_HOME=/tmp/cache \
    -e TMPDIR=/tmp \
    -e PIP_NO_CACHE_DIR=1 \
    -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
    -e "CPUSET=${cpuset}" \
    -e "RUN_DURATION=${DURATION}" \
    -v "${wheel_vol}:/wheels:ro" \
    python:3.12-slim bash -euo pipefail < "${adv_script}" > "${adv_out}" 2> "${adv_err}" || rc=$?

  printf '%s\n' "${rc}" > "${log_dir}/03_adversarial_container.rc"
  cat "${adv_out}"
  if [[ -s "${adv_err}" ]]; then
    printf '%s\n' "--- adversarial stderr ---" >&2
    cat "${adv_err}" >&2
  fi

  rm -f "${adv_script}"
  "${DOCKER_BIN}" volume rm -f "${wheel_vol}" >/dev/null 2>&1 || true
  return "${rc}"
}

# ---------------------------------------------------------------------------
# Markdown Report Generator (Zero Host Python Dependency)
# ---------------------------------------------------------------------------
write_summary_markdown() {
  local report="${LOG_ROOT}/summary.md"
  {
    printf '%s\n' "# CrossCPU Docker Emulation Test Report"
    printf '\n'
    printf -- '- **Run ID:** `%s`\n' "${RUN_ID}"
    printf -- '- **Timestamp:** %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf -- '- **Docker Engine:** arch=`%s`, ncpu=`%s`\n' "${DOCKER_ARCH}" "${DOCKER_NCPU}"
    printf -- '- **Distro/Adversarial Platform:** `%s`\n' "${DISTRO_PLATFORM}"
    printf -- '- **Log Directory:** `%s`\n' "${LOG_ROOT}"
    printf '\n'
    printf '%s\n' "## Target Summary"
    printf '\n'
    printf '%s\n' "| Target | Status | Duration | Detail |"
    printf '%s\n' "|---|---|---:|---|"

    local idx=0
    local total="${#RESULTS_TARGET[@]}"
    local failed_targets=()
    while (( idx < total )); do
      local t="${RESULTS_TARGET[${idx}]}"
      local st="${RESULTS_STATUS[${idx}]}"
      local el="${RESULTS_ELAPSED[${idx}]}"
      local det="${RESULTS_DETAIL[${idx}]}"
      printf '| `%s` | **%s** | %ss | %s |\n' "${t}" "${st}" "${el}" "${det}"
      if [[ "${st}" == "FAIL" ]]; then
        failed_targets+=("${t}")
      fi
      idx=$((idx + 1))
    done

    if [[ "${#failed_targets[@]}" -gt 0 ]]; then
      printf '\n'
      printf '%s\n' "## Failure Details"
      local ft
      for ft in "${failed_targets[@]}"; do
        printf '\n'
        printf '### `%s`\n' "${ft}"
        local tdir="${LOG_ROOT}/${ft}"
        local found_step="0"
        local rc_file
        for rc_file in "${tdir}"/*.rc; do
          [[ -f "${rc_file}" ]] || continue
          local sname
          sname="$(basename "${rc_file}" .rc)"
          local src_rc
          src_rc="$(cat "${rc_file}" 2>/dev/null || echo "")"
          local exp_rc
          exp_rc="$(cat "${tdir}/${sname}.expected_rc" 2>/dev/null || echo 0)"
          if [[ -z "${src_rc}" || "${src_rc}" != "${exp_rc}" ]]; then
            found_step="1"
            local scmd
            scmd="$(cat "${tdir}/${sname}.cmd" 2>/dev/null || echo "(unknown)")"
            printf '\n'
            printf '**`%s`** — exit `%s` (expected `%s`)\n' "${sname}" "$(decode_rc "${src_rc:-1}")" "${exp_rc}"
            printf '%s\n' '```text'
            printf 'Command: %s\n' "${scmd}"
            printf '%s\n' "--- stdout (tail -n 200) ---"
            if [[ -s "${tdir}/${sname}.stdout.log" ]]; then
              tail -n 200 "${tdir}/${sname}.stdout.log"
            else
              printf '%s\n' "(empty)"
            fi
            printf '%s\n' "--- stderr (tail -n 200) ---"
            if [[ -s "${tdir}/${sname}.stderr.log" ]]; then
              tail -n 200 "${tdir}/${sname}.stderr.log"
            else
              printf '%s\n' "(empty)"
            fi
            printf '%s\n' '```'
          fi
        done
        if [[ "${found_step}" == "0" ]]; then
          printf -- '- Target exited non-zero without per-step failure record; check `%s/`.\n' "${tdir}"
        fi
      done
    fi
  } > "${report}"
  printf '%s' "${report}"
}

# ---------------------------------------------------------------------------
# Argument Parsing & Dispatch
# ---------------------------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --list)
      LIST_ONLY="1"
      shift
      ;;
    --suite)
      [[ "$#" -ge 2 ]] || { printf 'Error: --suite requires an argument\n' >&2; exit 1; }
      SUITES+=("$2")
      shift 2
      ;;
    --target)
      [[ "$#" -ge 2 ]] || { printf 'Error: --target requires an argument\n' >&2; exit 1; }
      TARGETS+=("$2")
      shift 2
      ;;
    --distro-platform)
      [[ "$#" -ge 2 ]] || { printf 'Error: --distro-platform requires an argument\n' >&2; exit 1; }
      DISTRO_PLATFORM="$2"
      shift 2
      ;;
    --duration)
      [[ "$#" -ge 2 ]] || { printf 'Error: --duration requires an argument\n' >&2; exit 1; }
      DURATION="$2"
      shift 2
      ;;
    --log-dir)
      [[ "$#" -ge 2 ]] || { printf 'Error: --log-dir requires an argument\n' >&2; exit 1; }
      LOG_ROOT="$2"
      shift 2
      ;;
    --skip-binfmt)
      SKIP_BINFMT="1"
      shift
      ;;
    --fail-fast)
      FAIL_FAST="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -v|--verbose)
      VERBOSE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Error: unknown option %s (use --help)\n' "$1" >&2
      exit 1
      ;;
  esac
done

if [[ "${LIST_ONLY}" == "1" ]]; then
  print_list
  exit 0
fi

if [[ "${#SUITES[@]}" -eq 0 && "${#TARGETS[@]}" -eq 0 ]]; then
  SUITES=("all")
fi

# Resolve suites -> targets (Bash 3.2 safe)
if [[ "${#SUITES[@]}" -gt 0 ]]; then
  for raw_suite in "${SUITES[@]}"; do
    IFS=',' read -r -a suite_parts <<< "${raw_suite}"
    for s in "${suite_parts[@]}"; do
      s="$(printf '%s' "${s}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
      [[ -n "${s}" ]] || continue
      case "${s}" in
        all)
          for t in "${ALL_TARGETS[@]}"; do add_target_unique "${t}"; done
          ;;
        arch)
          for t in arch-amd64 arch-arm64 arch-riscv64 arch-ppc64le arch-s390x arch-generic; do add_target_unique "${t}"; done
          ;;
        isa)
          for t in isa-x86_64 isa-arm64; do add_target_unique "${t}"; done
          ;;
        distros)
          for t in distro-alpine distro-debian distro-fedora distro-arch; do add_target_unique "${t}"; done
          ;;
        adversarial)
          add_target_unique "adversarial"
          ;;
        *)
          printf 'Error: unknown --suite "%s" (valid: arch, isa, distros, adversarial, all)\n' "${s}" >&2
          exit 1
          ;;
      esac
    done
  done
fi

if [[ "${#TARGETS[@]}" -gt 0 ]]; then
  for raw_target in "${TARGETS[@]}"; do
    IFS=',' read -r -a target_parts <<< "${raw_target}"
    for t in "${target_parts[@]}"; do
      t="$(printf '%s' "${t}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
      [[ -n "${t}" ]] || continue
      if ! is_valid_target "${t}"; then
        printf 'Error: unknown --target "%s" (run --list to see valid targets)\n' "${t}" >&2
        exit 1
      fi
      add_target_unique "${t}"
    done
  done
fi

if [[ "${#SELECTED_TARGETS[@]}" -eq 0 ]]; then
  printf 'Error: no targets selected.\n' >&2
  exit 1
fi

if [[ -z "${LOG_ROOT}" ]]; then
  LOG_ROOT="${HOME}/.cache/cross-cpu-emulation/$(date +%Y%m%d-%H%M%S)"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Dry run (Docker-only emulation harness)\n'
  printf '  Log directory    : %s\n' "${LOG_ROOT}"
  printf '  Distro platform  : %s\n' "${DISTRO_PLATFORM}"
  printf '  Duration         : %s\n' "${DURATION}"
  printf '  Register binfmt  : %s\n' "$(if [[ "${SKIP_BINFMT}" == "1" ]]; then echo "no (--skip-binfmt)"; else echo "yes (tonistiigi/binfmt)"; fi)"
  printf '  Selected targets (%s):\n' "${#SELECTED_TARGETS[@]}"
  for t in "${SELECTED_TARGETS[@]}"; do
    printf '    - %-16s %s\n' "${t}" "$(target_desc "${t}")"
  done
  exit 0
fi

locate_and_verify_docker
mkdir -p "${LOG_ROOT}"

printf 'CrossCPU Docker-Only Emulation Runner (run_id=%s)\n' "${RUN_ID}"
printf 'Logs directory  : %s\n' "${LOG_ROOT}"
printf 'Selected targets: %s\n\n' "${SELECTED_TARGETS[*]}"

ensure_binfmt

ANY_FAILED="0"
for target in "${SELECTED_TARGETS[@]}"; do
  t_log_dir="${LOG_ROOT}/${target}"
  mkdir -p "${t_log_dir}"
  printf '%s Target: %s (%s)\n' "$(c_cyan '==>')" "${target}" "$(target_desc "${target}")"

  t_start="$(date +%s)"
  t_rc=0
  case "${target}" in
    arch-amd64|arch-arm64|arch-riscv64|arch-ppc64le|arch-s390x|arch-generic)
      run_target_arch "${target}" "${t_log_dir}" || t_rc=$?
      ;;
    isa-x86_64)
      run_target_isa_x86_64 "${t_log_dir}" || t_rc=$?
      ;;
    isa-arm64)
      run_target_isa_arm64 "${t_log_dir}" || t_rc=$?
      ;;
    distro-alpine|distro-debian|distro-fedora|distro-arch)
      run_target_distro "${target}" "${t_log_dir}" || t_rc=$?
      ;;
    adversarial)
      run_target_adversarial "${t_log_dir}" || t_rc=$?
      ;;
  esac
  t_end="$(date +%s)"
  t_elapsed=$((t_end - t_start))

  RESULTS_TARGET+=("${target}")
  RESULTS_ELAPSED+=("${t_elapsed}")
  if [[ "${t_rc}" -eq 0 ]]; then
    RESULTS_STATUS+=("PASS")
    RESULTS_DETAIL+=("$(target_desc "${target}")")
    printf '<== %s: %s (%ss)\n\n' "${target}" "$(c_green PASS)" "${t_elapsed}"
  else
    RESULTS_STATUS+=("FAIL")
    RESULTS_DETAIL+=("exited $(decode_rc "${t_rc}") — see ${t_log_dir}/")
    ANY_FAILED="1"
    printf '<== %s: %s (rc=%s, %ss)\n\n' "${target}" "$(c_red FAIL)" "$(decode_rc "${t_rc}")" "${t_elapsed}"
    if [[ "${FAIL_FAST}" == "1" ]]; then
      break
    fi
  fi
done

REPORT_PATH="$(write_summary_markdown)"

printf '%s\n' "======================================================================================"
printf '%-16s %-8s %9s  %s\n' "TARGET" "STATUS" "DURATION" "DETAIL"
printf '%s\n' "--------------------------------------------------------------------------------------"
idx=0
total="${#RESULTS_TARGET[@]}"
while (( idx < total )); do
  t="${RESULTS_TARGET[${idx}]}"
  st="${RESULTS_STATUS[${idx}]}"
  el="${RESULTS_ELAPSED[${idx}]}"
  det="${RESULTS_DETAIL[${idx}]}"
  if [[ "${st}" == "PASS" ]]; then
    badge="$(c_green PASS)"
  else
    badge="$(c_red FAIL)"
  fi
  pad=$(( ${#badge} - ${#st} ))
  printf "%-16s %-$((8 + pad))s %8ss  %s\n" "${t}" "${badge}" "${el}" "${det}"
  idx=$((idx + 1))
done
printf '%s\n' "--------------------------------------------------------------------------------------"
printf 'Full logs & report: %s\n' "${REPORT_PATH}"

if [[ "${ANY_FAILED}" != "0" ]]; then
  exit 1
fi
exit 0
