#!/bin/sh
# SPDX-License-Identifier: MIT
# Non-disruptive CrossCPU source build & bounded ISA/topology verification
# for shared GCC Compile Farm (cfarm) nodes.
# Creates ./logs on the current directory level and records each attempt
# separately by machine, architecture, OS, compiler, git ref, and ISA.

set -eu

REPO_URL="${REPO_URL:-https://github.com/abdullahbagyapan/cross-cpu.git}"
REF="${REF:-main}"
OUT_DIR="${OUT_DIR:-$PWD/logs}"

sanitize_tag() {
    printf '%s' "$1" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_//; s/_$//'
}

HOST_RAW="$(hostname -s 2>/dev/null || hostname)"
OS_RAW="$(uname -s)"
ARCH_RAW="$(uname -m)"
if [ "$OS_RAW" = "AIX" ]; then
    ARCH_RAW="$(uname -p 2>/dev/null || printf '%s' "$ARCH_RAW")"
fi

HOST="$(sanitize_tag "$HOST_RAW")"
OS_NAME="$(sanitize_tag "$OS_RAW")"
ARCH_NAME="$(sanitize_tag "$ARCH_RAW")"
STAMP="$(date -u '+%Y%m%dT%H%M%SZ' 2>/dev/null || date '+%Y%m%d%H%M%S')"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ccpu-cfarm.XXXXXX" 2>/dev/null || true)"
if [ -z "$WORK_DIR" ]; then
    WORK_DIR="${TMPDIR:-/tmp}/ccpu-cfarm.$$"
    mkdir -p "$WORK_DIR"
fi

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT INT TERM HUP

# Run builds and compute checks at lowest CPU and I/O priority.
LOW_PRIO="nice -n 19"
if command -v ionice >/dev/null 2>&1; then
    if ionice -c 3 true >/dev/null 2>&1; then
        LOW_PRIO="ionice -c 3 nice -n 19"
    fi
fi

find_python() {
    for py in "${PYTHON:-}" python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
        [ -n "$py" ] || continue
        if command -v "$py" >/dev/null 2>&1; then
            if "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
                command -v "$py"
                return 0
            fi
        fi
    done
    return 1
}

PY_BIN="$(find_python)" || {
    printf 'ERROR: Python >= 3.10 not found in PATH on %s\n' "$HOST" >&2
    exit 1
}

if [ -z "${CC:-}" ]; then
    for cand in cc gcc clang; do
        if command -v "$cand" >/dev/null 2>&1; then
            CC="$cand"
            export CC
            break
        fi
    done
fi

CC_TAG="$(sanitize_tag "${CC:-unset}")"
PY_TAG="$(sanitize_tag "$(basename "$PY_BIN")")"
REF_TAG="$(sanitize_tag "$REF")"
ATTEMPT_ID="${HOST}-${ARCH_NAME}-${OS_NAME}-${CC_TAG}-${PY_TAG}-${REF_TAG}-${STAMP}-$$"

mkdir -p "$OUT_DIR"
ATTEMPT_DIR="$OUT_DIR/$ATTEMPT_ID"
mkdir -p "$ATTEMPT_DIR"

LOG_FILE="$OUT_DIR/cfarm-${ATTEMPT_ID}.log"
SUMMARY_TSV="$ATTEMPT_DIR/summary.tsv"
: >"$LOG_FILE"
printf 'attempt_id\thost\tarch\tos\tcc\tpython\tref\ttarget\texit_code\tstatus\tlog_file\n' >"$SUMMARY_TSV"

log_section() {
    printf '\n==================== %s ====================\n' "$1" >>"$LOG_FILE"
}

record_result() {
    target="$1"
    rc="$2"
    status="$3"
    step_log="$4"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\n'         "$ATTEMPT_ID" "$HOST" "$ARCH_NAME" "$OS_NAME" "$CC_TAG" "$PY_TAG" "$REF_TAG"         "$target" "$rc" "$status" "$step_log" >>"$SUMMARY_TSV"
}

# 1. Host & Toolchain Metadata
STEP_ENV="$ATTEMPT_DIR/00-host-toolchain.log"
log_section "HOST & TOOLCHAIN ($HOST)"
{
    printf 'attempt_id: %s\n' "$ATTEMPT_ID"
    printf 'timestamp_utc: %s\n' "$STAMP"
    printf 'uname: '
    uname -a
    printf 'python: %s (%s)\n' "$PY_BIN" "$("$PY_BIN" --version 2>&1)"
    printf 'cc: %s\n' "${CC:-unset}"
    if [ -n "${CC:-}" ]; then
        "$CC" --version 2>&1 | head -n 2 || "$CC" -V 2>&1 | head -n 2 || true
    fi
} >"$STEP_ENV"
cat "$STEP_ENV" >>"$LOG_FILE"

# 2. OS Topology Ground Truth
STEP_TOPO="$ATTEMPT_DIR/01-os-topology.log"
log_section "OS TOPOLOGY GROUND TRUTH"
{
    if command -v lscpu >/dev/null 2>&1; then
        lscpu
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -a 2>/dev/null | grep -E '^(hw\.|machdep\.cpu)' | head -n 80 || true
    elif command -v prtconf >/dev/null 2>&1; then
        prtconf | head -n 60 || true
    else
        printf 'No lscpu/sysctl/prtconf available\n'
    fi
} >"$STEP_TOPO"
cat "$STEP_TOPO" >>"$LOG_FILE"

# 3. Clone & Isolated Build
STEP_BUILD="$ATTEMPT_DIR/02-build.log"
log_section "CLONE & ISOLATED BUILD"
build_rc=0
{
    if [ -d "${SRC_DIR:-}" ]; then
        cp -R "$SRC_DIR" "$WORK_DIR/src"
    else
        git clone --depth 1 --branch "$REF" "$REPO_URL" "$WORK_DIR/src"
    fi

    "$PY_BIN" -m venv --system-site-packages "$WORK_DIR/venv"
    VENV_PIP="$WORK_DIR/venv/bin/pip"

    # Prefer local --no-build-isolation first so firewalled cfarm nodes do not hang on PyPI.
    if ! $LOW_PRIO "$VENV_PIP" install --no-cache-dir --no-build-isolation --no-deps "$WORK_DIR/src"; then
        $LOW_PRIO "$VENV_PIP" install --no-cache-dir "$WORK_DIR/src"
    fi
} >"$STEP_BUILD" 2>&1 || build_rc=$?
cat "$STEP_BUILD" >>"$LOG_FILE"

if [ "$build_rc" -ne 0 ]; then
    record_result "build" "$build_rc" "BUILD_FAIL" "$STEP_BUILD"
    printf 'Build failed [%s / %s / %s] (exit=%d) -> %s\n' "$HOST" "$ARCH_NAME" "$OS_NAME" "$build_rc" "$ATTEMPT_DIR" >&2
    exit "$build_rc"
fi
record_result "build" 0 "PASS" "$STEP_BUILD"

VENV_CCPU="$WORK_DIR/venv/bin/cross-cpu"
OVERALL_RC=0

# 4. Version, Info & List JSON
STEP_INFO="$ATTEMPT_DIR/03-info-list.log"
log_section "CROSS-CPU VERSION, INFO & LIST"
info_rc=0
{
    "$VENV_CCPU" --version
    printf '\n--- info --json ---\n'
    "$VENV_CCPU" info --json
    printf '\n--- list --json ---\n'
    "$VENV_CCPU" list --json
} >"$STEP_INFO" 2>&1 || info_rc=$?
cat "$STEP_INFO" >>"$LOG_FILE"
if [ "$info_rc" -eq 0 ]; then
    record_result "info-list" 0 "PASS" "$STEP_INFO"
else
    record_result "info-list" "$info_rc" "FAIL" "$STEP_INFO"
    OVERALL_RC=1
fi

# 5. Default Multi-Thread Run (250ms)
STEP_DEFAULT="$ATTEMPT_DIR/04-run-default-250ms.log"
log_section "DEFAULT MULTI-THREAD RUN (250ms)"
rc=0
$LOW_PRIO "$VENV_CCPU" run --duration 250ms -v >"$STEP_DEFAULT" 2>&1 || rc=$?
printf 'exit_code=%d\n' "$rc" >>"$STEP_DEFAULT"
cat "$STEP_DEFAULT" >>"$LOG_FILE"
if [ "$rc" -eq 0 ]; then
    record_result "default-250ms" 0 "PASS" "$STEP_DEFAULT"
else
    record_result "default-250ms" "$rc" "FAIL" "$STEP_DEFAULT"
    OVERALL_RC=1
fi

# 6. Per-ISA Bounded 100ms Dispatch Matrix (--cores 0)
if [ -z "${ISA_LIST:-}" ]; then
    case "$ARCH_NAME" in
        x86_64|amd64)
            ISA_LIST="auto scalar sse4 avx2 avx512 amx"
            ;;
        arm64|aarch64)
            ISA_LIST="auto scalar neon sve sme"
            ;;
        riscv64)
            ISA_LIST="auto scalar rvv"
            ;;
        ppc64|ppc64le|powerpc|powerpc64|powerpc64le)
            ISA_LIST="auto scalar vsx"
            ;;
        s390x)
            ISA_LIST="auto scalar vx"
            ;;
        *)
            ISA_LIST="auto scalar"
            ;;
    esac
fi

log_section "BOUNDED 100MS ISA DISPATCH MATRIX (--cores 0)"
SUMMARY_LINES=""
for isa in $ISA_LIST; do
    isa_tag="$(sanitize_tag "$isa")"
    STEP_ISA="$ATTEMPT_DIR/isa-${isa_tag}.log"
    rc=0
    $LOW_PRIO "$VENV_CCPU" run --cores 0 --isa "$isa" --duration 100ms -v >"$STEP_ISA" 2>&1 || rc=$?
    printf 'exit_code=%d\n' "$rc" >>"$STEP_ISA"

    printf '\n--- [isa=%s] ---\n' "$isa" >>"$LOG_FILE"
    cat "$STEP_ISA" >>"$LOG_FILE"

    # Exit 0 (PASS) and 2 (INIT_ERROR when host CPU lacks the forced ISA) are expected;
    # signals (>128) or verification/hardware errors (3, 4) mark overall failure.
    case "$rc" in
        0)
            status="PASS"
            ;;
        2)
            status="UNSUPPORTED_ISA"
            ;;
        *)
            status="FAIL"
            OVERALL_RC=1
            ;;
    esac
    record_result "isa:${isa}" "$rc" "$status" "$STEP_ISA"
    SUMMARY_LINES="${SUMMARY_LINES}  isa=${isa} -> exit=${rc} (${status}) [${STEP_ISA}]\n"
done

printf 'Done [%s / %s / %s / cc=%s / ref=%s]\n' "$HOST" "$ARCH_NAME" "$OS_NAME" "$CC_TAG" "$REF_TAG"
printf 'Combined log : %s\n' "$LOG_FILE"
printf 'Attempt dir  : %s\n' "$ATTEMPT_DIR"
printf 'Summary TSV  : %s\n' "$SUMMARY_TSV"
printf '%b' "$SUMMARY_LINES"
exit "$OVERALL_RC"
