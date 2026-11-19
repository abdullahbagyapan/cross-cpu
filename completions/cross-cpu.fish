# SPDX-License-Identifier: MIT
# Fish completion for cross-cpu

set -l __cross_cpu_cmds list info run
set -l __cross_cpu_isas auto amx avx512 avx2 sse4 sme sve neon rvv vsx vx scalar
set -l __cross_cpu_cores all physical p-cores e-cores
set -l __cross_cpu_workloads \
    all \
    compute-int64 compute-int32 compute-int16 compute-int8 compute-vnni \
    compute-fp64 compute-fp32 compute-fp16 compute-bf16 \
    compute-matrix-fp32 compute-matrix-bf16 compute-matrix-int8 \
    crypto-aes crypto-sha256 \
    cache-l1 cache-l2 cache-l3 \
    bandwidth-ram memory-pattern memory-sweep coherency-atomic

complete -c cross-cpu -f

# Top-level flags and subcommands
complete -c cross-cpu -n "not __fish_seen_subcommand_from $__cross_cpu_cmds" -s h -l help -d "Show help message"
complete -c cross-cpu -n "not __fish_seen_subcommand_from $__cross_cpu_cmds" -l version -d "Show CrossCPU version"
complete -c cross-cpu -n "not __fish_seen_subcommand_from $__cross_cpu_cmds" -a list -d "List canonical workloads and ISA readiness"
complete -c cross-cpu -n "not __fish_seen_subcommand_from $__cross_cpu_cmds" -a info -d "Show host topology and ISA probe details"
complete -c cross-cpu -n "not __fish_seen_subcommand_from $__cross_cpu_cmds" -a run -d "Run CPU stress and SDC validation workloads"

# Common subcommand flags
complete -c cross-cpu -n "__fish_seen_subcommand_from $__cross_cpu_cmds" -l format -x -a "text json" -d "Output format"
complete -c cross-cpu -n "__fish_seen_subcommand_from $__cross_cpu_cmds" -l json -d "Emit JSON output"
complete -c cross-cpu -n "__fish_seen_subcommand_from $__cross_cpu_cmds" -s v -l verbose -d "Emit verbose diagnostics to stderr"
complete -c cross-cpu -n "__fish_seen_subcommand_from $__cross_cpu_cmds" -s h -l help -d "Show command help"

# run-specific flags
complete -c cross-cpu -n "__fish_seen_subcommand_from run" -l cores -x -a "$__cross_cpu_cores" -d "Core preset or CPU list"
complete -c cross-cpu -n "__fish_seen_subcommand_from run" -l numa -x -a "all 0 1 2 3" -d "NUMA node filter"
complete -c cross-cpu -n "__fish_seen_subcommand_from run" -l isa -x -a "$__cross_cpu_isas" -d "ISA backend"
complete -c cross-cpu -n "__fish_seen_subcommand_from run" -l test -x -a "$__cross_cpu_workloads" -d "Workload name or all"
complete -c cross-cpu -n "__fish_seen_subcommand_from run" -l duration -x -a "250ms 500ms 1s 5s 10s 30s 1m 5m 1h" -d "Timed duration per workload"
complete -c cross-cpu -n "__fish_seen_subcommand_from run" -l warmup -x -a "0s 500ms 5s 10s" -d "Optional warmup duration"
complete -c cross-cpu -n "__fish_seen_subcommand_from run" -l no-banner -d "Suppress ASCII banner"
