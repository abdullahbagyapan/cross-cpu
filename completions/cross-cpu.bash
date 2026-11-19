# SPDX-License-Identifier: MIT
# Bash completion for cross-cpu

_cross_cpu_completions() {
    local cur prev cmd
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local commands="list info run"
    local root_opts="--help --version"
    local common_opts="--format --json -v --verbose --help"
    local run_opts="--cores --numa --isa --test --duration --warmup --format --json -v --verbose --no-banner --help"

    local core_presets="all physical p-cores e-cores"
    local isa_names="auto amx avx512 avx2 sse4 sme sve neon rvv vsx vx scalar"
    local format_names="text json"
    local workloads="all compute-int64 compute-int32 compute-int16 compute-int8 compute-vnni compute-fp64 compute-fp32 compute-fp16 compute-bf16 compute-matrix-fp32 compute-matrix-bf16 compute-matrix-int8 crypto-aes crypto-sha256 cache-l1 cache-l2 cache-l3 bandwidth-ram memory-pattern memory-sweep coherency-atomic"

    cmd=""
    local i
    for ((i = 1; i < COMP_CWORD; i++)); do
        case "${COMP_WORDS[i]}" in
            list|info|run)
                cmd="${COMP_WORDS[i]}"
                break
                ;;
        esac
    done

    if [[ -z "$cmd" ]]; then
        if [[ "$cur" == -* ]]; then
            COMPREPLY=( $(compgen -W "$root_opts" -- "$cur") )
        else
            COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
        fi
        return 0
    fi

    case "$prev" in
        --format)
            COMPREPLY=( $(compgen -W "$format_names" -- "$cur") )
            return 0
            ;;
        --cores)
            COMPREPLY=( $(compgen -W "$core_presets" -- "$cur") )
            return 0
            ;;
        --numa)
            COMPREPLY=( $(compgen -W "all 0 1 2 3" -- "$cur") )
            return 0
            ;;
        --isa)
            COMPREPLY=( $(compgen -W "$isa_names" -- "$cur") )
            return 0
            ;;
        --test)
            local prefix=""
            local tail="$cur"
            if [[ "$cur" == *,* ]]; then
                prefix="${cur%,*},"
                tail="${cur##*,}"
            fi
            local matches
            matches=( $(compgen -W "$workloads" -- "$tail") )
            COMPREPLY=()
            local m
            for m in "${matches[@]}"; do
                COMPREPLY+=( "${prefix}${m}" )
            done
            return 0
            ;;
        --duration|--warmup)
            COMPREPLY=( $(compgen -W "250ms 500ms 1s 5s 10s 30s 1m 5m 1h" -- "$cur") )
            return 0
            ;;
    esac

    case "$cmd" in
        list|info)
            COMPREPLY=( $(compgen -W "$common_opts" -- "$cur") )
            ;;
        run)
            COMPREPLY=( $(compgen -W "$run_opts" -- "$cur") )
            ;;
    esac
    return 0
}

complete -F _cross_cpu_completions cross-cpu
