#!/bin/bash
set -e
EXP=/home/yjw/workspace/research/KSSM/exp2_scale_distill
WL=$EXP/workloads
TR=$EXP/traces
COL=/home/yjw/workspace/research/KSSM/exp0b/kssm_collector
DATA=/mnt/nvme12/yjw_kssm_exp2
PY=/home/yjw/miniconda3/envs/sglang/bin/python
DUR=65

collect_one() {
    local name=$1; shift
    local wdir="$DATA/$name"
    rm -rf "$wdir" && mkdir -p "$wdir"
    echo "[$(date +%H:%M:%S)] Collecting NEW app: $name (${DUR}s)..."
    sudo "$COL" "$DATA" "$TR/${name}.bin" 2>"$TR/${name}.collector.log" &
    local cpid=$!
    for _ in $(seq 1 50); do grep -q READY "$TR/${name}.collector.log" 2>/dev/null && break; sleep 0.1; done

    "$@" "$wdir" "$DUR" &
    local wpid=$!
    echo "$wpid" > "$TR/${name}.pid"
    wait "$wpid"

    sleep 1
    sudo kill -TERM "$cpid" 2>/dev/null || true
    wait "$cpid" 2>/dev/null || true
    sudo chown yjw:yjw "$TR/${name}.bin" "$TR/${name}.collector.log"
    cat "$TR/${name}.collector.log"
    ls -lh "$TR/${name}.bin"
    rm -rf "$wdir"
}

collect_one compiler    "$PY" "$WL/compiler_build_wl.py"
collect_one git_vcs     "$PY" "$WL/git_vcs_wl.py"
collect_one log_segment "$WL/log_segment_wl"
collect_one zstd_arch   "$PY" "$WL/zstd_archive_wl.py"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 collect_one vllm_kv "$PY" "$WL/vllm_kv_swap_wl.py"
echo "=== ALL 5 NEW WORKLOADS COLLECTED (10-APP REAL CORPUS READY!) ==="
