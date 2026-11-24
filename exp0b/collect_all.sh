#!/bin/bash
set -e
EXP=/home/yjw/workspace/research/KSSM/exp0b
WL=$EXP/workloads
TR=$EXP/traces
DATA=/mnt/nvme12/yjw_kssm_exp0b
PY=/home/yjw/miniconda3/envs/sglang/bin/python
DUR=65

mkdir -p "$TR" "$DATA"

collect_one() {
    local name=$1
    shift
    local wdir="$DATA/$name"
    rm -rf "$wdir"
    mkdir -p "$wdir"

    echo "============================================================"
    echo "[$(date +%H:%M:%S)] Collecting workload: $name (duration=${DUR}s)"
    echo "============================================================"

    sudo "$EXP/kssm_collector" "$DATA" "$TR/${name}.bin" 2>"$TR/${name}.collector.log" &
    local cpid=$!
    for _ in $(seq 1 50); do
        grep -q READY "$TR/${name}.collector.log" 2>/dev/null && break
        sleep 0.1
    done

    # Launch workload and capture its exact PID (tgid)
    "$@" "$wdir" "$DUR" &
    local wpid=$!
    echo "$wpid" > "$TR/${name}.pid"
    wait "$wpid"

    # Wait 1s for final ringbuf flush BEFORE cleanup (so persistent files stay alive!)
    sleep 1
    sudo kill -TERM "$cpid" 2>/dev/null || true
    wait "$cpid" 2>/dev/null || true
    sudo chown yjw:yjw "$TR/${name}.bin" "$TR/${name}.collector.log"

    cat "$TR/${name}.collector.log"
    ls -lh "$TR/${name}.bin"

    # Clean up workload data AFTER collector stops so long-lived files are NOT marked unlinked
    rm -rf "$wdir"
}

collect_one rocksdb     "$WL/rocksdb_wl"
collect_one sqlite      "$PY" "$WL/sqlite_wl.py"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 collect_one pytorch "$PY" "$WL/pytorch_ckpt_wl.py"
collect_one llm         "$PY" "$WL/llm_safetensors_wl.py"
collect_one etl         "$PY" "$WL/etl_seqscan_wl.py"

echo "=== ALL 5 REAL WORKLOAD TRACES COLLECTED ==="
ls -lh "$TR"/*.bin
