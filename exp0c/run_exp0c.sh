#!/bin/bash
set -e
EXP=/home/yjw/workspace/research/KSSM/exp0c
DATA=/mnt/nvme12/yjw_kssm_exp0c
DB=$DATA/rocksdb_fg
CKPT=$DATA/ckpt_out
CG=/sys/fs/cgroup/kssm_exp0c
PY=/home/yjw/miniconda3/envs/sglang/bin/python
DUR=40

sudo pkill -x kssm_exp0c_daemon 2>/dev/null || true

cleanup() {
    sudo pkill -x kssm_exp0c_daemon 2>/dev/null || true
    sudo sysctl -w kernel.bpf_stats_enabled=0 >/dev/null 2>&1 || true
    rm -rf "$CKPT"
    # Clean cgroup if empty
    sudo rmdir "$CG" 2>/dev/null || true
}
trap cleanup EXIT

# Setup cgroup v2 with strict 8 GiB memory limit & 0 swap!
sudo rmdir "$CG" 2>/dev/null || true
sudo mkdir -p "$CG"
echo 8589934592 | sudo tee "$CG/memory.max" >/dev/null
echo 0 | sudo tee "$CG/memory.swap.max" >/dev/null
echo "=== cgroup v2 configured: memory.max=$(cat $CG/memory.max) (8 GiB), swap.max=$(cat $CG/memory.swap.max) ==="

evict_db_files() {
    python3 - "$DB" << 'PY'
import os, sys
d = sys.argv[1]
for fn in os.listdir(d):
    p = os.path.join(d, fn)
    if os.path.isfile(p):
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
PY
}

read_cg_stat() {
    awk '/^(file |pgscan |pgsteal |pgrefill )/{printf "%s=%s ", $1, $2}' "$CG/memory.stat"
}

sudo sysctl -w kernel.bpf_stats_enabled=1 >/dev/null

run_case() {
    local case_name=$1   # solo | baseline | heuristic_size | kssm
    local with_bg=$2     # 0 | 1
    local daemon_mode=$3 # none | heuristic_size | kssm

    echo ""
    echo "============================================================================================"
    echo " CASE: $case_name (with_bg=$with_bg, controller=$daemon_mode, dur=${DUR}s, cgroup=8GiB)"
    echo "============================================================================================"

    rm -rf "$CKPT" && mkdir -p "$CKPT"
    evict_db_files
    rm -f /tmp/fg_ready.flag /tmp/fg_${case_name}.out /tmp/bg_${case_name}.out /tmp/dm_${case_name}.out /tmp/dm_${case_name}.err

    # Start FG RocksDB inside 8 GiB cgroup v2!
    (
        echo $$ | sudo tee "$CG/cgroup.procs" >/dev/null
        exec "$EXP/fg_rocksdb_bench" run "$DB" "$DUR" 4 /tmp/fg_ready.flag
    ) > /tmp/fg_${case_name}.out 2>&1 &
    local fg_pid=$!

    # Wait until FG RocksDB finishes warming 5.2 GiB DB into 8 GiB cgroup page cache
    for _ in $(seq 1 300); do
        [ -f /tmp/fg_ready.flag ] && break
        sleep 0.2
    done
    local stat_before
    stat_before=$(read_cg_stat)

    # Start eBPF controller daemon (outside cgroup, system-level!) right as measurement starts
    local dm_pid=""
    if [ "$daemon_mode" != "none" ]; then
        sudo "$EXP/kssm_exp0c_daemon" "$DATA" "$daemon_mode" "$EXP/kssm_weights.bin" 0 \
            > /tmp/dm_${case_name}.out 2> /tmp/dm_${case_name}.err &
        dm_pid=$!
        for _ in $(seq 1 40); do
            grep -q READY /tmp/dm_${case_name}.err 2>/dev/null && break
            sleep 0.05
        done
    fi

    # Start BG LLM .safetensors loader + periodic Checkpoint writer inside SAME 8 GiB cgroup v2!
    local bg_pid=""
    if [ "$with_bg" = "1" ]; then
        (
            echo $$ | sudo tee "$CG/cgroup.procs" >/dev/null
            exec "$PY" "$EXP/bg_llm_ckpt.py" "$CKPT" "$DUR" mixed
        ) > /tmp/bg_${case_name}.out 2>&1 &
        bg_pid=$!
    fi

    wait "$fg_pid" || true
    if [ -n "$bg_pid" ]; then wait "$bg_pid" || true; fi
    local stat_after
    stat_after=$(read_cg_stat)

    if [ -n "$dm_pid" ]; then
        sudo kill -TERM "$dm_pid" 2>/dev/null || true
        wait "$dm_pid" 2>/dev/null || true
    fi

    echo "  [cgroup before]: $stat_before"
    echo "  [cgroup after ]: $stat_after"
    grep "FG_RESULT" /tmp/fg_${case_name}.out || cat /tmp/fg_${case_name}.out
    if [ "$with_bg" = "1" ]; then
        grep "BG_RESULT" /tmp/bg_${case_name}.out || cat /tmp/bg_${case_name}.out
    fi
    if [ "$daemon_mode" != "none" ]; then
        cat /tmp/dm_${case_name}.out
    fi
}

run_case solo           0 none
run_case baseline_lru   1 none
run_case heuristic_size 1 heuristic_size
run_case kssm_ebpf_live 1 kssm

echo ""
echo "============================================================================================"
echo " EXP 0-C SUMMARY COMPARISON"
echo "============================================================================================"
python3 - << 'PY'
import re
cases = ["solo", "baseline_lru", "heuristic_size", "kssm_ebpf_live"]
rows = {}
for c in cases:
    txt = open(f"/tmp/fg_{c}.out").read()
    m = re.search(r"qps=([\d\.]+)\s+mean_us=([\d\.]+)\s+p50_us=(\d+)\s+p90_us=(\d+)\s+p95_us=(\d+)\s+p99_us=(\d+)\s+p999_us=(\d+)", txt)
    bg_llm = bg_ckpt = 0.0
    try:
        bt = open(f"/tmp/bg_{c}.out").read()
        mb = re.search(r"llm_mbps=([\d\.]+).*ckpt_mbps=([\d\.]+)", bt)
        if mb: bg_llm, bg_ckpt = float(mb.group(1)), float(mb.group(2))
    except FileNotFoundError:
        pass
    if m:
        rows[c] = (float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(5)), int(m.group(6)), int(m.group(7)), bg_llm, bg_ckpt)

print(f"{'Case':18s} | {'FG QPS':>10s} | {'Mean(us)':>8s} | {'P50(us)':>7s} | {'P95(us)':>7s} | {'P99(us)':>8s} | {'P999(us)':>8s} | {'LLM MB/s':>9s} | {'Ckpt MB/s':>9s}")
print("-"*105)
for c in cases:
    if c not in rows: continue
    qps, mn, p50, p95, p99, p999, llm, ckpt = rows[c]
    print(f"{c:18s} | {qps:10.1f} | {mn:8.1f} | {p50:7d} | {p95:7d} | {p99:8d} | {p999:8d} | {llm:9.1f} | {ckpt:9.1f}")

if "baseline_lru" in rows and "kssm_ebpf_live" in rows:
    b_p99 = rows["baseline_lru"][4]; k_p99 = rows["kssm_ebpf_live"][4]
    b_qps = rows["baseline_lru"][0]; k_qps = rows["kssm_ebpf_live"][0]
    b_llm = rows["baseline_lru"][6]; k_llm = rows["kssm_ebpf_live"][6]
    red = 100.0 * (b_p99 - k_p99) / max(1, b_p99)
    qps_gain = 100.0 * (k_qps - b_qps) / max(1, b_qps)
    llm_ratio = 100.0 * k_llm / max(1e-6, b_llm)
    print(f"\n>>> KSSM vs Baseline Linux Default LRU:")
    print(f"    FG RocksDB P99 Latency Reduction : {red:+.1f}% ({b_p99} us -> {k_p99} us) [Target: >= 40%]")
    print(f"    FG RocksDB QPS Speedup           : {qps_gain:+.1f}% ({b_qps:.0f} -> {k_qps:.0f} QPS)")
    print(f"    BG LLM .safetensors Throughput   : {llm_ratio:.1f}% of baseline ({b_llm:.1f} -> {k_llm:.1f} MB/s)")
PY
