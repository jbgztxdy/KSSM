#!/bin/bash
set -e
EXP=/home/yjw/workspace/research/KSSM/exp0c
DATA=/mnt/nvme12/yjw_kssm_exp0c
DB=$DATA/rocksdb_fg
CKPT=$DATA/ckpt_out
CG=/sys/fs/cgroup/kssm_exp0c
PY=/home/yjw/miniconda3/envs/sglang/bin/python
DUR=35

sudo pkill -x kssm_exp0c_daemon 2>/dev/null || true

cleanup() {
    sudo pkill -x kssm_exp0c_daemon 2>/dev/null || true
    sudo sysctl -w kernel.bpf_stats_enabled=0 >/dev/null 2>&1 || true
    rm -rf "$CKPT"
    sudo rmdir "$CG" 2>/dev/null || true
}
trap cleanup EXIT

evict_all_target_files() {
    python3 - "$DB" << 'PY'
import os, sys
dirs = [
    sys.argv[1],
    "/mnt/nvme12/paul/models/DeepSeek-R1-Distill-Qwen-14B",
    "/mnt/nvme12/paul/models/Meta-Llama-3-8B-Instruct",
]
for d in dirs:
    if not os.path.isdir(d): continue
    for fn in os.listdir(d):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
PY
}

reset_cgroup() {
    sudo rmdir "$CG" 2>/dev/null || true
    sudo mkdir -p "$CG"
    echo 8589934592 | sudo tee "$CG/memory.max" >/dev/null
    echo 0 | sudo tee "$CG/memory.swap.max" >/dev/null
}

sudo sysctl -w kernel.bpf_stats_enabled=1 >/dev/null

run_case() {
    local name=$1 bg_mode=$2 dm_mode=$3
    echo ""
    echo "============================================================================================"
    echo " CASE: $name (bg=$bg_mode, controller=$dm_mode, dur=${DUR}s, fresh 8 GiB cgroup v2)"
    echo "============================================================================================"
    rm -rf "$CKPT" && mkdir -p "$CKPT"
    evict_all_target_files
    reset_cgroup
    rm -f /tmp/fgc_ready.flag /tmp/fgc_${name}.out /tmp/bgc_${name}.out /tmp/dmc_${name}.out /tmp/dmc_${name}.err

    (
        me=$BASHPID
        sudo sh -c "echo $me > $CG/cgroup.procs"
        exec "$EXP/fg_rocksdb_bench" run "$DB" "$DUR" 4 /tmp/fgc_ready.flag
    ) > /tmp/fgc_${name}.out 2>&1 &
    local fgp=$!

    for _ in $(seq 1 300); do [ -f /tmp/fgc_ready.flag ] && break; sleep 0.2; done
    local sb=$(awk '/^(file |pgscan |pgsteal )/{printf "%s=%s ", $1, $2}' "$CG/memory.stat")

    local dmp=""
    if [ "$dm_mode" != "none" ]; then
        sudo "$EXP/kssm_exp0c_daemon" "$DATA" "$dm_mode" "$EXP/kssm_weights.bin" 0 \
            > /tmp/dmc_${name}.out 2> /tmp/dmc_${name}.err &
        dmp=$!
        for _ in $(seq 1 40); do grep -q READY /tmp/dmc_${name}.err 2>/dev/null && break; sleep 0.05; done
    fi

    local bgp=""
    if [ "$bg_mode" != "none" ]; then
        (
            me=$BASHPID
            sudo sh -c "echo $me > $CG/cgroup.procs"
            exec "$PY" "$EXP/bg_llm_ckpt.py" "$CKPT" "$DUR" "$bg_mode"
        ) > /tmp/bgc_${name}.out 2>&1 &
        bgp=$!
    fi

    wait "$fgp" || true
    if [ -n "$bgp" ]; then wait "$bgp" || true; fi
    local sa=$(awk '/^(file |pgscan |pgsteal )/{printf "%s=%s ", $1, $2}' "$CG/memory.stat")

    if [ -n "$dmp" ]; then
        sudo kill -TERM "$dmp" 2>/dev/null || true
        wait "$dmp" 2>/dev/null || true
    fi

    echo "  [cgroup before]: $sb"
    echo "  [cgroup after ]: $sa"
    grep "FG_RESULT" /tmp/fgc_${name}.out
    if [ "$bg_mode" != "none" ]; then grep "BG_RESULT" /tmp/bgc_${name}.out; fi
    if [ "$dm_mode" != "none" ]; then cat /tmp/dmc_${name}.out; fi
}

run_case solo           none     none
run_case base_llm_only  llm_only none
run_case kssm_llm_only  llm_only kssm
run_case base_mixed     mixed    none
run_case heur_mixed     mixed    heuristic_size
run_case kssm_mixed     mixed    kssm

echo ""
echo "=========================================================================================================="
echo " EXP 0-C FINAL CLEAN SUMMARY (Fresh 8 GiB cgroup v2 per case, 100% identical cold start)"
echo "=========================================================================================================="
python3 - << 'PY'
import re
cases = ["solo", "base_llm_only", "kssm_llm_only", "base_mixed", "heur_mixed", "kssm_mixed"]
rows = {}
for c in cases:
    txt = open(f"/tmp/fgc_{c}.out").read()
    m = re.search(r"qps=([\d\.]+)\s+mean_us=([\d\.]+)\s+p50_us=(\d+)\s+p90_us=(\d+)\s+p95_us=(\d+)\s+p99_us=(\d+)\s+p999_us=(\d+)", txt)
    llm = ckpt = 0.0
    try:
        bt = open(f"/tmp/bgc_{c}.out").read()
        mb = re.search(r"llm_mbps=([\d\.]+).*ckpt_mbps=([\d\.]+)", bt)
        if mb: llm, ckpt = float(mb.group(1)), float(mb.group(2))
    except FileNotFoundError:
        pass
    if m:
        rows[c] = (float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(5)), int(m.group(6)), int(m.group(7)), llm, ckpt)

print(f"{'Case':16s} | {'FG QPS':>10s} | {'Mean(us)':>8s} | {'P50(us)':>7s} | {'P95(us)':>7s} | {'P99(us)':>8s} | {'P999(us)':>8s} | {'LLM MB/s':>9s} | {'Ckpt MB/s':>9s}")
print("-"*106)
for c in cases:
    if c not in rows: continue
    qps, mn, p50, p95, p99, p999, llm, ckpt = rows[c]
    print(f"{c:16s} | {qps:10.1f} | {mn:8.1f} | {p50:7d} | {p95:7d} | {p99:8d} | {p999:8d} | {llm:9.1f} | {ckpt:9.1f}")

for bname, kname, title in [
    ("base_llm_only", "kssm_llm_only", "Pure LLM .safetensors Co-location (100% Neural Readout)"),
    ("base_mixed",    "kssm_mixed",    "Full Mixed Co-location (LLM .safetensors + Periodic Checkpoint)"),
]:
    if bname in rows and kname in rows:
        bp99, kp99 = rows[bname][4], rows[kname][4]
        bqps, kqps = rows[bname][0], rows[kname][0]
        bllm, kllm = rows[bname][6], rows[kname][6]
        print(f"\n>>> {title}:")
        print(f"    FG RocksDB P99 Latency Reduction : {100.0*(bp99-kp99)/max(1,bp99):+.1f}% ({bp99} us -> {kp99} us) [Target >= 40%]")
        print(f"    FG RocksDB QPS Speedup           : {100.0*(kqps-bqps)/max(1,bqps):+.1f}% ({bqps:.0f} -> {kqps:.0f} QPS)")
        print(f"    BG LLM .safetensors Throughput   : {100.0*kllm/max(1e-6,bllm):.1f}% of baseline ({bllm:.1f} -> {kllm:.1f} MB/s)")
PY
