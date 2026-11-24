#!/bin/bash
set -e
EXP=/home/yjw/workspace/research/KSSM/exp4_read_hardened
EXP0C=/home/yjw/workspace/research/KSSM/exp0c
DATA=/mnt/nvme12/yjw_kssm_exp0c
DB=$DATA/rocksdb_fg
EMB=$DATA/hot_seq_embed.bin
CG=/sys/fs/cgroup/kssm_exp4
PY=/home/yjw/miniconda3/envs/sglang/bin/python
DUR=35

sudo pkill -x kssm_exp4_daemon 2>/dev/null || true

# Lock dedicated CPUs 0..15 to 'performance' governor for rock-solid frequency fairness!
set_gov() {
    local g=$1
    for c in $(seq 0 15); do
        echo "$g" | sudo tee /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_governor >/dev/null 2>&1 || true
    done
}

cleanup() {
    sudo pkill -x kssm_exp4_daemon 2>/dev/null || true
    sudo sysctl -w kernel.bpf_stats_enabled=0 >/dev/null 2>&1 || true
    set_gov ondemand
    sudo rmdir "$CG" 2>/dev/null || true
}
trap cleanup EXIT

set_gov performance
echo "=== CPU 0..15 locked to governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) ==="
sudo sysctl -w kernel.bpf_stats_enabled=1 >/dev/null

evict_all() {
    python3 - "$DB" "$EMB" << 'PY'
import os, sys
paths = [sys.argv[2]]
for d in [sys.argv[1],
          "/mnt/nvme12/paul/models/DeepSeek-R1-Distill-Qwen-14B",
          "/mnt/nvme12/paul/models/Meta-Llama-3-8B-Instruct"]:
    if os.path.isdir(d):
        for fn in os.listdir(d):
            p = os.path.join(d, fn)
            if os.path.isfile(p): paths.append(p)
for p in paths:
    fd = os.open(p, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
PY
}

reset_cg() {
    sudo rmdir "$CG" 2>/dev/null || true
    sudo mkdir -p "$CG"
    echo 8589934592 | sudo tee "$CG/memory.max" >/dev/null
    echo 0 | sudo tee "$CG/memory.swap.max" >/dev/null
}

run_case() {
    local name=$1 with_bg=$2 dm_mode=$3
    echo ""
    echo "=================================================================================================="
    echo " CASE: $name (bg_llm=$with_bg, controller=$dm_mode, dur=${DUR}s, CPUs 0-15 performance, 8GiB cgroup)"
    echo "=================================================================================================="
    evict_all
    reset_cg
    rm -f /tmp/e4_ready.flag /tmp/e4_fg1_${name}.out /tmp/e4_fg2_${name}.out /tmp/e4_bg_${name}.out /tmp/e4_dm_${name}.out /tmp/e4_dm_${name}.err

    # Start FG-1 (RocksDB 5.2 GiB on CPUs 0-3) inside 8 GiB cgroup
    (
        me=$BASHPID
        sudo sh -c "echo $me > $CG/cgroup.procs"
        exec taskset -c 0-3 "$EXP0C/fg_rocksdb_bench" run "$DB" "$DUR" 4 /tmp/e4_ready.flag
    ) > /tmp/e4_fg1_${name}.out 2>&1 &
    local fg1p=$!

    for _ in $(seq 1 300); do [ -f /tmp/e4_ready.flag ] && break; sleep 0.2; done

    # Start controller right as measurement window opens
    local dmp=""
    if [ "$dm_mode" != "none" ]; then
        sudo taskset -c 12-15 "$EXP/kssm_exp4_daemon" "$DATA" "$dm_mode" "$EXP0C/kssm_weights.bin" \
            > /tmp/e4_dm_${name}.out 2> /tmp/e4_dm_${name}.err &
        dmp=$!
        for _ in $(seq 1 40); do grep -q READY /tmp/e4_dm_${name}.err 2>/dev/null && break; sleep 0.05; done
    fi

    # Start FG-2 (Hot Reusable Sequential 512 MiB Embed Scanner on CPUs 4-5) inside SAME 8 GiB cgroup!
    (
        me=$BASHPID
        sudo sh -c "echo $me > $CG/cgroup.procs"
        exec taskset -c 4-5 "$PY" "$EXP/fg2_hot_seq_embed.py" "$EMB" "$DUR"
    ) > /tmp/e4_fg2_${name}.out 2>&1 &
    local fg2p=$!

    # Start BG (43 GB .safetensors One-Shot Loader on CPUs 6-9) inside SAME 8 GiB cgroup!
    local bgp=""
    if [ "$with_bg" = "1" ]; then
        (
            me=$BASHPID
            sudo sh -c "echo $me > $CG/cgroup.procs"
            exec taskset -c 6-9 "$PY" "$EXP0C/bg_llm_ckpt.py" "/tmp/unused_ckpt" "$DUR" llm_only
        ) > /tmp/e4_bg_${name}.out 2>&1 &
        bgp=$!
    fi

    wait "$fg1p" || true
    wait "$fg2p" || true
    if [ -n "$bgp" ]; then wait "$bgp" || true; fi
    local sa=$(awk '/^(file |pgscan |pgsteal )/{printf "%s=%s ", $1, $2}' "$CG/memory.stat")
    if [ -n "$dmp" ]; then sudo kill -TERM "$dmp" 2>/dev/null || true; wait "$dmp" 2>/dev/null || true; fi

    echo "  [cgroup after]: $sa"
    grep "FG_RESULT" /tmp/e4_fg1_${name}.out
    grep "FG2_HOTSEQ" /tmp/e4_fg2_${name}.out
    if [ "$with_bg" = "1" ]; then grep "BG_RESULT" /tmp/e4_bg_${name}.out; fi
    if [ "$dm_mode" != "none" ]; then cat /tmp/e4_dm_${name}.out; fi
}

run_case solo_fg       0 none
run_case base_lru      1 none
run_case heur_size     1 size
run_case heur_seqdrop  1 seqdrop
run_case kssm_adaptive 1 kssm

echo ""
echo "======================================================================================================================="
echo " EXP 4 FINAL SUMMARY: 3-Way Co-location (FG1: RocksDB 5.2G + FG2: HotSeq 512M + BG: LLM 43G, Locked Performance Gov)"
echo "======================================================================================================================="
python3 - << 'PY'
import re
cases = ["solo_fg", "base_lru", "heur_size", "heur_seqdrop", "kssm_adaptive"]
print(f"{'Case':15s} | {'FG1 QPS':>9s} | {'FG1 P99(us)':>11s} | {'FG2 Seq(MB/s)':>13s} | {'FG2 P99(ms)':>11s} | {'BG LLM(MB/s)':>12s} | {'Audit FP (Emb / SST)'}")
print("-"*119)
data = {}
for c in cases:
    t1 = open(f"/tmp/e4_fg1_{c}.out").read()
    m1 = re.search(r"qps=([\d\.]+).*p99_us=(\d+)", t1)
    t2 = open(f"/tmp/e4_fg2_{c}.out").read()
    m2 = re.search(r"bw_mbps=([\d\.]+).*p99_ms=([\d\.]+)", t2)
    llm = 0.0
    try:
        mb = re.search(r"llm_mbps=([\d\.]+)", open(f"/tmp/e4_bg_{c}.out").read())
        if mb: llm = float(mb.group(1))
    except FileNotFoundError: pass
    aud = "-"
    try:
        at = open(f"/tmp/e4_dm_{c}.out").read().strip()
        ma = re.search(r"HotSeqEmbed_FP=(\d+\([\d\.]+G\)).*RocksDB_SST_FP=(\d+\([\d\.]+G\))", at)
        if ma: aud = f"Emb={ma.group(1)}, SST={ma.group(2)}"
    except FileNotFoundError: pass
    if m1 and m2:
        qps, p99, bw2, p99_2 = float(m1.group(1)), int(m1.group(2)), float(m2.group(1)), float(m2.group(2))
        data[c] = (qps, p99, bw2, p99_2, llm)
        print(f"{c:15s} | {qps:9.0f} | {p99:11d} | {bw2:13.1f} | {p99_2:11.1f} | {llm:12.1f} | {aud}")

if "heur_seqdrop" in data and "kssm_adaptive" in data and "base_lru" in data:
    b = data["base_lru"]; s = data["heur_seqdrop"]; k = data["kssm_adaptive"]
    print(f"\n>>> KSSM Adaptive vs Default Linux LRU:")
    print(f"    FG1 RocksDB P99 Reduction    : {100*(b[1]-k[1])/b[1]:+.1f}% ({b[1]} us -> {k[1]} us)")
    print(f"    FG2 HotSeq Scan Speedup      : {k[2]/b[2]:.2f}x ({b[2]:.1f} -> {k[2]:.1f} MB/s) | P99: {b[3]:.1f}ms -> {k[3]:.1f}ms")
    print(f"    BG  LLM Loading Throughput   : {100*k[4]/b[4]:.1f}% of baseline ({b[4]:.1f} -> {k[4]:.1f} MB/s)")
    print(f">>> KSSM Adaptive vs Strong Handcrafted Heuristic_SeqDrop (Reviewer's Toughest Read Baseline!):")
    print(f"    FG2 HotSeq Scan Speedup      : {k[2]/s[2]:.2f}x ({s[2]:.1f} -> {k[2]:.1f} MB/s) | P99: {s[3]:.1f}ms -> {k[3]:.1f}ms ({100*(s[3]-k[3])/s[3]:+.1f}%)")
    print(f"    BG  LLM Loading Throughput   : {100*k[4]/s[4]:.1f}% of SeqDrop ({s[4]:.1f} -> {k[4]:.1f} MB/s)")
PY
