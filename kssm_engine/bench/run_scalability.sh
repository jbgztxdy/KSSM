#!/bin/bash
set -e
ENG=/home/yjw/workspace/research/KSSM/kssm_engine
SDIR=/mnt/nvme12/yjw_kssm_exp0c/scale_tmp
PERF=/usr/lib/linux-tools/6.8.0-117-generic/perf

sudo pkill -x kssm_daemon 2>/dev/null || true
sudo sysctl -w kernel.bpf_stats_enabled=1 >/dev/null

cleanup() {
    sudo pkill -x kssm_daemon 2>/dev/null || true
    sudo sysctl -w kernel.bpf_stats_enabled=0 >/dev/null 2>&1 || true
    rm -f "$SDIR"/indep.* "$SDIR"/shared_hot.dat "$SDIR"/dio.*
}
trap cleanup EXIT

echo "=== PREPARING TEST FILES IN $SDIR ==="
rm -f "$SDIR"/indep.* "$SDIR"/shared_hot.dat "$SDIR"/dio.*
# Prepare 1 shared 64MB file & 64 independent 16MB files (1 GiB total -> fits easily in RAM Page Cache!)
fio --name=shared --filename="$SDIR/shared_hot.dat" --rw=write --bs=1M --size=64M --direct=1 --group_reporting >/dev/null 2>&1
fio --name=indep  --directory="$SDIR" --rw=write --bs=1M --size=16M --numjobs=64 --direct=1 --group_reporting >/dev/null 2>&1
# Warm them into page cache once
fio --name=shared --filename="$SDIR/shared_hot.dat" --rw=read --bs=1M --size=64M --direct=0 --group_reporting >/dev/null 2>&1
fio --name=indep  --directory="$SDIR" --rw=read --bs=1M --size=16M --numjobs=64 --direct=0 --group_reporting >/dev/null 2>&1
echo "Prep done."

run_point() {
    local scen=$1 nj=$2 var=$3 rw=$4 direct=$5 farg=$6
    local dpid=""
    if [ "$var" != "baseline" ]; then
        sudo "$ENG/bin/kssm_daemon" "$SDIR" "$var" 0 >/tmp/sc_dm.out 2>/tmp/sc_dm.err &
        dpid=$!
        for _ in $(seq 1 40); do grep -q READY /tmp/sc_dm.err 2>/dev/null && break; sleep 0.05; done
    fi

    local oj=/tmp/sc_fio.json
    local pl=/tmp/sc_perf.txt
    sudo $PERF stat -o "$pl" -e task-clock \
        fio --name=w $farg --rw="$rw" --bs=4k --numjobs="$nj" --ioengine=psync \
            --direct="$direct" --runtime=4 --time_based --ramp_time=1 \
            --group_reporting --output-format=json > "$oj" 2>/dev/null

    if [ -n "$dpid" ]; then
        sudo kill -TERM "$dpid" 2>/dev/null || true
        wait "$dpid" 2>/dev/null || true
    fi

    python3 - "$scen" "$nj" "$var" "$oj" "$pl" "/tmp/sc_dm.out" << 'PY'
import json, re, sys
scen, nj, var, jf, pf, df = sys.argv[1:7]
d = json.load(open(jf)); j = d["jobs"][0]
iops = j["read"]["iops"] + j["write"]["iops"]
miops = iops / 1e6
bw_gb = (j["read"]["bw_bytes"] + j["write"]["bw_bytes"]) / (1024**3)
p_txt = open(pf).read()
m_clk = re.search(r"([\d,\.]+)\s+msec\s+task-clock", p_txt)
clk = float(m_clk.group(1).replace(",", "")) if m_clk else 1.0
extra = ""
if var != "baseline":
    dt = open(df).read()
    m_rw = re.findall(r"prog\s+kssm_ext4_(read|write)\s+\|\s+run_cnt=\s*(\d+)\s+\|\s+run_time_ms=\s*([\d\.]+)\s+\|\s+avg_ns=([\d\.]+)", dt)
    tot_ms = sum(float(ms) for _, cnt, ms, _ in m_rw if int(cnt) > 1000)
    avg_s = " ".join(f"{op}:{avg}ns" for op, cnt, _, avg in m_rw if int(cnt) > 1000)
    m_tr = re.search(r"ssm_upd=\d+\s+\(([\d\.]+)%\)", dt)
    tr_s = m_tr.group(1) + "%" if m_tr else "-"
    bpf_pct = 100.0 * tot_ms / clk
    extra = f" | BPF_CPU={bpf_pct:4.2f}% | trig={tr_s:5s} | {avg_s}"
print(f"{scen:14s} | cores={int(nj):2d} | {var:13s} | {miops:6.2f} MIOPS ({bw_gb:5.2f} GB/s){extra}")
PY
}

echo ""
echo "============================================================================================================"
echo " SCENARIO A: Independent-File 4KB PageCache Read Scaling (1 -> 4 -> 8 -> 16 -> 32 -> 64 Cores!)"
echo "============================================================================================================"
for nj in 1 4 8 16 32 64; do
    for var in baseline tl_percpu common_lru; do
        run_point "Indep_Read" "$nj" "$var" read 0 "--directory=$SDIR --filename_format=indep.\$jobnum.0 --size=16M"
    done
    echo "------------------------------------------------------------------------------------------------------------"
done

echo ""
echo "============================================================================================================"
echo " SCENARIO B: Extreme Single Shared Hot-File Contention (1 -> 4 -> 16 -> 32 -> 64 Cores Pounding 1 File!)"
echo "============================================================================================================"
for nj in 1 4 16 32 64; do
    for var in baseline tl_percpu shared_inode common_lru; do
        run_point "Shared_1File" "$nj" "$var" randread 0 "--filename=$SDIR/shared_hot.dat --size=64M"
    done
    echo "------------------------------------------------------------------------------------------------------------"
done

echo ""
echo "============================================================================================================"
echo " SCENARIO C: Real NVMe PM174X Direct 4KB Random Write Scaling (1 -> 4 -> 16 -> 32 Cores!)"
echo "============================================================================================================"
for nj in 1 4 16 32; do
    for var in baseline tl_percpu; do
        run_point "NVMe_RandWr" "$nj" "$var" randwrite 1 "--directory=$SDIR --filename_format=indep.\$jobnum.0 --size=16M"
    done
    echo "------------------------------------------------------------------------------------------------------------"
done
