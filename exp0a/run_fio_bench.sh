#!/bin/bash
set -e
cd /home/yjw/workspace/research/kssm_probe/exp0a
BPFTOOL=/usr/lib/linux-tools/6.8.0-117-generic/bpftool
PERF=/usr/lib/linux-tools/6.8.0-117-generic/perf

sudo pkill -x kssm_exp0a_v4_runner 2>/dev/null || true

echo "=== PREPARING FIO TEST FILES ON /mnt/nvme12/yjw_kssm_probe ==="
mkdir -p /mnt/nvme12/yjw_kssm_probe/fio_dir
fio --name=prep --directory=/mnt/nvme12/yjw_kssm_probe/fio_dir --rw=write --bs=1M --size=512M --numjobs=4 --direct=1 --group_reporting >/dev/null 2>&1
sync

sudo sysctl -w kernel.bpf_stats_enabled=1 >/dev/null

cleanup() {
    sudo pkill -x kssm_exp0a_v4_runner 2>/dev/null || true
    sudo sysctl -w kernel.bpf_stats_enabled=0 >/dev/null
    rm -rf /mnt/nvme12/yjw_kssm_probe/fio_dir
}
trap cleanup EXIT

run_fio_case() {
    local wl_name=$1
    local rw=$2
    local direct=$3
    local variant=$4

    local dpid=""
    if [ "$variant" != "baseline" ]; then
        sudo ./kssm_exp0a_v4_runner --daemon "$variant" >/tmp/kssm_daemon.log 2>/tmp/kssm_daemon.err &
        dpid=$!
        for _ in $(seq 1 40); do
            grep -q READY /tmp/kssm_daemon.err 2>/dev/null && break
            sleep 0.1
        done
    fi

    local out_json=/tmp/fio_${wl_name}_${variant}.json
    local perf_log=/tmp/perf_${wl_name}_${variant}.txt
    sudo $PERF stat -o "$perf_log" -e task-clock,cycles,instructions \
        fio --name=prep --directory=/mnt/nvme12/yjw_kssm_probe/fio_dir \
            --rw="$rw" --bs=4k --size=512M --numjobs=4 --ioengine=psync \
            --direct="$direct" --runtime=6 --time_based --ramp_time=1 \
            --group_reporting --output-format=json > "$out_json" 2>/dev/null

    if [ -n "$dpid" ]; then
        sudo kill -TERM "$dpid" 2>/dev/null || true
        wait "$dpid" 2>/dev/null || true
    fi

    python3 - "$wl_name" "$variant" "$out_json" "$perf_log" "/tmp/kssm_daemon.log" << "PY"
import json, re, sys
wl, var, jf, pf, df = sys.argv[1:6]
with open(jf) as f:
    d = json.load(f)
j = d["jobs"][0]
iops = j["read"]["iops"] + j["write"]["iops"]
bw_mb = (j["read"]["bw_bytes"] + j["write"]["bw_bytes"]) / (1024*1024)
usr = j["usr_cpu"]
sys_cpu = j["sys_cpu"]
tot_cpu = usr + sys_cpu
p_txt = open(pf).read()
m_cyc = re.search(r"([\d,]+)\s+cycles", p_txt)
m_clk = re.search(r"([\d,\.]+)\s+msec\s+task-clock", p_txt)
cyc = int(m_cyc.group(1).replace(",", "")) if m_cyc else 0
clk = float(m_clk.group(1).replace(",", "")) if m_clk else 0.0
cyc_per_io = cyc / (iops * 7.0) if iops > 0 else 0  # runtime 6s + ramp 1s = 7s total perf window
bpf_info = ""
if var != "baseline":
    dtxt = open(df).read()
    m_avg = re.findall(r"prog\s+(\S+)\s+\|\s+run_cnt=\s*(\d+)\s+\|\s+run_time_ms=\s*([\d\.]+)\s+\|\s+avg_ns=([\d\.]+)", dtxt)
    m_hk = re.search(r"hooks=(\d+)\s+\|\s+ssm_updates=(\d+)\s+\(([\d\.]+)%\)", dtxt)
    parts = [f"{n}:{avg}ns({ms}ms)" for n, cnt, ms, avg in m_avg if int(cnt) > 1000]
    trig = f"trig={m_hk.group(3)}%" if m_hk else ""
    bpf_info = " | " + " ".join(parts) + (" " + trig if trig else "")
print(f"{wl:14s} | {var:20s} | IOPS={iops:10.0f} | BW={bw_mb:7.1f}MB/s | CPU={tot_cpu:5.1f}% (s={sys_cpu:5.1f}%) | clk={clk:7.1f}ms | cyc/IO={cyc_per_io:6.0f}{bpf_info}")
PY
}

echo "=== RUNNING FIO 4KB HIGH-CONCURRENCY BENCHMARKS (numjobs=4, /mnt/nvme12) ==="
for cfg in "randwrite_dio randwrite 1" "randread_dio randread 1" "seqwrite_buf write 0" "seqread_buf read 0"; do
    set -- $cfg
    for var in baseline empty gated_subspace gated_subspace_nostk gated_shared; do
        run_fio_case "$1" "$2" "$3" "$var"
    done
    echo "------------------------------------------------------------------------------------------------------------------------------------------"
done
