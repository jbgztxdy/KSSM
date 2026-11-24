#!/bin/bash
set -e
cd /home/yjw/workspace/research/kssm_probe/exp0a
PERF=/usr/lib/linux-tools/6.8.0-117-generic/perf

sudo pkill -x kssm_exp0a_v5_runner 2>/dev/null || true

# 1. Verify 0 FPU/SSE/AVX/AMX instructions in actual x86_64 JIT machine code via bpf_prog_info!
cat << "C_EOF" > /tmp/check_jit_fpu.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
int main(void) {
    struct bpf_object *obj = bpf_object__open_file("/home/yjw/workspace/research/kssm_probe/exp0a/kssm_exp0a_v5.bpf.o", NULL);
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj)
        bpf_program__set_autoload(p, strcmp(bpf_program__name(p), "ext4_kssm_write") == 0);
    if (bpf_object__load(obj) != 0) return 1;
    p = bpf_object__find_program_by_name(obj, "ext4_kssm_write");
    struct bpf_prog_info info = {};
    uint32_t len = sizeof(info);
    uint8_t buf[16384];
    info.jited_prog_len = sizeof(buf);
    info.jited_prog_insns = (uint64_t)(uintptr_t)buf;
    if (bpf_obj_get_info_by_fd(bpf_program__fd(p), &info, &len) == 0) {
        FILE *f = fopen("/tmp/kssm_jit.bin", "wb");
        fwrite(buf, 1, info.jited_prog_len, f);
        fclose(f);
        printf("Dumped %u bytes of real x86_64 JIT machine code to /tmp/kssm_jit.bin\n", info.jited_prog_len);
    }
    bpf_object__close(obj);
    return 0;
}
C_EOF
gcc -O2 /tmp/check_jit_fpu.c -o /tmp/check_jit_fpu -lbpf -lelf -lz
sudo /tmp/check_jit_fpu
objdump -D -b binary -m i386:x86-64 /tmp/kssm_jit.bin > /tmp/kssm_jit.dis
echo "Total x86_64 JIT instructions: $(grep -cE "^ +[0-9a-f]+:" /tmp/kssm_jit.dis)"
echo "FPU/SSE/AVX/AMX instructions : $(grep -iE "xmm|ymm|zmm|tmm|vadd|vmul|mulss|addss|mulsd|addsd|fld|fst|tile" /tmp/kssm_jit.dis | wc -l)"

# 2. Prepare NVMe12 fio files
mkdir -p /mnt/nvme12/yjw_kssm_probe/fio_dir
fio --name=prep --directory=/mnt/nvme12/yjw_kssm_probe/fio_dir --rw=write --bs=1M --size=512M --numjobs=4 --direct=1 --group_reporting >/dev/null 2>&1
sync

sudo sysctl -w kernel.bpf_stats_enabled=1 >/dev/null

cleanup() {
    sudo pkill -x kssm_exp0a_v5_runner 2>/dev/null || true
    sudo sysctl -w kernel.bpf_stats_enabled=0 >/dev/null
    rm -rf /mnt/nvme12/yjw_kssm_probe/fio_dir
}
trap cleanup EXIT

run_case() {
    local wl=$1 rw=$2 direct=$3 var=$4
    local dpid=""
    if [ "$var" != "baseline" ]; then
        sudo ./kssm_exp0a_v5_runner --daemon "$var" >/tmp/kssm_v5.log 2>/tmp/kssm_v5.err &
        dpid=$!
        for _ in $(seq 1 40); do
            grep -q READY /tmp/kssm_v5.err 2>/dev/null && break
            sleep 0.1
        done
    fi

    local oj=/tmp/fio5_${wl}_${var}.json
    local pl=/tmp/perf5_${wl}_${var}.txt
    sudo $PERF stat -o "$pl" -e task-clock,cycles,instructions \
        fio --name=prep --directory=/mnt/nvme12/yjw_kssm_probe/fio_dir \
            --rw="$rw" --bs=4k --size=512M --numjobs=4 --ioengine=psync \
            --direct="$direct" --runtime=6 --time_based --ramp_time=1 \
            --group_reporting --output-format=json > "$oj" 2>/dev/null

    if [ -n "$dpid" ]; then
        sudo kill -TERM "$dpid" 2>/dev/null || true
        wait "$dpid" 2>/dev/null || true
    fi

    python3 - "$wl" "$var" "$oj" "$pl" "/tmp/kssm_v5.log" << "PY"
import json, re, sys
wl, var, jf, pf, df = sys.argv[1:6]
with open(jf) as f: d = json.load(f)
j = d["jobs"][0]
iops = j["read"]["iops"] + j["write"]["iops"]
bw_mb = (j["read"]["bw_bytes"] + j["write"]["bw_bytes"]) / (1024*1024)
tot_cpu = j["usr_cpu"] + j["sys_cpu"]
sys_cpu = j["sys_cpu"]
p_txt = open(pf).read()
m_clk = re.search(r"([\d,\.]+)\s+msec\s+task-clock", p_txt)
clk = float(m_clk.group(1).replace(",", "")) if m_clk else 0.0
bpf_info = ""
if var != "baseline":
    dtxt = open(df).read()
    m_avg = re.findall(r"prog\s+(\S+)\s+\|\s+run_cnt=\s*(\d+)\s+\|\s+run_time_ms=\s*([\d\.]+)\s+\|\s+avg_ns=([\d\.]+)", dtxt)
    m_hk = re.search(r"ssm_updates=(\d+)", dtxt)
    tot_bpf_ms = 0.0
    io_cnt = 0
    parts = []
    for n, cnt, ms, avg in m_avg:
        c = int(cnt)
        if c > 1000:
            tot_bpf_ms += float(ms)
            if "ext4_kssm" in n: io_cnt += c
            parts.append(f"{n}:{avg}ns({ms}ms)")
    bpf_cpu_pct = 100.0 * tot_bpf_ms / clk if clk > 0 else 0.0
    trig_pct = 100.0 * int(m_hk.group(1)) / io_cnt if (m_hk and io_cnt > 0) else 0.0
    bpf_info = f" | BPF_CPU={bpf_cpu_pct:4.2f}% | trig={trig_pct:4.2f}% | " + " ".join(parts)
print(f"{wl:14s} | {var:16s} | IOPS={iops:10.0f} | BW={bw_mb:7.1f}MB/s | fioCPU={tot_cpu:5.1f}% | clk={clk:7.1f}ms{bpf_info}")
PY
}

echo "=== V5 MODE-TRANSITION + 128KB BYTE-BUDGET GATING (FULL STACKID ON, numjobs=4) ==="
for cfg in "randwrite_dio randwrite 1" "randread_dio randread 1" "seqwrite_buf write 0" "seqread_buf read 0"; do
    set -- $cfg
    for var in baseline gated_subspace gated_shared; do
        run_case "$1" "$2" "$3" "$var"
    done
    echo "-------------------------------------------------------------------------------------------------------------------------------------------"
done
