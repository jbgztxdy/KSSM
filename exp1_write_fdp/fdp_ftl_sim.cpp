#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <unordered_map>
#include <vector>

#pragma pack(push, 1)
struct FileRec {
    double birth_t;
    double death_t;
    uint32_t n_pages;    // 4KB pages
    uint8_t phase_id;
    uint8_t true_cls;
    uint8_t streams[8];  // 0..7 policy stream IDs (0=default, 1=short, 2=med, 3=long)
    uint8_t pad[2];
};
#pragma pack(pop)

struct SimEvent {
    double ts;
    uint32_t file_idx;
    uint8_t is_death; // 0=write(birth), 1=trim(death)
    bool operator>(const SimEvent &o) const {
        if (ts != o.ts) return ts > o.ts;
        return is_death < o.is_death; // process deaths before births at identical timestamp
    }
};

struct ReclaimUnit {
    uint32_t id;
    uint32_t written_pages;
    uint32_t valid_pages;
    uint8_t  state;      // 0=FREE, 1=OPEN, 2=SEALED
    uint8_t  ruh;        // 0..3 host RUHs, 4=GC RUH
    // Store owner file_idx of each page in this RU so GC can copy valid pages accurately!
    std::vector<int32_t> page_owner; // -1 = invalid, >=0 = file_idx (or -2 = static cold background)
};

struct PolicyResult {
    double total_waf;
    double shift_window_waf; // WAF during [0..420s] after each workload switch!
    double host_gib;
    double gc_gib;
    uint64_t erases;
    uint32_t p50_us, p95_us, p99_us, p999_us;
    std::vector<double> bin_waf_100s; // 30 bins x 100s = 3000s timeline
};

// Enterprise NVMe FDP SSD geometry (scaled proportionally to 180 GiB trace = >15x full-drive overwrites!)
// Logical capacity = 16 GiB (with 6 GiB persistent static cold data + live dynamic files ~6-9 GiB -> ~80-90% fill!)
// RU (Superblock) = 64 MiB (16,384 x 4KB pages)
static PolicyResult simulate_fdp(const std::vector<FileRec> &files, int policy_col, double op_ratio, uint32_t LOGICAL_RUS, uint32_t STATIC_RUS)
{
    const uint32_t PAGES_PER_RU = 16384; // 64 MiB per Reclaim Unit (erase superblock)
    const uint32_t TOTAL_RUS    = (uint32_t)std::round(LOGICAL_RUS * (1.0 + op_ratio));
    const uint32_t STATIC_COLD_PAGES = STATIC_RUS * PAGES_PER_RU;
    const uint32_t GC_FREE_WATERMARK = 6; // trigger greedy GC when free RUs <= 6

    std::vector<ReclaimUnit> rus(TOTAL_RUS);
    std::vector<uint32_t> free_list;
    free_list.reserve(TOTAL_RUS);
    for (uint32_t i = 0; i < TOTAL_RUS; i++) {
        rus[i].id = i;
        rus[i].written_pages = 0;
        rus[i].valid_pages = 0;
        rus[i].state = 0;
        rus[i].ruh = 0;
        rus[i].page_owner.assign(PAGES_PER_RU, -1);
        free_list.push_back(i);
    }

    // Active open RU per RUH: 0..3 = host FDP RUHs, 4 = dedicated GC RUH
    int32_t open_ru[5] = {-1, -1, -1, -1, -1};

    auto alloc_free_ru = [&](uint8_t ruh) -> int32_t {
        if (free_list.empty()) return -1;
        uint32_t id = free_list.back();
        free_list.pop_back();
        auto &r = rus[id];
        r.written_pages = 0;
        r.valid_pages = 0;
        r.state = 1;
        r.ruh = ruh;
        std::fill(r.page_owner.begin(), r.page_owner.end(), -1);
        return (int32_t)id;
    };

    // Pre-fill 10 GiB static cold data cleanly into RUH 3 (long-lived background resident data)
    // and track each dynamic file's allocated (ru_id, page_off) ranges compactly!
    // Since files are written in contiguous chunks into an open RU, store segments (ru_id, start_off, len) per file!
    struct Seg { uint32_t ru_id; uint32_t start; uint32_t len; };
    std::vector<std::vector<Seg>> file_segs(files.size());
    std::vector<uint8_t> file_alive(files.size(), 0);

    // Pre-condition drive: write 10 GiB static cold data (sealed RUs of static data)
    {
        uint32_t rem = STATIC_COLD_PAGES;
        while (rem > 0) {
            int32_t rid = alloc_free_ru(3);
            if (rid < 0) break;
            uint32_t chunk = std::min(rem, PAGES_PER_RU);
            auto &r = rus[rid];
            for (uint32_t p = 0; p < chunk; p++) r.page_owner[p] = -2; // static cold resident
            r.written_pages = chunk;
            r.valid_pages = chunk;
            r.state = (chunk == PAGES_PER_RU) ? 2 : 1;
            if (r.state == 1) open_ru[3] = rid;
            rem -= chunk;
        }
    }

    uint64_t total_host_pages = 0;
    uint64_t total_gc_pages = 0;
    uint64_t total_erases = 0;
    uint64_t sw_host_pages = 0; // post-switch [0..420s] window host pages
    uint64_t sw_gc_pages = 0;   // post-switch [0..420s] window GC pages

    const int N_BINS = 30; // 30 x 100s = 3000s
    std::vector<uint64_t> bin_host(N_BINS, 0), bin_gc(N_BINS, 0);
    std::vector<uint32_t> write_lats_us;
    write_lats_us.reserve(files.size());

    // Greedy GC: pick sealed RU with MINIMUM valid_pages; copy valid pages to GC_RUH (4)
    auto run_gc_once = [&](double cur_t) -> uint32_t {
        int32_t victim = -1;
        uint32_t min_valid = PAGES_PER_RU + 1;
        for (uint32_t i = 0; i < TOTAL_RUS; i++) {
            if (rus[i].state == 2 && rus[i].valid_pages < min_valid) {
                min_valid = rus[i].valid_pages;
                victim = (int32_t)i;
                if (min_valid == 0) break; // instant 0-copy free reclaim!
            }
        }
        if (victim < 0) return 0;

        auto &vr = rus[victim];
        uint32_t copied = 0;
        if (vr.valid_pages > 0) {
            for (uint32_t p = 0; p < vr.written_pages; p++) {
                int32_t own = vr.page_owner[p];
                if (own == -1) continue;
                // Ensure GC_RUH (4) has an open RU
                if (open_ru[4] < 0) open_ru[4] = alloc_free_ru(4);
                if (open_ru[4] < 0) break;
                auto &gr = rus[open_ru[4]];
                uint32_t dst_p = gr.written_pages++;
                gr.page_owner[dst_p] = own;
                gr.valid_pages++;
                copied++;
                // If dynamic file, append relocated 1-page segment (coalesced if contiguous)
                if (own >= 0) {
                    auto &sv = file_segs[own];
                    if (!sv.empty() && sv.back().ru_id == (uint32_t)open_ru[4] &&
                        sv.back().start + sv.back().len == dst_p) {
                        sv.back().len++;
                    } else {
                        sv.push_back({(uint32_t)open_ru[4], dst_p, 1});
                    }
                }
                if (gr.written_pages == PAGES_PER_RU) {
                    gr.state = 2; // sealed
                    open_ru[4] = -1;
                }
            }
        }
        vr.written_pages = 0;
        vr.valid_pages = 0;
        vr.state = 0;
        free_list.push_back((uint32_t)victim);

        total_gc_pages += copied;
        total_erases++;
        int b = std::min(N_BINS - 1, std::max(0, (int)(cur_t / 100.0)));
        bin_gc[b] += copied;
        double phase_off = std::fmod(cur_t, 600.0);
        if (cur_t >= 240.0 && phase_off < 420.0) sw_gc_pages += copied; // count shift windows (skip initial warm fill)

        // Enterprise 16-channel TLC NAND GC stall latency model (us):
        // Erase = 3500 us; copy-back = copied * (45 read + 25 prog) / 16 channels
        uint32_t stall_us = (uint32_t)(3500.0 + (copied * 70.0) / 16.0);
        return stall_us;
    };

    // Build chronological priority queue of births (writes) and deaths (TRIMs)
    std::priority_queue<SimEvent, std::vector<SimEvent>, std::greater<SimEvent>> pq;
    for (uint32_t i = 0; i < files.size(); i++) {
        pq.push({files[i].birth_t, i, 0});
        pq.push({files[i].death_t, i, 1});
    }

    while (!pq.empty()) {
        SimEvent ev = pq.top();
        pq.pop();
        uint32_t fi = ev.file_idx;
        const auto &fr = files[fi];

        if (ev.is_death) {
            if (!file_alive[fi]) continue;
            file_alive[fi] = 0;
            // Invalidate all physical pages belonging to this file (TRIM on unlink!)
            for (const auto &seg : file_segs[fi]) {
                auto &r = rus[seg.ru_id];
                if (r.state == 0) continue; // already erased
                for (uint32_t p = seg.start; p < seg.start + seg.len; p++) {
                    if (r.page_owner[p] == (int32_t)fi) {
                        r.page_owner[p] = -1;
                        if (r.valid_pages > 0) r.valid_pages--;
                    }
                }
            }
            file_segs[fi].clear();
            file_segs[fi].shrink_to_fit();
            continue;
        }

        // Write event: allocate n_pages into policy's chosen RUH (0..3)
        uint8_t ruh = fr.streams[policy_col] & 3;
        file_alive[fi] = 1;
        uint32_t rem = fr.n_pages;
        uint32_t extra_gc_us = 0;

        while (rem > 0) {
            while (free_list.size() <= GC_FREE_WATERMARK) {
                uint32_t st = run_gc_once(ev.ts);
                if (st == 0) break;
                extra_gc_us += st;
            }
            if (open_ru[ruh] < 0) {
                open_ru[ruh] = alloc_free_ru(ruh);
                if (open_ru[ruh] < 0) {
                    extra_gc_us += run_gc_once(ev.ts);
                    open_ru[ruh] = alloc_free_ru(ruh);
                }
            }
            if (open_ru[ruh] < 0) break; // safety guard

            auto &r = rus[open_ru[ruh]];
            uint32_t avail = PAGES_PER_RU - r.written_pages;
            uint32_t take = std::min(rem, avail);
            uint32_t start = r.written_pages;
            for (uint32_t p = start; p < start + take; p++)
                r.page_owner[p] = (int32_t)fi;
            r.written_pages += take;
            r.valid_pages += take;
            file_segs[fi].push_back({(uint32_t)open_ru[ruh], start, take});
            if (r.written_pages == PAGES_PER_RU) {
                r.state = 2; // seal RU
                open_ru[ruh] = -1;
            }
            rem -= take;
        }

        total_host_pages += fr.n_pages;
        int b = std::min(N_BINS - 1, std::max(0, (int)(ev.ts / 100.0)));
        bin_host[b] += fr.n_pages;
        double phase_off = std::fmod(ev.ts, 600.0);
        if (ev.ts >= 240.0 && phase_off < 420.0) sw_host_pages += fr.n_pages;

        // Per-file write completion latency (normalized to 128KB chunk base ~25us + GC stall amortized across write batch)
        uint32_t base_us = 25 + (fr.n_pages / 64);
        // Amortize GC stall across a small burst queue (~4 concurrent writes)
        write_lats_us.push_back(base_us + extra_gc_us / 4);
    }

    std::sort(write_lats_us.begin(), write_lats_us.end());
    auto pct = [&](double p) -> uint32_t {
        return write_lats_us[(size_t)(p * (write_lats_us.size() - 1))];
    };

    PolicyResult pr;
    pr.host_gib = total_host_pages * 4096.0 / (1024.0 * 1024.0 * 1024.0);
    pr.gc_gib   = total_gc_pages * 4096.0 / (1024.0 * 1024.0 * 1024.0);
    pr.total_waf = total_host_pages ? (double)(total_host_pages + total_gc_pages) / total_host_pages : 1.0;
    pr.shift_window_waf = sw_host_pages ? (double)(sw_host_pages + sw_gc_pages) / sw_host_pages : 1.0;
    pr.erases = total_erases;
    pr.p50_us = pct(0.50);
    pr.p95_us = pct(0.95);
    pr.p99_us = pct(0.99);
    pr.p999_us = pct(0.999);
    pr.bin_waf_100s.resize(N_BINS, 1.0);
    for (int b = 0; b < N_BINS; b++) {
        if (bin_host[b] > 0)
            pr.bin_waf_100s[b] = (double)(bin_host[b] + bin_gc[b]) / bin_host[b];
    }
    return pr;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "shift_50min_stream.bin";
    uint32_t log_rus = (argc > 2) ? (uint32_t)atoi(argv[2]) : 160;
    uint32_t sta_rus = (argc > 3) ? (uint32_t)atoi(argv[3]) : 64;
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t n = (size_t)sz / sizeof(FileRec);
    std::vector<FileRec> files(n);
    if (fread(files.data(), sizeof(FileRec), n, f) != n) { perror("fread"); return 1; }
    fclose(f);

    const char *names[8] = {
        "No-Stream (Single-RUH)",
        "Heimdall (EuroSys'25)",
        "PCStream (FAST'19)",
        "eFDP+ (arXiv'26, 420s)",
        "eFDP+ (Agile, 60s)",
        "KSSM-ZeroShot (46-58ns)",
        "KSSM-Full (+SlowLoop)",
        "Oracle (Ideal 3-RUH)"
    };

    double ops[3] = {0.07, 0.14, 0.28};
    for (int oi = 0; oi < 3; oi++) {
        double op = ops[oi];
        printf("\n====================================================================================================================\n");
        printf(" NVMe FDP (4-RUH) SSD FTL SIMULATION — 50-Min 5-App Shift Stream (%zu files, Logical=%.1f GiB, Static=%.1f GiB, OP=%.0f%%)\n", n, log_rus*64.0/1024.0, sta_rus*64.0/1024.0, op * 100.0);
        printf("====================================================================================================================\n");
        printf("%-24s | %8s | %12s | %9s | %7s | %8s | %8s | %8s | %9s\n",
               "Placement Policy", "Tot WAF", "Shift0-420s", "GC Copy", "Erases", "P50(us)", "P95(us)", "P99(us)", "P99.9(us)");
        printf("--------------------------------------------------------------------------------------------------------------------\n");

        std::vector<PolicyResult> all_res(8);
        for (int p = 0; p < 8; p++) {
            all_res[p] = simulate_fdp(files, p, op, log_rus, sta_rus);
            const auto &r = all_res[p];
            printf("%-24s | %8.3f | %12.3f | %7.2f G | %7llu | %8u | %8u | %8u | %9u\n",
                   names[p], r.total_waf, r.shift_window_waf, r.gc_gib,
                   (unsigned long long)r.erases, r.p50_us, r.p95_us, r.p99_us, r.p999_us);
        }

        if (oi == 1) {
            /* Print 5-phase per-phase breakdown & 100s timeline for standard OP=14% */
            printf("\n--- OP=14%% Per-100s Timeline WAF (Phase switches at t=600s, 1200s, 1800s, 2400s!) ---\n");
            printf("TimeBin | No-Stream | PCStream | eFDP+(420s) | eFDP+(60s) | KSSM-ZS | KSSM-Full | Oracle\n");
            for (int b = 3; b < 30; b++) { // skip initial cold fill 0-300s
                printf("%4d-%4ds | %9.3f | %8.3f | %11.3f | %10.3f | %7.3f | %9.3f | %6.3f%s\n",
                       b*100, (b+1)*100,
                       all_res[0].bin_waf_100s[b], all_res[2].bin_waf_100s[b],
                       all_res[3].bin_waf_100s[b], all_res[4].bin_waf_100s[b],
                       all_res[5].bin_waf_100s[b], all_res[6].bin_waf_100s[b],
                       all_res[7].bin_waf_100s[b],
                       (b % 6 == 0) ? "  <-- WORKLOAD SHIFT!" : "");
            }
        }
    }
    return 0;
}
