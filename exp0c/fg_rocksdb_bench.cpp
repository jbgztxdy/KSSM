#include <rocksdb/filter_policy.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/table.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

using namespace rocksdb;

// Fast Zipfian-like hot/warm generator (80% requests hit hot 25% keys, 20% hit full 5.5GB DB)
struct SkewGen {
    std::mt19937_64 rng;
    uint64_t n_keys;
    uint64_t hot_keys;
    SkewGen(uint64_t seed, uint64_t n) : rng(seed), n_keys(n), hot_keys(n / 4) {}
    uint64_t next() {
        if ((rng() % 100) < 80) return rng() % hot_keys;
        return rng() % n_keys;
    }
};

static void advise_dir_dontneed(const std::string &dir) {
    DIR *d = opendir(dir.c_str());
    if (!d) return;
    struct dirent *ent;
    while ((ent = readdir(d)) != nullptr) {
        if (ent->d_name[0] == '.') continue;
        std::string p = dir + "/" + ent->d_name;
        int fd = open(p.c_str(), O_RDONLY);
        if (fd >= 0) {
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            close(fd);
        }
    }
    closedir(d);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <build|run> <db_path> [duration_s] [n_threads] [ready_file]\n", argv[0]);
        return 1;
    }
    std::string mode = argv[1];
    std::string db_path = argv[2];
    int dur_s = (argc >= 4) ? atoi(argv[3]) : 45;
    int n_threads = (argc >= 5) ? atoi(argv[4]) : 4;
    std::string ready_file = (argc >= 6) ? argv[5] : "";

    const uint64_t NUM_KEYS = 1350000; // ~5.5 GiB uncompressed (4 KB values)

    Options opts;
    opts.create_if_missing = (mode == "build");
    opts.compression = kNoCompression;
    opts.write_buffer_size = 64 * 1024 * 1024;
    opts.target_file_size_base = 64 * 1024 * 1024; // 64 MB SSTs (realistic production size!)
    opts.max_bytes_for_level_base = 256 * 1024 * 1024;
    opts.max_background_jobs = 8;
    opts.use_direct_reads = false;

    BlockBasedTableOptions topts;
    topts.no_block_cache = true; // rely purely on Linux OS Page Cache
    topts.filter_policy.reset(NewBloomFilterPolicy(10, false));
    opts.table_factory.reset(NewBlockBasedTableFactory(topts));

    if (mode == "build") {
        DB *db = nullptr;
        Status s = DB::Open(opts, db_path, &db);
        if (!s.ok()) { std::cerr << "Open build failed: " << s.ToString() << "\n"; return 1; }
        std::string val(4000, 'R');
        WriteOptions wopts;
        wopts.disableWAL = true;
        auto t0 = std::chrono::steady_clock::now();
        for (uint64_t i = 0; i < NUM_KEYS; i++) {
            char kbuf[32];
            snprintf(kbuf, sizeof(kbuf), "k_%010llu", (unsigned long long)i);
            val[0] = (char)(i & 0xFF);
            val[100] = (char)((i >> 8) & 0xFF);
            val[2000] = (char)((i >> 16) & 0xFF);
            db->Put(wopts, kbuf, val);
            if ((i % 250000) == 0 && i > 0) {
                printf("  built %llu / %llu keys...\n", (unsigned long long)i, (unsigned long long)NUM_KEYS);
                fflush(stdout);
            }
        }
        db->Flush(FlushOptions());
        db->CompactRange(CompactRangeOptions(), nullptr, nullptr);
        delete db;
        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("RocksDB 5.5GiB DB built & compacted in %.1f s\n", dt);
        return 0;
    }

    // Run mode: open read-only, warm up page cache inside cgroup, then run latency-sensitive point lookups!
    DB *db = nullptr;
    Status s = DB::OpenForReadOnly(opts, db_path, &db);
    if (!s.ok()) { std::cerr << "OpenForReadOnly failed: " << s.ToString() << "\n"; return 1; }

    // Warm up the entire 5.5 GiB DB into the 8 GiB cgroup page cache before starting measurement!
    printf("[FG RocksDB] Warming up 5.5 GiB working set into 8 GiB cgroup page cache...\n");
    fflush(stdout);
    {
        auto tw0 = std::chrono::steady_clock::now();
        std::vector<std::thread> wts;
        for (int t = 0; t < 4; t++) {
            wts.emplace_back([&, t]() {
                ReadOptions ro;
                std::string out;
                // Touch every 16th key sequentially to bring all SST blocks into cgroup page cache
                for (uint64_t i = t; i < NUM_KEYS; i += 4) {
                    char kbuf[32];
                    snprintf(kbuf, sizeof(kbuf), "k_%010llu", (unsigned long long)i);
                    db->Get(ro, kbuf, &out);
                }
            });
        }
        for (auto &th : wts) th.join();
        double wdt = std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count();
        printf("[FG RocksDB] Warmup done in %.1f s.\n", wdt);
        fflush(stdout);
    }

    if (!ready_file.empty()) {
        FILE *rf = fopen(ready_file.c_str(), "w");
        if (rf) { fprintf(rf, "READY\n"); fclose(rf); }
    }

    // Run steady-state point lookups and collect exact per-op latencies (microseconds)
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> total_ops{0};
    std::vector<std::vector<uint32_t>> thread_lats(n_threads);

    auto t_start = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (int t = 0; t < n_threads; t++) {
        thread_lats[t].reserve(1500000);
        workers.emplace_back([&, t]() {
            SkewGen gen(1000 + t * 97, NUM_KEYS);
            ReadOptions ro;
            std::string out;
            uint64_t local_ops = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                uint64_t k = gen.next();
                char kbuf[32];
                snprintf(kbuf, sizeof(kbuf), "k_%010llu", (unsigned long long)k);
                auto op0 = std::chrono::steady_clock::now();
                Status st = db->Get(ro, kbuf, &out);
                auto op1 = std::chrono::steady_clock::now();
                if (st.ok()) {
                    uint32_t us = (uint32_t)std::chrono::duration_cast<std::chrono::microseconds>(op1 - op0).count();
                    thread_lats[t].push_back(us);
                    local_ops++;
                }
            }
            total_ops.fetch_add(local_ops, std::memory_order_relaxed);
        });
    }

    // Print per-5s progress
    for (int sec = 0; sec < dur_s; sec += 5) {
        int slp = std::min(5, dur_s - sec);
        std::this_thread::sleep_for(std::chrono::seconds(slp));
    }
    stop.store(true, std::memory_order_relaxed);
    for (auto &th : workers) th.join();
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    std::vector<uint32_t> all_lats;
    size_t tot = 0;
    for (auto &v : thread_lats) tot += v.size();
    all_lats.reserve(tot);
    for (auto &v : thread_lats) all_lats.insert(all_lats.end(), v.begin(), v.end());
    std::sort(all_lats.begin(), all_lats.end());

    auto pct = [&](double p) -> uint32_t {
        if (all_lats.empty()) return 0;
        size_t idx = (size_t)(p * (all_lats.size() - 1));
        return all_lats[idx];
    };
    double mean_us = 0.0;
    for (uint32_t x : all_lats) mean_us += x;
    mean_us /= std::max<size_t>(1, all_lats.size());

    printf("FG_RESULT ops=%llu elapsed=%.2f qps=%.1f mean_us=%.1f p50_us=%u p90_us=%u p95_us=%u p99_us=%u p999_us=%u\n",
           (unsigned long long)all_lats.size(), elapsed, all_lats.size() / elapsed,
           mean_us, pct(0.50), pct(0.90), pct(0.95), pct(0.99), pct(0.999));
    fflush(stdout);

    delete db;
    return 0;
}
