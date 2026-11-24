#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/table.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace rocksdb;
static std::atomic<bool> g_stop{false};

int main(int argc, char **argv) {
    std::string db_path = (argc > 1) ? argv[1] : "/mnt/nvme12/yjw_kssm_exp0b/rocksdb_data";
    int duration_sec = (argc > 2) ? atoi(argv[2]) : 70;

    Options opts;
    opts.create_if_missing = true;
    opts.write_buffer_size = 4 * 1024 * 1024;            // 2 MB memtable -> frequent L0 flushes
    opts.max_write_buffer_number = 3;
    opts.level0_file_num_compaction_trigger = 3;
    opts.max_bytes_for_level_base = 32 * 1024 * 1024;     // 8 MB L1 -> fast L1->L2->L3 compactions
    opts.max_bytes_for_level_multiplier = 4;
    opts.target_file_size_base = 4 * 1024 * 1024;        // 2 MB SSTs across levels
    opts.max_background_jobs = 4;
    opts.compression = kNoCompression;                   // realistic I/O volume without CPU bottleneck
    opts.use_direct_reads = false;                       // buffered I/O to exercise page cache reuse

    // Disable block cache to make hot SST block reads hit OS page cache / ext4_file_read_iter
    BlockBasedTableOptions table_opts;
    table_opts.no_block_cache = true;
    opts.table_factory.reset(NewBlockBasedTableFactory(table_opts));

    DB *db = nullptr;
    Status s = DB::Open(opts, db_path, &db);
    if (!s.ok()) {
        std::cerr << "RocksDB Open failed: " << s.ToString() << "\n";
        return 1;
    }

    const uint64_t NUM_KEYS = 220000;
    const uint64_t HOT_KEYS = 4000;
    std::string val_template(1024, 0x5A);

    // Pre-populate initial dataset so L0/L1/L2 exist early
    for (uint64_t i = 0; i < 140000; i++) {
        char kbuf[32];
        snprintf(kbuf, sizeof(kbuf), "key_%010llu", (unsigned long long)i);
        val_template[0] = (char)(i & 0xFF);
        db->Put(WriteOptions(), kbuf, val_template);
    }
    db->Flush(FlushOptions());

    // Thread 1: Continuous writer triggering L0 flushes & multi-level compactions
    std::thread writer([&]() {
        std::mt19937_64 rng(42);
        std::string v(1024, 0x4B);
        WriteOptions wopts;
        wopts.sync = false;
        uint64_t seq = 0;
        while (!g_stop.load()) {
            uint64_t k = rng() % NUM_KEYS;
            char kbuf[32];
            snprintf(kbuf, sizeof(kbuf), "key_%010llu", (unsigned long long)k);
            v[0] = (char)(seq++ & 0xFF);
            db->Put(wopts, kbuf, v);
            if ((seq & 63) == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
            }
        }
    });

    // Thread 2: Hot Zipfian random point lookups (Hot Reusable reads)
    std::thread reader_hot([&]() {
        std::mt19937_64 rng(123);
        ReadOptions ropts;
        std::string out;
        while (!g_stop.load()) {
            // 85% skewed to hot key range
            uint64_t k = (rng() % 100 < 85) ? (rng() % HOT_KEYS) : (rng() % NUM_KEYS);
            char kbuf[32];
            snprintf(kbuf, sizeof(kbuf), "key_%010llu", (unsigned long long)k);
            db->Get(ropts, kbuf, &out);
            std::this_thread::sleep_for(std::chrono::microseconds(150));
        }
    });

    // Thread 3: Periodic full-range iterator scan (One-Shot Scan reads)
    std::thread scanner([&]() {
        ReadOptions ropts;
        ropts.fill_cache = false;
        while (!g_stop.load()) {
            std::unique_ptr<Iterator> it(db->NewIterator(ropts));
            size_t cnt = 0;
            for (it->SeekToFirst(); it->Valid() && !g_stop.load(); it->Next()) {
                cnt += it->value().size();
                if ((cnt & 0xFFFFF) == 0)
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(duration_sec));
    g_stop.store(true);
    writer.join();
    reader_hot.join();
    scanner.join();

    // Keep DB alive briefly before closing
    std::this_thread::sleep_for(std::chrono::seconds(1));
    delete db;
    printf("RocksDB workload completed (%d s)\n", duration_sec);
    return 0;
}
