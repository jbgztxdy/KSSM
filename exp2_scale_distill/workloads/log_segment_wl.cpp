#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <random>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

static std::atomic<bool> g_stop{false};

static void write_file_full(const std::string &p, const char *buf, size_t sz, bool do_sync, bool append_mode) {
    int fl = O_CREAT | O_WRONLY | (append_mode ? O_APPEND : O_TRUNC);
    int fd = open(p.c_str(), fl, 0644);
    if (fd < 0) return;
    size_t off = 0;
    while (off < sz) {
        size_t chunk = std::min<size_t>(131072, sz - off);
        ssize_t r = write(fd, buf + off, chunk);
        if (r <= 0) break;
        off += (size_t)r;
    }
    if (do_sync) fdatasync(fd);
    close(fd);
}

int main(int argc, char **argv) {
    std::string base = (argc > 1) ? argv[1] : "/mnt/nvme12/yjw_kssm_exp2/logseg";
    int dur = (argc > 2) ? atoi(argv[2]) : 65;

    std::string d_hot  = base + "/hot_tail";
    std::string d_cold = base + "/cold_hist";
    std::string d_s0   = base + "/micro_wal";
    std::string d_s1   = base + "/active_seg";
    std::string d_s2   = base + "/compacted_seg";
    for (auto &d : {base, d_hot, d_cold, d_s0, d_s1, d_s2}) mkdir(d.c_str(), 0755);

    std::vector<char> bigbuf(16 * 1024 * 1024, 0x3C);
    std::vector<char> segbuf(1024 * 1024, 0x5A);
    std::string hot_idx = d_hot + "/offset_index.idx";
    write_file_full(hot_idx, segbuf.data(), 512 * 1024, false, false);

    std::vector<std::string> cold_segs;
    for (int i = 0; i < 14; i++) {
        std::string p = d_cold + "/hist_" + std::to_string(i) + ".log";
        write_file_full(p, bigbuf.data(), 16 * 1024 * 1024, false, false);
        cold_segs.push_back(p);
    }

    // Thread 1: Hot tail consumer (random reads on offset index -> Hot Reusable)
    std::thread t_hot([&]() {
        std::mt19937 rng(7);
        std::vector<char> rbuf(8192);
        while (!g_stop.load()) {
            int fd = open(hot_idx.c_str(), O_RDONLY);
            if (fd >= 0) {
                for (int k = 0; k < 10; k++) {
                    off_t pos = (off_t)(rng() % 100) * 4096;
                    ssize_t r = pread(fd, rbuf.data(), 8192, pos);
                    (void)r;
                }
                close(fd);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    });

    // Thread 2: Audit sequential scanner over 14 cold segments (One-Shot Scan)
    std::thread t_scan([&]() {
        std::vector<char> rbuf(524288);
        for (auto &p : cold_segs) {
            if (g_stop.load()) break;
            int fd = open(p.c_str(), O_RDONLY);
            if (fd >= 0) {
                while (!g_stop.load() && read(fd, rbuf.data(), rbuf.size()) > 0) {}
                close(fd);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2800));
        }
    });

    // Thread 3: Log writer & segment compactor (Short micro-WAL / Medium active seg / Long compacted seg)
    std::thread t_wr([&]() {
        std::deque<std::pair<double, std::string>> q0, q1;
        auto t0 = std::chrono::steady_clock::now();
        int seq = 0;
        std::vector<char> tmp_r(131072);
        while (!g_stop.load()) {
            seq++;
            double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            // Short-lived micro WAL (append mode, deleted in 2.5s)
            std::string p0 = d_s0 + "/wal_" + std::to_string(seq) + ".aof";
            write_file_full(p0, segbuf.data(), 512 * 1024, false, true);
            q0.push_back({now, p0});

            // Medium-lived active segment (16s TTL)
            if (seq % 2 == 0) {
                if (!q0.empty()) {
                    int fd = open(q0.back().second.c_str(), O_RDONLY);
                    if (fd >= 0) { ssize_t r = read(fd, tmp_r.data(), tmp_r.size()); (void)r; close(fd); }
                }
                std::string p1 = d_s1 + "/seg_" + std::to_string(seq) + ".log";
                write_file_full(p1, segbuf.data(), 1024 * 1024, true, false);
                q1.push_back({now, p1});
            }

            // Long-lived compacted segment (reads active seg + hot index + fsync + rename)
            if (seq % 4 == 0) {
                int fd = open(hot_idx.c_str(), O_RDONLY);
                if (fd >= 0) { ssize_t r = read(fd, tmp_r.data(), 65536); (void)r; close(fd); }
                std::string pt = d_s2 + "/comp_" + std::to_string(seq) + ".tmp";
                std::string pf = d_s2 + "/comp_" + std::to_string(seq) + ".seg";
                write_file_full(pt, segbuf.data(), 1024 * 1024, true, false);
                rename(pt.c_str(), pf.c_str());
            }

            while (!q0.empty() && now - q0.front().first >= 2.5) {
                unlink(q0.front().second.c_str()); q0.pop_front();
            }
            while (!q1.empty() && now - q1.front().first >= 16.0) {
                unlink(q1.front().second.c_str()); q1.pop_front();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(450));
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(dur));
    g_stop.store(true);
    t_hot.join(); t_scan.join(); t_wr.join();
    printf("log_segment_kv completed (%ds)\n", dur);
    return 0;
}
