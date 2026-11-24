#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "kssm_exp0c_shared.h"

#define PACK4(v) (((uint64_t)(v)) | ((uint64_t)(v)<<16) | ((uint64_t)(v)<<32) | ((uint64_t)(v)<<48))
#define FD_CACHE_SZ 256

struct fd_entry {
    uint64_t ino;
    uint32_t pid;
    int      fd;
    uint64_t dropped_bytes;
    uint64_t drop_calls;
    char     path[256];
};

static struct fd_entry g_cache[FD_CACHE_SZ];
static int g_cache_n = 0;
static volatile sig_atomic_t g_stop = 0;
static uint64_t g_total_dropped_bytes = 0;
static uint64_t g_total_drop_calls = 0;
static uint64_t g_fd_miss_drops = 0;

static void on_sig(int s) { (void)s; g_stop = 1; }

static struct fd_entry *resolve_ino_fd(uint32_t pid, uint64_t ino)
{
    for (int i = 0; i < g_cache_n; i++) {
        if (g_cache[i].ino == ino && g_cache[i].fd >= 0) {
            return &g_cache[i];
        }
    }
    /* Scan /proc/<pid>/fd ONCE per newly detected scan file, then cache! */
    char dirp[64];
    snprintf(dirp, sizeof(dirp), "/proc/%u/fd", pid);
    DIR *d = opendir(dirp);
    if (!d) return NULL;

    struct dirent *ent;
    int found_fd = -1;
    char link_target[256] = {0};

    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char fdp[128];
        snprintf(fdp, sizeof(fdp), "/proc/%u/fd/%s", pid, ent->d_name);
        struct stat st;
        if (stat(fdp, &st) == 0 && (uint64_t)st.st_ino == ino) {
            ssize_t rl = readlink(fdp, link_target, sizeof(link_target) - 1);
            if (rl > 0) link_target[rl] = '\0';
            /* Open read-only handle via /proc/<pid>/fd/<n> -> works even if unlinked! */
            found_fd = open(fdp, O_RDONLY);
            break;
        }
    }
    closedir(d);

    if (found_fd < 0) return NULL;

    int slot;
    if (g_cache_n < FD_CACHE_SZ) {
        slot = g_cache_n++;
    } else {
        /* Evict oldest slot */
        slot = 0;
        if (g_cache[0].fd >= 0) close(g_cache[0].fd);
    }
    g_cache[slot].ino = ino;
    g_cache[slot].pid = pid;
    g_cache[slot].fd = found_fd;
    g_cache[slot].dropped_bytes = 0;
    g_cache[slot].drop_calls = 0;
    snprintf(g_cache[slot].path, sizeof(g_cache[slot].path), "%s", link_target[0] ? link_target : "(unknown)");
    return &g_cache[slot];
}

static int handle_hint(void *ctx, void *data, size_t sz)
{
    (void)ctx;
    if (sz != sizeof(struct kssm_drop_hint)) return 0;
    const struct kssm_drop_hint *h = (const struct kssm_drop_hint *)data;

    struct fd_entry *fe = resolve_ino_fd(h->pid, h->ino);
    if (!fe) {
        g_fd_miss_drops++;
        return 0;
    }

    if (h->is_write) {
        /* Async writeback + drop-behind for streaming checkpoint writes */
        sync_file_range(fe->fd, (off64_t)h->drop_start, (off64_t)h->drop_len,
                        SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
        posix_fadvise(fe->fd, (off_t)h->drop_start, (off_t)h->drop_len, POSIX_FADV_DONTNEED);
    } else {
        /* Pure drop-behind for One-Shot Scan reads (leaves 16MB ahead untouched for readahead!) */
        posix_fadvise(fe->fd, (off_t)h->drop_start, (off_t)h->drop_len, POSIX_FADV_DONTNEED);
    }
    fe->dropped_bytes += h->drop_len;
    fe->drop_calls++;
    g_total_dropped_bytes += h->drop_len;
    g_total_drop_calls++;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <mount_dir> <kssm|heuristic_size> <weights.bin> [thresh_q8]\n", argv[0]);
        return 1;
    }
    const char *mnt_dir = argv[1];
    int mode = (strcmp(argv[2], "heuristic_size") == 0) ? MODE_HEURISTIC_SIZE : MODE_KSSM;
    const char *weights_path = argv[3];
    int32_t thresh_q8 = (argc >= 5) ? atoi(argv[4]) : 0;

    struct stat st;
    if (stat(mnt_dir, &st) != 0) { perror("stat"); return 1; }
    uint32_t kdev = (major(st.st_dev) << 20) | minor(st.st_dev);

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    struct bpf_object *obj = bpf_object__open_file("/home/yjw/workspace/research/KSSM/exp0c/kssm_exp0c.bpf.o", NULL);
    if (!obj) { fprintf(stderr, "open bpf failed\n"); return 1; }

    struct bpf_map *m;
    bpf_object__for_each_map(m, obj) {
        if (strstr(bpf_map__name(m), ".rodata")) {
            size_t sz = 0;
            uint32_t *v = (uint32_t *)bpf_map__initial_value(m, &sz);
            if (v && sz >= 12) {
                v[0] = kdev;
                v[1] = (uint32_t)mode;
                v[2] = (uint32_t)thresh_q8;
            }
            break;
        }
    }

    if (bpf_object__load(obj) != 0) { fprintf(stderr, "BPF load failed (verifier rejected?)\n"); return 1; }

    /* Load real Exp 0-B trained Q5.8 weights into weights_map! */
    int wfd = bpf_object__find_map_fd_by_name(obj, "weights_map");
    struct kssm_live_weights *w = calloc(1, sizeof(*w));
    FILE *wf = fopen(weights_path, "rb");
    if (!wf || fread(w, sizeof(*w), 1, wf) != 1) {
        fprintf(stderr, "Failed to read weights from %s (expected %zu bytes)\n", weights_path, sizeof(*w));
        return 1;
    }
    fclose(wf);
    uint32_t k0 = 0;
    if (bpf_map_update_elem(wfd, &k0, w, BPF_ANY) != 0) { perror("update weights_map"); return 1; }
    free(w);

    /* Initialize zero_inode_map with Excess-16384 neutral hidden state! */
    int zfd = bpf_object__find_map_fd_by_name(obj, "zero_inode_map");
    /* Match struct kssm_inode_st size (176 bytes): first 16 u64 = PACK4(16384) */
    uint8_t zbuf[256] = {0};
    uint64_t *zh = (uint64_t *)zbuf;
    for (int g = 0; g < 16; g++) zh[g] = PACK4(16384ULL);
    if (bpf_map_update_elem(zfd, &k0, zbuf, BPF_ANY) != 0) { perror("update zero_inode_map"); return 1; }

    struct bpf_link *links[8] = {0};
    int nl = 0;
    struct bpf_program *p;
    bpf_object__for_each_program(p, obj) {
        struct bpf_link *l = bpf_program__attach(p);
        if (libbpf_get_error(l)) { fprintf(stderr, "attach failed: %s\n", bpf_program__name(p)); return 1; }
        links[nl++] = l;
    }

    int rb_fd = bpf_object__find_map_fd_by_name(obj, "drop_rb");
    struct ring_buffer *rb = ring_buffer__new(rb_fd, handle_hint, NULL, NULL);
    if (!rb) { fprintf(stderr, "ringbuf new failed\n"); return 1; }

    fprintf(stderr, "READY mode=%s kdev=0x%x thresh_q8=%d hooks=%d\n", argv[2], kdev, thresh_q8, nl);
    fflush(stderr);

    while (!g_stop)
        ring_buffer__poll(rb, 10);
    ring_buffer__consume(rb);

    /* Print full transparent audit report! */
    printf("=== KSSM EXP0C DAEMON AUDIT (mode=%s) ===\n", argv[2]);
    bpf_object__for_each_program(p, obj) {
        struct bpf_prog_info info = {};
        uint32_t ilen = sizeof(info);
        if (bpf_obj_get_info_by_fd(bpf_program__fd(p), &info, &ilen) == 0) {
            double avg = info.run_cnt ? (double)info.run_time_ns / info.run_cnt : 0.0;
            printf("  prog %-16s | run_cnt=%10llu | run_time_ms=%8.2f | avg_ns=%.2f\n",
                   bpf_program__name(p), (unsigned long long)info.run_cnt, info.run_time_ns / 1e6, avg);
        }
    }

    int sfd = bpf_object__find_map_fd_by_name(obj, "stats_map");
    int ncpus = libbpf_num_possible_cpus();
    struct kssm_exp0c_stats *sv = calloc(ncpus, sizeof(*sv));
    if (sv && bpf_map_lookup_elem(sfd, &k0, sv) == 0) {
        struct kssm_exp0c_stats tot = {0};
        for (int i = 0; i < ncpus; i++) {
            tot.total_rw_hooks += sv[i].total_rw_hooks;
            tot.ssm_updates    += sv[i].ssm_updates;
            tot.scan_pos_votes += sv[i].scan_pos_votes;
            tot.hot_neg_votes  += sv[i].hot_neg_votes;
            tot.hints_emitted  += sv[i].hints_emitted;
            tot.hints_dropped  += sv[i].hints_dropped;
        }
        printf("  rw_hooks=%llu | ssm_updates=%llu | scan_votes=%llu | hot_votes=%llu | hints=%llu (rb_drop=%llu)\n",
               (unsigned long long)tot.total_rw_hooks, (unsigned long long)tot.ssm_updates,
               (unsigned long long)tot.scan_pos_votes, (unsigned long long)tot.hot_neg_votes,
               (unsigned long long)tot.hints_emitted, (unsigned long long)tot.hints_dropped);
    }
    free(sv);

    printf("  total_fadvise_calls=%llu | total_evicted=%.2f GiB | fd_miss=%llu | distinct_files=%d\n",
           (unsigned long long)g_total_drop_calls, g_total_dropped_bytes / (1024.0*1024.0*1024.0),
           (unsigned long long)g_fd_miss_drops, g_cache_n);

    /* Transparent per-file audit: count RocksDB SST false positives vs safetensors/ckpt true positives! */
    uint64_t sst_bytes = 0, st_bytes = 0, ckpt_bytes = 0, other_bytes = 0;
    int sst_files = 0, st_files = 0, ckpt_files = 0;
    for (int i = 0; i < g_cache_n; i++) {
        if (strstr(g_cache[i].path, ".sst")) { sst_bytes += g_cache[i].dropped_bytes; sst_files++; }
        else if (strstr(g_cache[i].path, ".safetensors")) { st_bytes += g_cache[i].dropped_bytes; st_files++; }
        else if (strstr(g_cache[i].path, ".ckpt")) { ckpt_bytes += g_cache[i].dropped_bytes; ckpt_files++; }
        else { other_bytes += g_cache[i].dropped_bytes; }
        if (g_cache[i].fd >= 0) close(g_cache[i].fd);
    }
    printf("  AUDIT_BREAKDOWN: .safetensors=%d files (%.2f GiB) | .ckpt=%d files (%.2f GiB) | RocksDB .sst FP=%d files (%.2f GiB) | other=%.2f GiB\n",
           st_files, st_bytes / (1024.0*1024.0*1024.0),
           ckpt_files, ckpt_bytes / (1024.0*1024.0*1024.0),
           sst_files, sst_bytes / (1024.0*1024.0*1024.0),
           other_bytes / (1024.0*1024.0*1024.0));
    fflush(stdout);

    ring_buffer__free(rb);
    for (int i = 0; i < nl; i++) bpf_link__destroy(links[i]);
    bpf_object__close(obj);
    return 0;
}
