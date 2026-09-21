// engram_rows.c — gather DeepSeek-V4.1 Engram rows for one TP rank from a memory-mapped
// safetensors shard (a node-local sparse copy, or the checkpoint itself) into pinned staging
// buffers. Designed to run as a cudaLaunchHostFunc node inside a CUDA graph: no CUDA calls,
// no allocation on the hot path, fixed buffers owned by the caller and reused by every replay.
//
// Layout: rows are 256 fp8 e4m3 bytes in the `weight` tensor and 8 e8m0 bytes in the `scale`
// tensor, both [num_rows, ...] row-major at fixed byte offsets in the shard. The mapping is
// advised MADV_RANDOM so a cold row costs one 4 KiB page, not a readahead window: readahead on
// a sparse file drags in the holes around a row, which is I/O for nothing. Cold rows are paged
// in by a pool of threads (NVMe wants queue depth); a posix_fadvise(WILLNEED) prepass over all
// ids gets every read in flight before the copies start. Rows outside [row_lo, row_hi) are
// written as zeros, which is what the TP all-reduce expects from a rank that does not own them.
//
// Worker pool (ABI version 2). Spawning a thread per call cost about 1.3 ms for a decode
// lookup of ~80 rows, nearly all of it in pthread_create/join of 64 threads, so the workers are
// now started once by engram_rows_open and stopped and joined by engram_rows_close; a gather
// never creates a thread. Dispatch goes through per-worker mailboxes, not through a shared
// work counter:
//   * Every worker owns one cache line (a `Lane`) holding a 32-bit futex word `state`
//     (IDLE / ARMED / STOP), a `parked` flag it raises before sleeping in the kernel, and the
//     slice it is to gather (`work`, `lo`, `hi`). The dispatcher is the only writer of the slice
//     and writes it before flipping `state` to ARMED; the worker reads it only after it has
//     seen ARMED, so no other synchronisation is needed for the slice itself.
//   * A job of n ids is cut into L = min(n / 16, workers + 1) contiguous, near-equal slices:
//     a worker is never woken for fewer than 16 rows, and fewer than 32 ids means one slice,
//     which the caller does inline. Slice k goes to worker k; the caller keeps the last slice
//     for itself and gathers it while the workers run.
//   * Waking: the caller issues FUTEX_WAKE on a worker's own word only when that worker has
//     declared itself parked. An idle worker polls its mailbox for about a microsecond before
//     it parks, so a worker re-armed immediately costs no syscall, while in the decode loop,
//     where gathers are milliseconds apart, the caller wakes exactly the workers it uses and
//     no other thread stirs. The parked/state handshake uses sequentially consistent accesses
//     on both sides so a wake cannot be lost (the worker either sees ARMED before it sleeps or
//     the caller sees `parked` and wakes it).
//   * Completion: `pending` (one futex word per store) is set to the number of armed workers
//     before any of them is armed; each worker decrements it after its slice is written and the
//     one that brings it to zero wakes the caller, which polls for a few tens of microseconds
//     and then sleeps on it. The caller's acquire of pending == 0 is what makes every worker's
//     output visible before the host node returns and the graph's next node reads the buffers.
//   * One job at a time: a mutex serialises callers, so two threads gathering through the same
//     store cannot share the mailboxes. Under the CUDA graph only one gather is ever in flight;
//     the lock is uncontended and costs a few nanoseconds.
// The pool size is engram_rows_open's `threads` (SPARK_ENGRAM_THREADS), clamped to [0, 256]; a
// pool of 0 does everything inline. The pool's memory (one cache line per worker) is allocated
// at open time; nothing on the gather path allocates.
//
// MIT License, Copyright (c) 2026 Aiden Le.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define W_BYTES 256
#define S_BYTES 8
#define MAX_WORKERS 256        // pool size cap: a larger `threads` is clamped
#define MIN_IDS_PER_LANE 16    // a worker is never woken for fewer rows than this
#define WORKER_SPIN 256        // polls of its mailbox an idle worker makes before parking (~1 us)
#define CALLER_SPIN 20000      // polls of `pending` the caller makes before parking (~50 us)
#define LINE 128               // isolation of words written by different threads (>= any cache line)

enum { LANE_IDLE = 0u, LANE_ARMED = 1u, LANE_STOP = 2u };

_Static_assert(sizeof(atomic_uint) == 4, "futex words are 32-bit");

typedef struct Store Store;

// Exported: the caller (Python through ctypes, and the CUDA host-function node) owns this and
// the three buffers, and reuses them for every graph replay. The field order IS the ABI
// (engram_rows_abi_version() == 2): count first, ids last.
typedef struct {
  uint64_t count;       // rows to gather in this call
  Store *store;
  uint8_t *w_out;       // pinned host, [capacity, 256]
  uint8_t *s_out;       // pinned host, [capacity, 8]
  const int64_t *ids;   // pinned host, [capacity]
} Work;

// One worker's mailbox, alone on its cache line. Only the dispatcher writes work/lo/hi, and
// only before it flips `state` to ARMED; only the worker reads them, after it has seen ARMED.
typedef struct {
  _Alignas(LINE) atomic_uint state;   // futex word: LANE_IDLE / LANE_ARMED / LANE_STOP
  atomic_uint parked;                 // 1 while the worker sleeps in futex_wait on `state`
  Work *work;
  uint64_t lo, hi;                    // the slice [lo, hi) of work->ids this worker gathers
  Store *store;
  pthread_t tid;
} Lane;

struct Store {
  int fd;
  uint8_t *map;
  size_t map_len;
  uint64_t weight_off, scale_off;  // byte offsets of row 0 in the shard
  uint64_t row_lo, row_hi;         // global rows this rank owns
  int nworkers;                    // workers actually running (<= requested)
  Lane *lanes;                     // [nworkers]
  pthread_mutex_t job_lock;        // one job at a time on the mailboxes
  _Alignas(LINE) atomic_uint pending;   // futex word: workers still busy on the current job
  _Alignas(LINE) atomic_uint_fast64_t calls, rows_owned, rows_zeroed, nanos;
};

static inline void cpu_relax(void) {
#if defined(__aarch64__)
  __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  __asm__ __volatile__("pause" ::: "memory");
#else
  __asm__ __volatile__("" ::: "memory");
#endif
}

// Sleep while *word == expected; returns on wake, on a changed word, or spuriously, so every
// caller re-checks its predicate.
static void futex_wait(atomic_uint *word, unsigned expected) {
  syscall(SYS_futex, (void *)word, FUTEX_WAIT_PRIVATE, expected, (void *)0, (void *)0, 0);
}

static void futex_wake(atomic_uint *word, int nwaiters) {
  syscall(SYS_futex, (void *)word, FUTEX_WAKE_PRIVATE, nwaiters, (void *)0, (void *)0, 0);
}

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline int owns(const Store *s, int64_t id) {
  return id >= (int64_t)s->row_lo && id < (int64_t)s->row_hi;
}

static void gather_range(Work *w, uint64_t lo, uint64_t hi) {
  Store *s = w->store;
  uint64_t owned = 0, zeroed = 0;
  for (uint64_t i = lo; i < hi; i++) {
    int64_t id = w->ids[i];
    if (!owns(s, id)) {
      memset(w->w_out + i * W_BYTES, 0, W_BYTES);
      memset(w->s_out + i * S_BYTES, 0, S_BYTES);
      zeroed++;
      continue;
    }
    memcpy(w->w_out + i * W_BYTES, s->map + s->weight_off + (uint64_t)id * W_BYTES, W_BYTES);
    memcpy(w->s_out + i * S_BYTES, s->map + s->scale_off + (uint64_t)id * S_BYTES, S_BYTES);
    owned++;
  }
  atomic_fetch_add_explicit(&s->rows_owned, owned, memory_order_relaxed);
  atomic_fetch_add_explicit(&s->rows_zeroed, zeroed, memory_order_relaxed);
}

// Readahead prepass: every owned row's weight page and scale page requested at once, so the
// NVMe sees the whole job's queue depth before the first copy blocks on a fault.
static uint64_t advise_rows(const Store *s, const int64_t *ids, uint64_t n) {
  uint64_t owned = 0;
  for (uint64_t i = 0; i < n; i++) {
    int64_t id = ids[i];
    if (!owns(s, id)) continue;
    posix_fadvise(s->fd, (off_t)(s->weight_off + (uint64_t)id * W_BYTES), W_BYTES, POSIX_FADV_WILLNEED);
    posix_fadvise(s->fd, (off_t)(s->scale_off + (uint64_t)id * S_BYTES), S_BYTES, POSIX_FADV_WILLNEED);
    owned++;
  }
  return owned;
}

static void *worker_main(void *arg) {
  Lane *ln = (Lane *)arg;
  for (;;) {
    unsigned st = LANE_IDLE;
    for (int i = 0; i < WORKER_SPIN; i++) {
      st = atomic_load_explicit(&ln->state, memory_order_acquire);
      if (st != LANE_IDLE) break;
      cpu_relax();
    }
    if (st == LANE_IDLE) {
      // Park. `parked` is raised before the last look at `state`; the dispatcher stores ARMED
      // before it looks at `parked`. Both pairs are seq_cst, so at least one side sees the
      // other's write: either we observe ARMED here and never sleep, or the dispatcher observes
      // parked == 1 and issues the wake (a wake before we sleep is caught by the futex's own
      // compare of the word against LANE_IDLE).
      atomic_store_explicit(&ln->parked, 1u, memory_order_seq_cst);
      while ((st = atomic_load_explicit(&ln->state, memory_order_seq_cst)) == LANE_IDLE)
        futex_wait(&ln->state, LANE_IDLE);
      atomic_store_explicit(&ln->parked, 0u, memory_order_relaxed);
    }
    if (st == LANE_STOP) break;
    gather_range(ln->work, ln->lo, ln->hi);
    // Back to IDLE unless close() has meanwhile asked us to stop; a plain store could erase it.
    unsigned expect = LANE_ARMED;
    atomic_compare_exchange_strong_explicit(&ln->state, &expect, LANE_IDLE,
                                            memory_order_relaxed, memory_order_relaxed);
    // Release: our copies precede the decrement; the caller's acquire of 0 orders them before
    // its return.
    if (atomic_fetch_sub_explicit(&ln->store->pending, 1u, memory_order_acq_rel) == 1u)
      futex_wake(&ln->store->pending, INT_MAX);
  }
  return NULL;
}

static void wait_for_workers(Store *s) {
  for (int i = 0; i < CALLER_SPIN; i++) {
    if (atomic_load_explicit(&s->pending, memory_order_acquire) == 0u) return;
    cpu_relax();
  }
  unsigned v;
  while ((v = atomic_load_explicit(&s->pending, memory_order_acquire)) != 0u)
    futex_wait(&s->pending, v);
}

// The CUDA host-function entry point: void (*)(void *userData). Returns only when every row
// of the job has been written.
void engram_rows_gather(void *user) {
  Work *w = (Work *)user;
  Store *s = w->store;
  uint64_t t0 = now_ns();
  uint64_t n = w->count;
  atomic_fetch_add_explicit(&s->calls, 1u, memory_order_relaxed);
  if (n == 0) return;
  advise_rows(s, w->ids, n);

  uint64_t lanes = n / MIN_IDS_PER_LANE;
  if (lanes > (uint64_t)s->nworkers + 1u) lanes = (uint64_t)s->nworkers + 1u;
  if (lanes <= 1u) {
    gather_range(w, 0, n);
  } else {
    pthread_mutex_lock(&s->job_lock);
    unsigned nw = (unsigned)(lanes - 1u);
    atomic_store_explicit(&s->pending, nw, memory_order_relaxed);   // before any worker is armed
    for (unsigned k = 0; k < nw; k++) {
      Lane *ln = &s->lanes[k];
      ln->work = w;
      ln->lo = (uint64_t)k * n / lanes;
      ln->hi = (uint64_t)(k + 1u) * n / lanes;
      atomic_store_explicit(&ln->state, LANE_ARMED, memory_order_seq_cst);
      if (atomic_load_explicit(&ln->parked, memory_order_seq_cst)) futex_wake(&ln->state, 1);
    }
    gather_range(w, (uint64_t)nw * n / lanes, n);   // the caller's own slice, the last one
    wait_for_workers(s);
    pthread_mutex_unlock(&s->job_lock);
  }
  atomic_fetch_add_explicit(&s->nanos, now_ns() - t0, memory_order_relaxed);
}

static void stop_workers(Store *s) {
  for (int k = 0; k < s->nworkers; k++) {
    atomic_store_explicit(&s->lanes[k].state, LANE_STOP, memory_order_seq_cst);
    futex_wake(&s->lanes[k].state, INT_MAX);   // unconditional: cheaper than being wrong here
  }
  for (int k = 0; k < s->nworkers; k++) pthread_join(s->lanes[k].tid, NULL);
  s->nworkers = 0;
}

void engram_rows_close(Store *s);

Store *engram_rows_open(const char *path, uint64_t weight_off, uint64_t scale_off,
                        uint64_t row_lo, uint64_t row_hi, int threads) {
  Store *s = (Store *)aligned_alloc(LINE, sizeof(Store));   // sizeof is a multiple of LINE
  if (!s) return NULL;
  memset(s, 0, sizeof *s);
  s->fd = open(path, O_RDONLY | O_CLOEXEC);
  if (s->fd < 0) { free(s); return NULL; }
  struct stat st;
  if (fstat(s->fd, &st) != 0) { close(s->fd); free(s); return NULL; }
  s->map_len = (size_t)st.st_size;
  // The owned range must lie inside the file: a row past its end would SIGBUS inside the
  // graph callback, where nothing can catch it. Refuse at open time instead.
  if (row_hi > row_lo &&
      (weight_off > s->map_len || (s->map_len - weight_off) / W_BYTES < row_hi ||
       scale_off > s->map_len || (s->map_len - scale_off) / S_BYTES < row_hi)) {
    close(s->fd); free(s); errno = EINVAL; return NULL;
  }
  s->map = (uint8_t *)mmap(NULL, s->map_len, PROT_READ, MAP_SHARED, s->fd, 0);
  if (s->map == MAP_FAILED) { close(s->fd); free(s); return NULL; }
  madvise(s->map, s->map_len, MADV_RANDOM);
  s->weight_off = weight_off; s->scale_off = scale_off;
  s->row_lo = row_lo; s->row_hi = row_hi;
  pthread_mutex_init(&s->job_lock, NULL);
  atomic_init(&s->pending, 0u);

  if (threads < 0) threads = 0;
  if (threads > MAX_WORKERS) threads = MAX_WORKERS;
  if (threads > 0) {
    s->lanes = (Lane *)aligned_alloc(LINE, (size_t)threads * sizeof(Lane));
    if (!s->lanes) { engram_rows_close(s); return NULL; }
    memset(s->lanes, 0, (size_t)threads * sizeof(Lane));
    for (int k = 0; k < threads; k++) {
      Lane *ln = &s->lanes[k];
      ln->store = s;
      atomic_init(&ln->state, LANE_IDLE);
      atomic_init(&ln->parked, 0u);
      if (pthread_create(&ln->tid, NULL, worker_main, ln) != 0) break;   // smaller pool, still works
      char name[16];
      snprintf(name, sizeof name, "engram-rows%d", k);
      pthread_setname_np(ln->tid, name);
      s->nworkers = k + 1;
    }
  }
  return s;
}

void engram_rows_close(Store *s) {
  if (!s) return;
  stop_workers(s);
  free(s->lanes);
  pthread_mutex_destroy(&s->job_lock);
  if (s->map && s->map != MAP_FAILED) munmap(s->map, s->map_len);
  if (s->fd >= 0) close(s->fd);
  free(s);
}

// Off the hot path (the prefetch thread): ask the page cache for the rows of a coming chunk,
// no copy. Returns the number of owned ids advised.
uint64_t engram_rows_prefetch(Store *s, const int64_t *ids, uint64_t n) {
  if (!s || !ids) return 0;
  return advise_rows(s, ids, n);
}

void engram_rows_stats(Store *s, uint64_t out[4]) {
  out[0] = atomic_load_explicit(&s->calls, memory_order_relaxed);
  out[1] = atomic_load_explicit(&s->rows_owned, memory_order_relaxed);
  out[2] = atomic_load_explicit(&s->rows_zeroed, memory_order_relaxed);
  out[3] = atomic_load_explicit(&s->nanos, memory_order_relaxed);
}

int engram_rows_abi_version(void) { return 2; }
