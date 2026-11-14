/* Thread pool for parallel_for.
 *
 * A frame needs a few hundred short parallel loops (the DIS solver alone has
 * ~200), so waking threads through a condition variable each time (~25 us)
 * would cost several ms per frame. Workers therefore spin on an atomic job
 * counter for a short while and only then go to sleep; the caller spins until
 * the workers are done.
 */
#include "parallel.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "common.h"

#define MAX_THREADS 64
#define SPIN_US 300.0 /* wait this long for the next job before sleeping */
#define BUSY_SPINS 2000 /* then yield the core to other processes (e.g. ffmpeg) */

#if defined(__x86_64__) || defined(__i386__)
#define CPU_RELAX() __builtin_ia32_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define CPU_RELAX() __asm__ __volatile__("yield")
#else
#define CPU_RELAX() ((void)0)
#endif

static struct {
    int n; /* total threads including the caller */
    pthread_t th[MAX_THREADS];
    pthread_mutex_t mu;
    pthread_cond_t go;
    int sleeping; /* workers waiting on `go` (guarded by mu) */
    atomic_ulong gen;   /* job generation */
    atomic_int pending; /* workers still busy with the current job */
    atomic_int quit;
    /* current job (written before gen is increased) */
    range_fn fn;
    void *ctx;
    int total, parts;
} P;

void *xalloc(size_t bytes) {
    void *p = NULL;
    if (bytes == 0) bytes = 64;
    bytes = (bytes + 63) & ~(size_t)63;
    if (posix_memalign(&p, 64, bytes) != 0 || !p) {
        fprintf(stderr, "out of memory (%zu bytes)\n", bytes);
        exit(1);
    }
    memset(p, 0, bytes);
    return p;
}
void xfree(void *p) { free(p); }

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

static void run_part(int part) {
    const long long a = (long long)P.total * part / P.parts;
    const long long b = (long long)P.total * (part + 1) / P.parts;
    if (b > a) P.fn(P.ctx, (int)a, (int)b);
}

static void *worker(void *arg) {
    const int id = (int)(intptr_t)arg; /* 1 .. n-1 */
    unsigned long seen = 0;
    for (;;) {
        /* wait for a new job: spin first, then sleep */
        double t0 = 0;
        int spins = 0;
        while (atomic_load_explicit(&P.gen, memory_order_acquire) == seen) {
            if (atomic_load_explicit(&P.quit, memory_order_relaxed)) return NULL;
            if (spins < BUSY_SPINS) CPU_RELAX();
            else sched_yield();
            if ((++spins & 255) == 0) {
                if (t0 == 0) t0 = now_us();
                else if (now_us() - t0 > SPIN_US) {
                    pthread_mutex_lock(&P.mu);
                    P.sleeping++;
                    while (atomic_load_explicit(&P.gen, memory_order_acquire) == seen &&
                           !atomic_load_explicit(&P.quit, memory_order_relaxed))
                        pthread_cond_wait(&P.go, &P.mu);
                    P.sleeping--;
                    pthread_mutex_unlock(&P.mu);
                    t0 = 0;
                    spins = 0;
                }
            }
        }
        seen = atomic_load_explicit(&P.gen, memory_order_acquire);
        if (id < P.parts) run_part(id);
        atomic_fetch_sub_explicit(&P.pending, 1, memory_order_acq_rel);
    }
}

void pool_init(int nthreads) {
    if (nthreads <= 0) {
        long c = sysconf(_SC_NPROCESSORS_ONLN);
        nthreads = c > 0 ? (int)c : 1;
    }
    if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;
    P.n = nthreads;
    pthread_mutex_init(&P.mu, NULL);
    pthread_cond_init(&P.go, NULL);
    atomic_store(&P.gen, 0);
    atomic_store(&P.pending, 0);
    atomic_store(&P.quit, 0);
    for (int i = 1; i < P.n; ++i) pthread_create(&P.th[i], NULL, worker, (void *)(intptr_t)i);
}

void pool_shutdown(void) {
    atomic_store(&P.quit, 1);
    pthread_mutex_lock(&P.mu);
    pthread_cond_broadcast(&P.go);
    pthread_mutex_unlock(&P.mu);
    for (int i = 1; i < P.n; ++i) pthread_join(P.th[i], NULL);
    P.n = 1;
}

int pool_threads(void) { return P.n > 0 ? P.n : 1; }

void parallel_for(int n, int grain, range_fn fn, void *ctx) {
    if (n <= 0) return;
    if (grain < 1) grain = 1;
    int parts = n / grain;
    if (parts > P.n) parts = P.n;
    if (parts <= 1 || P.n <= 1) {
        fn(ctx, 0, n);
        return;
    }
    P.fn = fn;
    P.ctx = ctx;
    P.total = n;
    P.parts = parts;
    atomic_store_explicit(&P.pending, P.n - 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&P.gen, 1, memory_order_acq_rel);
    pthread_mutex_lock(&P.mu);
    if (P.sleeping) pthread_cond_broadcast(&P.go);
    pthread_mutex_unlock(&P.mu);

    run_part(0);

    for (int spins = 0; atomic_load_explicit(&P.pending, memory_order_acquire) > 0; ++spins) {
        if (spins < BUSY_SPINS) CPU_RELAX();
        else sched_yield();
    }
}
