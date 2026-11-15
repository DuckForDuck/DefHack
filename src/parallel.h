/* Tiny thread pool: parallel_for over [0, n) split into contiguous ranges.
 * The calling thread takes part in the work. */
#ifndef THERMAL_PARALLEL_H
#define THERMAL_PARALLEL_H

typedef void (*range_fn)(void *ctx, int start, int end);

/* nthreads <= 0: number of CPU cores */
void pool_init(int nthreads);
void pool_shutdown(void);
int pool_threads(void);

/* Calls fn(ctx, a, b) on disjoint ranges covering [0, n).
 * grain = smallest range worth giving to one thread. */
void parallel_for(int n, int grain, range_fn fn, void *ctx);

#endif
