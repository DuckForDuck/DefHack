/* Shared helpers: aligned buffers, SIMD vector types, small maths.
 *
 * Vectorisation uses the GCC/Clang vector extensions: the same source
 * compiles to NEON on Apple Silicon / ARM and to SSE/AVX on x86.
 * A vector of 8 floats (v8f) is one AVX register or two NEON registers.
 */
#ifndef THERMAL_COMMON_H
#define THERMAL_COMMON_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define VW 8 /* floats per vector */

/* "the arrays in this loop do not overlap": lets GCC and Clang vectorise
 * loops that touch many arrays */
#if defined(__clang__)
#define IVDEP _Pragma("clang loop vectorize(assume_safety)")
#elif defined(__GNUC__)
#define IVDEP _Pragma("GCC ivdep")
#else
#define IVDEP
#endif
typedef float v8f __attribute__((vector_size(32)));
typedef double v4d __attribute__((vector_size(32)));
typedef int32_t v8i __attribute__((vector_size(32)));

static inline v8f v8f_load(const float *p) {
    v8f v;
    memcpy(&v, p, sizeof v);
    return v;
}
static inline void v8f_store(float *p, v8f v) { memcpy(p, &v, sizeof v); }
static inline v8f v8f_set1(float a) { return (v8f){a, a, a, a, a, a, a, a}; }
/* lane-wise select: m ? a : b (m from a vector comparison: all bits set or 0) */
static inline v8f v8f_select(v8i m, v8f a, v8f b) { return (v8f)(((v8i)a & m) | ((v8i)b & ~m)); }
static inline v8f v8f_min(v8f a, v8f b) { return v8f_select(a < b, a, b); }
static inline v8f v8f_max(v8f a, v8f b) { return v8f_select(a > b, a, b); }
/* 8 bytes / 8 int16 -> 8 floats (this form gives one widening load and one
 * int->float conversion with both GCC and Clang, on x86 and ARM) */
static inline v8f v8f_from_u8(const uint8_t *p) {
    const v8i t = {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]};
    return __builtin_convertvector(t, v8f);
}
static inline v8f v8f_from_i16(const int16_t *p) {
    const v8i t = {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]};
    return __builtin_convertvector(t, v8f);
}
static inline float v8f_sum(v8f v) {
    /* pairwise, like a SIMD reduce */
    return ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7]));
}

static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float minf(float a, float b) { return a < b ? a : b; }
static inline float maxf(float a, float b) { return a > b ? a : b; }
static inline double mind(double a, double b) { return a < b ? a : b; }
static inline double maxd(double a, double b) { return a > b ? a : b; }

/* OpenCV border modes used by the pipeline */
/* REFLECT: fedcba|abcdef (edge repeated); REFLECT_101: fedcb|abcdef (edge not repeated) */
static inline int border_reflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i - 1;
        if (i >= n) i = 2 * n - 1 - i;
    }
    return i;
}
static inline int border_reflect101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
    }
    return i;
}

/* 64-byte aligned allocation (zeroed); exits on failure */
void *xalloc(size_t bytes);
#define ALLOC(T, n) ((T *)xalloc(sizeof(T) * (size_t)(n)))
void xfree(void *p);

#endif
