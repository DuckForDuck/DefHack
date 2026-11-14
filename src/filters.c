#include "filters.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "parallel.h"

/* ====================================================================== */
/* statistics                                                             */
/* ====================================================================== */

static inline void swapf(float *a, float *b) {
    float t = *a;
    *a = *b;
    *b = t;
}

float select_kth(float *v, int n, int k) {
    int lo = 0, hi = n - 1;
    while (hi > lo) {
        if (hi - lo < 24) { /* insertion sort of the small rest */
            for (int i = lo + 1; i <= hi; ++i) {
                float t = v[i];
                int j = i - 1;
                while (j >= lo && v[j] > t) { v[j + 1] = v[j]; --j; }
                v[j + 1] = t;
            }
            break;
        }
        const int mid = lo + ((hi - lo) >> 1);
        if (v[mid] < v[lo]) swapf(&v[mid], &v[lo]);
        if (v[hi] < v[lo]) swapf(&v[hi], &v[lo]);
        if (v[hi] < v[mid]) swapf(&v[hi], &v[mid]);
        const float pivot = v[mid];
        int i = lo, j = hi;
        while (i <= j) {
            while (v[i] < pivot) ++i;
            while (v[j] > pivot) --j;
            if (i <= j) { swapf(&v[i], &v[j]); ++i; --j; }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break; /* v[k] == pivot */
    }
    return v[k];
}

static float min_of(const float *v, int n) {
    float m = v[0];
    for (int i = 1; i < n; ++i) m = v[i] < m ? v[i] : m;
    return m;
}
static float max_of(const float *v, int n) {
    float m = v[0];
    for (int i = 1; i < n; ++i) m = v[i] > m ? v[i] : m;
    return m;
}

/* Exact order statistics of a large array without sorting it: a histogram
 * of the 16 high bits of the (order-preserving) float key finds the bin of
 * the wanted rank, then only that bin is searched. Two passes over the data
 * instead of the ~3 of a quickselect, and no copy of the input.
 * Uses static buffers: call from one thread at a time. */
#define RADIX_MIN 16384
static inline uint32_t fkey(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* out[0] = k-th smallest, out[1] = (k+1)-th smallest (= out[0] if k+1 == n) */
static void radix_pair(const float *v, int n, int k, double out[2]) {
    static int hist[65536];
    static float *buf[2] = {NULL, NULL};
    static int cap[2] = {0, 0};
    const float *src = v;
    int cnt = n, k1 = k + 1 < n ? k + 1 : k, which = 0;
    uint32_t base = 0;
    int shift = 16;
    for (;;) {
        /* histogram of (key - base) >> shift; all keys of src are >= base */
        memset(hist, 0, sizeof hist);
        for (int i = 0; i < cnt; ++i) hist[(fkey(src[i]) - base) >> shift]++;
        int below = 0, b0 = 0;
        while (below + hist[b0] <= k) below += hist[b0++];
        int b1 = b0, upto = below + hist[b0];
        while (upto <= k1) upto += hist[++b1];
        const int m = upto - below;
        if (m + 1 > cap[which]) {
            xfree(buf[which]);
            cap[which] = m + 1;
            buf[which] = ALLOC(float, cap[which]);
        }
        float *dst = buf[which];
        int c = 0; /* branch-free: always store, advance only on a hit */
        const uint32_t lo_b = (uint32_t)b0, span = (uint32_t)(b1 - b0);
        for (int i = 0; i < cnt; ++i) {
            dst[c] = src[i];
            c += ((fkey(src[i]) - base) >> shift) - lo_b <= span;
        }
        k -= below;
        k1 -= below;
        src = dst;
        cnt = m;
        if (m <= 4096 || shift == 0) break;
        /* refine inside the bins b0..b1 */
        base += (uint32_t)b0 << shift;
        const uint32_t range = ((span + 1) << shift) - 1; /* largest key - base */
        int bits = 0;
        while (bits < 32 && (range >> bits) != 0) ++bits;
        shift = bits > 16 ? bits - 16 : 0;
        which ^= 1;
    }
    float *w = (float *)src;
    out[0] = select_kth(w, cnt, k);
    out[1] = k1 != k && k + 1 < cnt ? min_of(w + k + 1, cnt - k - 1) : out[0];
}

double percentile(float *v, int n, double p) {
    if (n <= 0) return 0.0;
    const double idx = p / 100.0 * (double)(n - 1);
    const int lo = (int)floor(idx);
    const double t = idx - lo;
    double a, b;
    if (n >= RADIX_MIN) {
        double ab[2];
        radix_pair(v, n, lo, ab);
        a = ab[0];
        b = ab[1];
    } else {
        a = select_kth(v, n, lo);
        b = lo + 1 < n ? min_of(v + lo + 1, n - lo - 1) : a;
    }
    const double d = b - a; /* numpy lerp, exact at both ends */
    return t >= 0.5 ? b - d * (1.0 - t) : a + d * t;
}

double median(float *v, int n) {
    if (n <= 0) return 0.0;
    const int m = n / 2;
    if (n >= RADIX_MIN) {
        double ab[2];
        radix_pair(v, n, n % 2 ? m : m - 1, ab);
        return n % 2 ? ab[0] : 0.5 * (ab[0] + ab[1]);
    }
    const double hi = select_kth(v, n, m);
    if (n % 2) return hi;
    const double lo = max_of(v, m);
    return 0.5 * (lo + hi);
}

double estimate_noise_sigma(const float *img, int w, int h, const uint8_t *mask, float *lap) {
    int n = 0;
    for (int y = 1; y < h - 1; y += 2) {
        const float *up = img + (size_t)(y - 1) * w, *r = img + (size_t)y * w, *dn = img + (size_t)(y + 1) * w;
        const uint8_t *m = mask ? mask + (size_t)y * w : NULL;
        for (int x = 1; x < w - 1; x += 2) {
            if (m && m[x]) continue;
            lap[n++] = r[x] - 0.25f * up[x] - 0.25f * dn[x] - 0.25f * r[x - 1] - 0.25f * r[x + 1];
        }
    }
    if (n == 0) return 0.0;
    float *tmp = lap + n; /* second half of the scratch */
    memcpy(tmp, lap, sizeof(float) * (size_t)n);
    const double med = median(tmp, n);
    for (int i = 0; i < n; ++i) lap[i] = fabsf((float)(lap[i] - med));
    const double mad = median(lap, n);
    return 1.4826 * mad / sqrt(1.25);
}

/* ====================================================================== */
/* Gaussian blur (separable, symmetric kernel)                            */
/* ====================================================================== */

#define MAX_KSIZE 129

typedef struct {
    const float *src;
    float *dst;
    int w, h, r;
    float k[MAX_KSIZE]; /* k[0] = centre, k[i] = distance i */
} SepCtx;

static void gauss_rows(void *vc, int y0, int y1) {
    SepCtx *c = vc;
    const int w = c->w, r = c->r;
    float pad[w + 2 * r + VW];
    for (int y = y0; y < y1; ++y) {
        const float *s = c->src + (size_t)y * w;
        float *d = c->dst + (size_t)y * w;
        for (int i = 0; i < r; ++i) {
            pad[i] = s[border_reflect(i - r, w)];
            pad[r + w + i] = s[border_reflect(w + i, w)];
        }
        memcpy(pad + r, s, sizeof(float) * (size_t)w);
        const float *p = pad + r;
        int x = 0;
        for (; x + VW <= w; x += VW) {
            v8f acc = v8f_load(p + x) * c->k[0];
            for (int i = 1; i <= r; ++i) acc += (v8f_load(p + x - i) + v8f_load(p + x + i)) * c->k[i];
            v8f_store(d + x, acc);
        }
        for (; x < w; ++x) {
            float acc = p[x] * c->k[0];
            for (int i = 1; i <= r; ++i) acc += (p[x - i] + p[x + i]) * c->k[i];
            d[x] = acc;
        }
    }
}

static void gauss_cols(void *vc, int y0, int y1) {
    SepCtx *c = vc;
    const int w = c->w, h = c->h, r = c->r;
    const float *rows[MAX_KSIZE];
    for (int y = y0; y < y1; ++y) {
        for (int i = -r; i <= r; ++i) rows[i + r] = c->src + (size_t)border_reflect(y + i, h) * w;
        const float *const *R = rows + r;
        float *d = c->dst + (size_t)y * w;
        int x = 0;
        for (; x + VW <= w; x += VW) {
            v8f acc = v8f_load(R[0] + x) * c->k[0];
            for (int i = 1; i <= r; ++i) acc += (v8f_load(R[-i] + x) + v8f_load(R[i] + x)) * c->k[i];
            v8f_store(d + x, acc);
        }
        for (; x < w; ++x) {
            float acc = R[0][x] * c->k[0];
            for (int i = 1; i <= r; ++i) acc += (R[-i][x] + R[i][x]) * c->k[i];
            d[x] = acc;
        }
    }
}

void gaussian_blur(const float *src, float *dst, int w, int h, double sigma, float *tmp) {
    int ksize = ((int)lrint(sigma * 4 * 2 + 1)) | 1; /* OpenCV for float images */
    if (ksize > MAX_KSIZE) ksize = MAX_KSIZE;
    const int r = ksize / 2;
    SepCtx c;
    c.w = w;
    c.h = h;
    c.r = r;
    double kd[MAX_KSIZE], sum = 0;
    for (int i = 0; i < ksize; ++i) {
        const double x = i - (ksize - 1) * 0.5;
        kd[i] = exp(-x * x / (2.0 * sigma * sigma));
        sum += kd[i];
    }
    for (int i = 0; i <= r; ++i) c.k[i] = (float)(kd[r + i] / sum);
    c.src = src;
    c.dst = tmp;
    parallel_for(h, 16, gauss_rows, &c);
    c.src = tmp;
    c.dst = dst;
    parallel_for(h, 16, gauss_cols, &c);
}

/* ====================================================================== */
/* Box filter (sums in double, like OpenCV for float images)              */
/* ====================================================================== */

typedef struct {
    const float *src;
    float *dst;
    double *tmp;
    int w, h, kx, ky;
    double scale;
} BoxCtx;

static void box_rows(void *vc, int y0, int y1) {
    BoxCtx *c = vc;
    const int w = c->w, kx = c->kx, ax = kx / 2;
    double pad[w + kx + 1];
    for (int y = y0; y < y1; ++y) {
        const float *s = c->src + (size_t)y * w;
        double *d = c->tmp + (size_t)y * w;
        if (kx == 1) {
            for (int x = 0; x < w; ++x) d[x] = s[x];
            continue;
        }
        /* row with its reflected border: pad[i] = s[i - ax] */
        for (int i = 0; i < ax; ++i) pad[i] = s[border_reflect(i - ax, w)];
        for (int x = 0; x < w; ++x) pad[ax + x] = s[x];
        for (int i = ax + w; i < w + kx - 1; ++i) pad[i] = s[border_reflect(i - ax, w)];
        double sum = 0;
        for (int i = 0; i < kx; ++i) sum += pad[i];
        d[0] = sum;
        for (int x = 1; x < w; ++x) {
            sum += pad[x + kx - 1] - pad[x - 1];
            d[x] = sum;
        }
    }
}

static void box_cols(void *vc, int y0, int y1) {
    BoxCtx *c = vc;
    const int w = c->w, h = c->h, ky = c->ky, ay = ky / 2;
    const double scale = c->scale;
    double sum[w + 4];
    memset(sum, 0, sizeof(double) * (size_t)w);
    for (int i = -ay; i < ky - ay; ++i) {
        const double *r = c->tmp + (size_t)border_reflect(y0 + i, h) * w;
        int x = 0;
        for (; x + 4 <= w; x += 4) {
            v4d a, b;
            memcpy(&a, sum + x, sizeof a);
            memcpy(&b, r + x, sizeof b);
            a += b;
            memcpy(sum + x, &a, sizeof a);
        }
        for (; x < w; ++x) sum[x] += r[x];
    }
    for (int y = y0; y < y1; ++y) {
        float *d = c->dst + (size_t)y * w;
        for (int x = 0; x < w; ++x) d[x] = (float)(sum[x] * scale);
        if (y + 1 == y1) break;
        const double *add = c->tmp + (size_t)border_reflect(y + ky - ay, h) * w;
        const double *sub = c->tmp + (size_t)border_reflect(y - ay, h) * w;
        int x = 0;
        for (; x + 4 <= w; x += 4) {
            v4d a, b, e;
            memcpy(&a, sum + x, sizeof a);
            memcpy(&b, add + x, sizeof b);
            memcpy(&e, sub + x, sizeof e);
            a += b - e;
            memcpy(sum + x, &a, sizeof a);
        }
        for (; x < w; ++x) sum[x] += add[x] - sub[x];
    }
}

void box_filter(const float *src, float *dst, int w, int h, int kx, int ky, double *tmp) {
    BoxCtx c = {src, dst, tmp, w, h, kx, ky, 1.0 / ((double)kx * ky)};
    parallel_for(h, 16, box_rows, &c);
    parallel_for(h, 32, box_cols, &c);
}

/* ====================================================================== */
/* Sobel 3x3, BORDER_REFLECT_101                                           */
/* ====================================================================== */

typedef struct {
    const float *src;
    float *dst;
    int w, h, dx;
} SobelCtx;

static void sobel_rows(void *vc, int y0, int y1) {
    SobelCtx *c = vc;
    const int w = c->w, h = c->h;
    float pu[w + 2], pc[w + 2], pd[w + 2];
    for (int y = y0; y < y1; ++y) {
        const float *su = c->src + (size_t)border_reflect101(y - 1, h) * w;
        const float *sc = c->src + (size_t)y * w;
        const float *sd = c->src + (size_t)border_reflect101(y + 1, h) * w;
        memcpy(pu + 1, su, sizeof(float) * (size_t)w);
        memcpy(pc + 1, sc, sizeof(float) * (size_t)w);
        memcpy(pd + 1, sd, sizeof(float) * (size_t)w);
        const int l = border_reflect101(-1, w), r = border_reflect101(w, w);
        pu[0] = su[l]; pc[0] = sc[l]; pd[0] = sd[l];
        pu[w + 1] = su[r]; pc[w + 1] = sc[r]; pd[w + 1] = sd[r];
        float *d = c->dst + (size_t)y * w;
        int x = 0;
        if (c->dx) {
            for (; x + VW <= w; x += VW) {
                const v8f ru = v8f_load(pu + x + 2) - v8f_load(pu + x);
                const v8f rc = v8f_load(pc + x + 2) - v8f_load(pc + x);
                const v8f rd = v8f_load(pd + x + 2) - v8f_load(pd + x);
                v8f_store(d + x, ru + rc * 2.f + rd);
            }
            for (; x < w; ++x) d[x] = (pu[x + 2] - pu[x]) + (pc[x + 2] - pc[x]) * 2.f + (pd[x + 2] - pd[x]);
        } else {
            for (; x + VW <= w; x += VW) {
                const v8f ru = v8f_load(pu + x) + v8f_load(pu + x + 1) * 2.f + v8f_load(pu + x + 2);
                const v8f rd = v8f_load(pd + x) + v8f_load(pd + x + 1) * 2.f + v8f_load(pd + x + 2);
                v8f_store(d + x, rd - ru);
            }
            for (; x < w; ++x)
                d[x] = (pd[x] + pd[x + 1] * 2.f + pd[x + 2]) - (pu[x] + pu[x + 1] * 2.f + pu[x + 2]);
        }
    }
}

void sobel3(const float *src, float *dst, int w, int h, int dx, int dy) {
    (void)dy;
    SobelCtx c = {src, dst, w, h, dx};
    parallel_for(h, 16, sobel_rows, &c);
}

/* ====================================================================== */
/* morphology                                                             */
/* ====================================================================== */

typedef struct {
    const float *src;
    float *mx, *mn;
    int w, h;
} MM8Ctx;

static void mm8_rows(void *vc, int y0, int y1) {
    MM8Ctx *c = vc;
    const int w = c->w, h = c->h;
    float pu[w + 2], pc[w + 2], pd[w + 2];
    for (int y = y0; y < y1; ++y) {
        const float *su = c->src + (size_t)border_reflect(y - 1, h) * w;
        const float *sc = c->src + (size_t)y * w;
        const float *sd = c->src + (size_t)border_reflect(y + 1, h) * w;
        memcpy(pu + 1, su, sizeof(float) * (size_t)w);
        memcpy(pc + 1, sc, sizeof(float) * (size_t)w);
        memcpy(pd + 1, sd, sizeof(float) * (size_t)w);
        pu[0] = su[0]; pc[0] = sc[0]; pd[0] = sd[0];
        pu[w + 1] = su[w - 1]; pc[w + 1] = sc[w - 1]; pd[w + 1] = sd[w - 1];
        float *dmx = c->mx + (size_t)y * w, *dmn = c->mn + (size_t)y * w;
        int x = 0;
        for (; x + VW <= w; x += VW) {
            const v8f a = v8f_load(pu + x), b = v8f_load(pu + x + 1), e = v8f_load(pu + x + 2);
            const v8f f = v8f_load(pc + x), g = v8f_load(pc + x + 2);
            const v8f i = v8f_load(pd + x), j = v8f_load(pd + x + 1), k = v8f_load(pd + x + 2);
            v8f hi = v8f_max(v8f_max(v8f_max(a, b), v8f_max(e, f)), v8f_max(v8f_max(g, i), v8f_max(j, k)));
            v8f lo = v8f_min(v8f_min(v8f_min(a, b), v8f_min(e, f)), v8f_min(v8f_min(g, i), v8f_min(j, k)));
            v8f_store(dmx + x, hi);
            v8f_store(dmn + x, lo);
        }
        for (; x < w; ++x) {
            const float n8[8] = {pu[x], pu[x + 1], pu[x + 2], pc[x], pc[x + 2], pd[x], pd[x + 1], pd[x + 2]};
            float hi = n8[0], lo = n8[0];
            for (int q = 1; q < 8; ++q) { hi = maxf(hi, n8[q]); lo = minf(lo, n8[q]); }
            dmx[x] = hi;
            dmn[x] = lo;
        }
    }
}

void max_min_8(const float *src, float *mx, float *mn, int w, int h) {
    MM8Ctx c = {src, mx, mn, w, h};
    parallel_for(h, 16, mm8_rows, &c);
}

typedef struct {
    const float *src;
    float *dst;
    int w, h, r, is_max;
} RankCtx;

static void rank_rows(void *vc, int y0, int y1) {
    RankCtx *c = vc;
    const int w = c->w, r = c->r;
    for (int y = y0; y < y1; ++y) {
        const float *s = c->src + (size_t)y * w;
        float *d = c->dst + (size_t)y * w;
        for (int x = 0; x < w; ++x) {
            const int a = x - r < 0 ? 0 : x - r, b = x + r >= w ? w - 1 : x + r;
            float m = s[a];
            if (c->is_max) for (int i = a + 1; i <= b; ++i) m = maxf(m, s[i]);
            else for (int i = a + 1; i <= b; ++i) m = minf(m, s[i]);
            d[x] = m;
        }
    }
}

static void rank_cols(void *vc, int y0, int y1) {
    RankCtx *c = vc;
    const int w = c->w, h = c->h, r = c->r;
    for (int y = y0; y < y1; ++y) {
        const int a = y - r < 0 ? 0 : y - r, b = y + r >= h ? h - 1 : y + r;
        float *d = c->dst + (size_t)y * w;
        memcpy(d, c->src + (size_t)a * w, sizeof(float) * (size_t)w);
        for (int yy = a + 1; yy <= b; ++yy) {
            const float *s = c->src + (size_t)yy * w;
            int x = 0;
            for (; x + VW <= w; x += VW) {
                const v8f u = v8f_load(d + x), v = v8f_load(s + x);
                v8f_store(d + x, c->is_max ? v8f_max(u, v) : v8f_min(u, v));
            }
            for (; x < w; ++x) d[x] = c->is_max ? maxf(d[x], s[x]) : minf(d[x], s[x]);
        }
    }
}

static void rank_filter(const float *src, float *dst, int w, int h, int r, int is_max, float *tmp) {
    RankCtx c = {src, tmp, w, h, r, is_max};
    parallel_for(h, 16, rank_rows, &c);
    c.src = tmp;
    c.dst = dst;
    parallel_for(h, 16, rank_cols, &c);
}

void morph_open(const float *src, float *dst, int w, int h, int k, float *tmp) {
    const int r = k / 2;
    float *er = ALLOC(float, (size_t)w * h);
    rank_filter(src, er, w, h, r, 0, tmp);
    rank_filter(er, dst, w, h, r, 1, tmp);
    xfree(er);
}

/* ====================================================================== */
/* resampling                                                             */
/* ====================================================================== */

void resize_half_area_f32(const float *src, int w, int h, float *dst) {
    const int dw = w / 2, dh = h / 2;
    for (int y = 0; y < dh; ++y) {
        const float *a = src + (size_t)(2 * y) * w, *b = a + w;
        float *d = dst + (size_t)y * dw;
        for (int x = 0; x < dw; ++x) d[x] = ((a[2 * x] + a[2 * x + 1]) + (b[2 * x] + b[2 * x + 1])) * 0.25f;
    }
}

void resize_area_u8(const uint8_t *src, int w, int h, int f, uint8_t *dst) {
    const int dw = w / f, dh = h / f;
    const float scale = 1.f / (float)(f * f);
    for (int y = 0; y < dh; ++y) {
        uint8_t *d = dst + (size_t)y * dw;
        for (int x = 0; x < dw; ++x) {
            int s = 0;
            for (int i = 0; i < f; ++i) {
                const uint8_t *r = src + (size_t)(y * f + i) * w + x * f;
                for (int j = 0; j < f; ++j) s += r[j];
            }
            d[x] = f == 2 ? (uint8_t)((s + 2) >> 2) : (uint8_t)clampi((int)lrintf((float)s * scale), 0, 255);
        }
    }
}

static void linear_coeffs(int dst_n, int src_n, int *ofs, float *a0) {
    const double scale = (double)src_n / dst_n;
    for (int d = 0; d < dst_n; ++d) {
        float f = (float)((d + 0.5) * scale - 0.5);
        int s = (int)floorf(f);
        f -= (float)s;
        if (s < 0) { f = 0; s = 0; }
        if (s >= src_n - 1) { f = 0; s = src_n - 1; }
        ofs[d] = s;
        a0[d] = f; /* weight of s+1 */
    }
}

typedef struct {
    const float *src;
    float *dst;
    int sw, sh, dw, dh;
    const int *xo, *yo;
    const float *xa, *ya;
} RLCtx;

static void rl_rows(void *vc, int y0, int y1) {
    RLCtx *c = vc;
    float h0[c->dw], h1[c->dw];
    for (int dy = y0; dy < y1; ++dy) {
        const int sy = c->yo[dy];
        const int sy1 = sy + 1 < c->sh ? sy + 1 : c->sh - 1;
        const float fy = c->ya[dy];
        const float *r0 = c->src + (size_t)sy * c->sw, *r1 = c->src + (size_t)sy1 * c->sw;
        for (int dx = 0; dx < c->dw; ++dx) {
            const int sx = c->xo[dx];
            const float fx = c->xa[dx];
            if (sx + 1 < c->sw) {
                h0[dx] = r0[sx] * (1.f - fx) + r0[sx + 1] * fx;
                h1[dx] = r1[sx] * (1.f - fx) + r1[sx + 1] * fx;
            } else {
                h0[dx] = r0[sx] * (1.f - fx);
                h1[dx] = r1[sx] * (1.f - fx);
            }
        }
        float *d = c->dst + (size_t)dy * c->dw;
        const float b0 = 1.f - fy, b1 = fy;
        for (int dx = 0; dx < c->dw; ++dx) d[dx] = h0[dx] * b0 + h1[dx] * b1;
    }
}

void resize_linear_f32(const float *src, int sw, int sh, float *dst, int dw, int dh) {
    int xo[dw], yo[dh];
    float xa[dw], ya[dh];
    linear_coeffs(dw, sw, xo, xa);
    linear_coeffs(dh, sh, yo, ya);
    RLCtx c = {src, dst, sw, sh, dw, dh, xo, yo, xa, ya};
    parallel_for(dh, 16, rl_rows, &c);
}

/* remap: maps are rounded to 1/32 pixel and weights come from a table,
 * exactly as cv::remap does for float maps */
#define TAB 32
static float lin_tab[TAB * TAB][4];
static float cub_tab[TAB * TAB][16];
static int tabs_ready = 0;

static void init_tabs(void) {
    if (tabs_ready) return;
    float l1[TAB][2], c1[TAB][4];
    const float A = -0.75f;
    for (int i = 0; i < TAB; ++i) {
        const float x = (float)i * (1.f / TAB);
        l1[i][0] = 1.f - x;
        l1[i][1] = x;
        c1[i][0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
        c1[i][1] = ((A + 2) * x - (A + 3)) * x * x + 1;
        c1[i][2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
        c1[i][3] = 1.f - c1[i][0] - c1[i][1] - c1[i][2];
    }
    for (int iy = 0; iy < TAB; ++iy)
        for (int ix = 0; ix < TAB; ++ix) {
            for (int k = 0; k < 2; ++k)
                for (int l = 0; l < 2; ++l) lin_tab[iy * TAB + ix][k * 2 + l] = l1[iy][k] * l1[ix][l];
            for (int k = 0; k < 4; ++k)
                for (int l = 0; l < 4; ++l) cub_tab[iy * TAB + ix][k * 4 + l] = c1[iy][k] * c1[ix][l];
        }
    tabs_ready = 1;
}

typedef struct {
    const float *src, *mx, *my;
    float *dst;
    int sw, sh, dw;
} RemapCtx;

static void remap_lin_rows(void *vc, int y0, int y1) {
    RemapCtx *c = vc;
    const int sw = c->sw, sh = c->sh;
    for (int y = y0; y < y1; ++y) {
        const float *mx = c->mx + (size_t)y * c->dw, *my = c->my + (size_t)y * c->dw;
        float *d = c->dst + (size_t)y * c->dw;
        for (int x = 0; x < c->dw; ++x) {
            const int X = (int)lrintf(mx[x] * TAB), Y = (int)lrintf(my[x] * TAB);
            const int sx = X >> 5, sy = Y >> 5;
            const float *w = lin_tab[(Y & (TAB - 1)) * TAB + (X & (TAB - 1))];
            if ((unsigned)sx < (unsigned)(sw - 1) && (unsigned)sy < (unsigned)(sh - 1)) {
                const float *s = c->src + (size_t)sy * sw + sx;
                d[x] = s[0] * w[0] + s[1] * w[1] + s[sw] * w[2] + s[sw + 1] * w[3];
            } else {
                const int x0 = clampi(sx, 0, sw - 1), x1 = clampi(sx + 1, 0, sw - 1);
                const int ya = clampi(sy, 0, sh - 1), yb = clampi(sy + 1, 0, sh - 1);
                const float *r0 = c->src + (size_t)ya * sw, *r1 = c->src + (size_t)yb * sw;
                d[x] = r0[x0] * w[0] + r0[x1] * w[1] + r1[x0] * w[2] + r1[x1] * w[3];
            }
        }
    }
}

static void remap_cub_rows(void *vc, int y0, int y1) {
    RemapCtx *c = vc;
    const int sw = c->sw, sh = c->sh;
    for (int y = y0; y < y1; ++y) {
        const float *mx = c->mx + (size_t)y * c->dw, *my = c->my + (size_t)y * c->dw;
        float *d = c->dst + (size_t)y * c->dw;
        for (int x = 0; x < c->dw; ++x) {
            const int X = (int)lrintf(mx[x] * TAB), Y = (int)lrintf(my[x] * TAB);
            const int sx = (X >> 5) - 1, sy = (Y >> 5) - 1;
            const float *w = cub_tab[(Y & (TAB - 1)) * TAB + (X & (TAB - 1))];
            if ((unsigned)sx < (unsigned)(sw - 3) && (unsigned)sy < (unsigned)(sh - 3)) {
                const float *s = c->src + (size_t)sy * sw + sx;
                float sum = s[0] * w[0] + s[1] * w[1] + s[2] * w[2] + s[3] * w[3];
                s += sw;
                sum += s[0] * w[4] + s[1] * w[5] + s[2] * w[6] + s[3] * w[7];
                s += sw;
                sum += s[0] * w[8] + s[1] * w[9] + s[2] * w[10] + s[3] * w[11];
                s += sw;
                sum += s[0] * w[12] + s[1] * w[13] + s[2] * w[14] + s[3] * w[15];
                d[x] = sum;
            } else {
                int xs[4], ys[4];
                for (int i = 0; i < 4; ++i) {
                    xs[i] = clampi(sx + i, 0, sw - 1);
                    ys[i] = clampi(sy + i, 0, sh - 1);
                }
                float sum = 0.f;
                for (int i = 0; i < 4; ++i) {
                    const float *r = c->src + (size_t)ys[i] * sw;
                    for (int j = 0; j < 4; ++j) sum += r[xs[j]] * w[i * 4 + j];
                }
                d[x] = sum;
            }
        }
    }
}

void remap_linear_f32(const float *src, int sw, int sh, const float *mapx, const float *mapy, float *dst,
                      int dw, int dh) {
    init_tabs();
    RemapCtx c = {src, mapx, mapy, dst, sw, sh, dw};
    parallel_for(dh, 16, remap_lin_rows, &c);
}

void remap_cubic_f32(const float *src, int sw, int sh, const float *mapx, const float *mapy, float *dst, int dw,
                     int dh) {
    init_tabs();
    RemapCtx c = {src, mapx, mapy, dst, sw, sh, dw};
    parallel_for(dh, 16, remap_cub_rows, &c);
}
