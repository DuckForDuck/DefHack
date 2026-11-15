/* Pipeline of preset michael-features in C (steps 0-9).
 * Same maths as the C++ version (stages.cpp); per-pixel loops run in
 * parallel over rows and are written so the compiler vectorises them.
 */
#include "stages.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "blue_noise.h"
#include "common.h"
#include "dis.h"
#include "filters.h"
#include "parallel.h"

const char *const kStageNames[ST_COUNT] = {
    "Decode", "ShutterGuard", "BadPixels", "ResidualNUC", "Destripe", "TemporalDenoise", "SpatialDenoise",
    "BaseDetailSplit", "ToneMap", "CompactHotBoost", "MultiScaleDetail", "ShutterHold", "Quantise"};

#define NSCR 10
#define MAX_HISTORY 64

struct Pipeline {
    Params p;
    int w, h;
    size_t n;

    /* travels with the frame (FrameData) */
    float *img;       /* DN (steps 0-5), 0..1 (steps 6-7) */
    float *noise_map; /* remaining noise std per pixel, DN (step 3) */
    float *base, *detail, *linear, *slope;
    uint8_t *bad; /* 1 = bad pixel (step 1) */
    double noise_sigma, lin_slope;
    int frozen;

    /* scratch */
    float *s[NSCR];
    double *dtmp;
    float *lap;
    uint8_t *u8a, *u8b;

    /* ShutterGuard */
    double stds[MAX_HISTORY];
    int nstd, freeze;

    /* BadPixels */
    int bp_n, bp_has_prev, bp_has_rate, bp_has_change, bp_has_learned;
    float *bp_prev, *bp_rate, *bp_change;
    uint8_t *bp_learned;

    /* ResidualNUC */
    float *nuc_offset, *nuc_prev;
    int nuc_has_prev;

    /* Destripe */
    float *ds_col, *ds_row;
    int ds_has;
    double *ds_sum;
    int *ds_cnt;

    /* TemporalDenoise */
    Dis *dis;
    float *tnr_prev, *tnr_n, *tnr_v, *flowx, *flowy, *mapx, *mapy;
    uint8_t *cur8, *prev8;
    int tnr_has_prev;

    /* ToneMap */
    int tm_first;
    double lo, hi, hi_hold, mx, mn;
    int hold_max, hold_min;
    double *curve;
    int curve_cap;

    /* ShutterHold */
    float *last;
    int has_last;

    /* Quantise */
    float *thr;

    /* values handed to the parallel loops */
    const uint16_t *raw;
    int any_bad;
    float k_f[4];
    int k_i[4];
    double part1[64], part2[64]; /* per-chunk partial sums */
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static inline float ramp(float x, float lo, float hi) { return clampf((hi - x) / (hi - lo), 0.f, 1.f); }

/* y + add, approaching 0 and 1 smoothly (limit_add in multiscale.py) */
typedef struct {
    float *y;
    const float *add;
    int w;
} LimitCtx;
static void limit_rows(void *vc, int y0, int y1) {
    LimitCtx *c = vc;
    const float eps = 1e-6f;
    float *restrict y = c->y;
    const float *restrict a = c->add;
    IVDEP
    for (size_t i = (size_t)y0 * c->w; i < (size_t)y1 * c->w; ++i) {
        const float v = y[i], ad = a[i];
        y[i] = ad >= 0 ? v + ad * (1.f - v) / (1.f - v + ad + eps) : v + ad * v / (v - ad + eps);
    }
}
static void limit_add(float *y, const float *add, int w, int h) {
    LimitCtx c = {y, add, w};
    parallel_for(h, 16, limit_rows, &c);
}

/* Elementwise loops over the whole frame, split into NCHUNK fixed chunks that
 * run in parallel. Reductions add per-chunk partial sums in chunk order, so the
 * result does not depend on the number of threads. */
#define NCHUNK 64
typedef void (*chunk_fn)(Pipeline *pl, size_t a, size_t b, int chunk);
typedef struct {
    Pipeline *pl;
    chunk_fn fn;
} ChunkCtx;
static void chunk_range(void *vc, int c0, int c1) {
    ChunkCtx *c = vc;
    const size_t n = c->pl->n;
    for (int k = c0; k < c1; ++k) c->fn(c->pl, n * (size_t)k / NCHUNK, n * (size_t)(k + 1) / NCHUNK, k);
}
static void pmap(Pipeline *pl, chunk_fn fn) {
    ChunkCtx c = {pl, fn};
    parallel_for(NCHUNK, 1, chunk_range, &c);
}
static double sum_parts(const double *p) {
    double s = 0;
    for (int k = 0; k < NCHUNK; ++k) s += p[k];
    return s;
}

/* ---------------------------------------------------------------- setup */

Pipeline *pipeline_create(const Params *p, int w, int h) {
    Pipeline *pl = ALLOC(Pipeline, 1);
    pl->p = *p;
    pl->w = w;
    pl->h = h;
    pl->n = (size_t)w * h;
    const size_t n = pl->n + VW;
    pl->img = ALLOC(float, n);
    pl->noise_map = ALLOC(float, n);
    pl->base = ALLOC(float, n);
    pl->detail = ALLOC(float, n);
    pl->linear = ALLOC(float, n);
    pl->slope = ALLOC(float, n);
    pl->bad = ALLOC(uint8_t, n);
    for (int i = 0; i < NSCR; ++i) pl->s[i] = ALLOC(float, n);
    pl->dtmp = ALLOC(double, n);
    pl->lap = ALLOC(float, n);
    pl->u8a = ALLOC(uint8_t, n);
    pl->u8b = ALLOC(uint8_t, n);
    pl->bp_prev = ALLOC(float, n);
    pl->bp_rate = ALLOC(float, n);
    pl->bp_change = ALLOC(float, n);
    pl->bp_learned = ALLOC(uint8_t, n);
    pl->nuc_offset = ALLOC(float, n);
    pl->nuc_prev = ALLOC(float, n);
    pl->ds_col = ALLOC(float, w);
    pl->ds_row = ALLOC(float, h);
    pl->ds_sum = ALLOC(double, w > h ? w : h);
    pl->ds_cnt = ALLOC(int, w > h ? w : h);
    pl->dis = dis_create(p->tnr_flow_preset, w, h);
    pl->tnr_prev = ALLOC(float, n);
    pl->tnr_n = ALLOC(float, n);
    pl->tnr_v = ALLOC(float, n);
    pl->flowx = ALLOC(float, n);
    pl->flowy = ALLOC(float, n);
    pl->mapx = ALLOC(float, n);
    pl->mapy = ALLOC(float, n);
    pl->cur8 = ALLOC(uint8_t, n);
    pl->prev8 = ALLOC(uint8_t, n);
    pl->tm_first = 1;
    pl->curve_cap = 1 << 14;
    pl->curve = ALLOC(double, pl->curve_cap);
    pl->last = ALLOC(float, n);
    pl->thr = ALLOC(float, n);
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c)
            pl->thr[(size_t)r * w + c] =
                (kBlueNoiseRanks[(r % BLUE_NOISE_SIZE) * BLUE_NOISE_SIZE + c % BLUE_NOISE_SIZE] + 0.5f) /
                (float)(BLUE_NOISE_SIZE * BLUE_NOISE_SIZE);
    return pl;
}

void pipeline_free(Pipeline *pl) {
    if (!pl) return;
    void *bufs[] = {pl->img, pl->noise_map, pl->base, pl->detail, pl->linear, pl->slope, pl->bad, pl->dtmp,
                    pl->lap, pl->u8a, pl->u8b, pl->bp_prev, pl->bp_rate, pl->bp_change, pl->bp_learned,
                    pl->nuc_offset, pl->nuc_prev, pl->ds_col, pl->ds_row, pl->ds_sum, pl->ds_cnt, pl->tnr_prev,
                    pl->tnr_n, pl->tnr_v, pl->flowx, pl->flowy, pl->mapx, pl->mapy, pl->cur8, pl->prev8,
                    pl->curve, pl->last, pl->thr};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; ++i) xfree(bufs[i]);
    for (int i = 0; i < NSCR; ++i) xfree(pl->s[i]);
    dis_free(pl->dis);
    xfree(pl);
}

/* ---------------------------------------------------------------- step 0 */
static void decode_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    const uint16_t mask = (uint16_t)((1u << pl->p.bit_depth) - 1u);
    const uint16_t *restrict raw = pl->raw;
    float *restrict o = pl->img;
    for (size_t i = a; i < b; ++i) o[i] = (float)(raw[i] & mask);
}
static void decode(Pipeline *pl, const uint16_t *raw) {
    pl->raw = raw;
    pmap(pl, decode_chunk);
}

/* ---------------------------------------------------------------- step 0b */
static void shutter_guard(Pipeline *pl) {
    double s1 = 0, s2 = 0;
    size_t cnt = 0;
    for (int y = 0; y < pl->h; y += 4) {
        const float *r = pl->img + (size_t)y * pl->w;
        for (int x = 0; x < pl->w; x += 4) { s1 += r[x]; s2 += (double)r[x] * r[x]; ++cnt; }
    }
    const double mean = s1 / (double)cnt;
    const double s = sqrt(maxd(s2 / (double)cnt - mean * mean, 0.0));
    int frozen = 0;
    if (pl->nstd >= 4) {
        float h[MAX_HISTORY];
        for (int i = 0; i < pl->nstd; ++i) h[i] = (float)pl->stds[i];
        if (s < pl->p.shutter_ratio * median(h, pl->nstd)) {
            ++pl->freeze;
            if (pl->freeze <= pl->p.shutter_max_freeze) frozen = 1;
            else pl->nstd = 0; /* the scene really changed: accept it */
        }
    }
    if (!frozen) {
        pl->freeze = 0;
        int hist = pl->p.shutter_history < MAX_HISTORY ? pl->p.shutter_history : MAX_HISTORY;
        if (pl->nstd == hist) {
            memmove(pl->stds, pl->stds + 1, sizeof(double) * (size_t)(hist - 1));
            pl->nstd--;
        }
        pl->stds[pl->nstd++] = s;
    }
    pl->frozen = frozen;
}

/* ---------------------------------------------------------------- step 1 */
typedef struct {
    Pipeline *pl;
    const float *mx, *mn;
    const uint8_t *perm;
    uint8_t *cand, *imp;
    float inv, k, k_rep;
} BPCtx;

static void bp_test_rows(void *vc, int y0, int y1) {
    BPCtx *c = vc;
    const int w = c->pl->w;
    const float *v = c->pl->img;
    IVDEP
    for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) {
        const float e = maxf(v[i] - c->mx[i], c->mn[i] - v[i]) * c->inv;
        const int pm = c->perm ? c->perm[i] : 0;
        c->cand[i] = (e > c->k && !pm) ? 1 : 0;
        c->imp[i] = (e > c->k_rep && !pm) ? 1 : 0;
    }
}

/* x += a * (now - x), evaluated as OpenCV does: x + (now*a + x*(-a)) */
static void bp_ema_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    const float al = pl->k_f[0];
    const int has_rate = pl->bp_has_rate, has_prev = pl->bp_has_prev, has_change = pl->bp_has_change;
    const uint8_t *restrict cand = pl->u8a;
    const float *restrict img = pl->img;
    float *restrict rate = pl->bp_rate, *restrict change = pl->bp_change, *restrict prev = pl->bp_prev;
    for (size_t i = a; i < b; ++i) {
        const float cf = cand[i] ? 1.f : 0.f;
        rate[i] = has_rate ? rate[i] + (cf * al + rate[i] * (-al)) : cf;
        if (has_prev) {
            const float ch = fabsf(img[i] - prev[i]);
            change[i] = has_change ? change[i] + (ch * al + change[i] * (-al)) : ch;
        }
        prev[i] = img[i];
    }
}

static void bp_learned_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    const float pr = pl->k_f[0], st = pl->k_f[1];
    const int use_stuck = pl->k_i[0];
    for (size_t i = a; i < b; ++i) {
        const int persistent = pl->bp_rate[i] > pr;
        const int stuck = use_stuck ? pl->bp_change[i] < st : 0;
        pl->bp_learned[i] = (uint8_t)(persistent | stuck);
    }
}

static void bp_update_learned(Pipeline *pl) {
    pl->k_f[0] = (float)pl->p.bp_ema;
    pmap(pl, bp_ema_chunk);
    if (pl->bp_has_prev) pl->bp_has_change = 1;
    pl->bp_has_rate = 1;
    pl->bp_has_prev = 1;
    if (pl->bp_n < pl->p.bp_warmup || !pl->bp_has_change) return;
    const double scene_change = median(pl->bp_change, (int)pl->n);
    pl->k_f[0] = (float)pl->p.bp_persist_ratio;
    pl->k_f[1] = (float)(pl->p.bp_stuck_ratio * scene_change);
    pl->k_i[0] = scene_change > 0;
    pmap(pl, bp_learned_chunk);
    pl->bp_has_learned = 1;
}

static void bad_pixels(Pipeline *pl) {
    const int w = pl->w, h = pl->h;
    const size_t n = pl->n;
    /* permanent = learned mask of the previous frame */
    uint8_t *perm = NULL;
    int any_perm = 0;
    if (pl->bp_has_learned) {
        perm = pl->u8b;
        memcpy(perm, pl->bp_learned, n);
        for (size_t i = 0; i < n && !any_perm; ++i) any_perm |= perm[i];
    }
    const double sigma = estimate_noise_sigma(pl->img, w, h, any_perm ? perm : NULL, pl->lap);
    float *mx = pl->s[0], *mn = pl->s[1];
    max_min_8(pl->img, mx, mn, w, h);
    uint8_t *cand = pl->u8a, *imp = pl->bad;
    BPCtx c = {pl, mx, mn, perm, cand, imp, 1.f / (float)maxd(sigma, 0.001), (float)pl->p.bp_k,
               (float)pl->p.bp_k_replace};
    parallel_for(h, 16, bp_test_rows, &c);
    bp_update_learned(pl);
    ++pl->bp_n;

    /* mask = permanent | impulses (kept in pl->bad) */
    uint8_t *mask = pl->bad;
    if (perm)
        for (size_t i = 0; i < n; ++i) mask[i] |= perm[i];
    float *out = pl->s[2];
    memcpy(out, pl->img, sizeof(float) * n);
    int left_n = 0;
    int *left = NULL;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (!mask[(size_t)y * w + x]) continue;
            int done = 0;
            for (int r = 1; r <= 2 && !done; ++r) {
                float vals[24];
                int nv = 0;
                for (int dy = -r; dy <= r; ++dy)
                    for (int dx = -r; dx <= r; ++dx) {
                        if (!dy && !dx) continue;
                        const int yy = border_reflect101(y + dy, h), xx = border_reflect101(x + dx, w);
                        if (!mask[(size_t)yy * w + xx]) vals[nv++] = pl->img[(size_t)yy * w + xx];
                    }
                if (nv) {
                    out[(size_t)y * w + x] = (float)median(vals, nv);
                    done = 1;
                }
            }
            if (!done) {
                if (!left) left = ALLOC(int, n);
                left[left_n++] = y * w + x;
            }
        }
    if (left_n) {
        float *good = pl->s[3];
        int ng = 0;
        for (size_t i = 0; i < n; ++i)
            if (!mask[i]) good[ng++] = pl->img[i];
        const float fill = ng ? (float)median(good, ng) : 0.f;
        for (int i = 0; i < left_n; ++i) out[left[i]] = fill;
        xfree(left);
    }
    memcpy(pl->img, out, sizeof(float) * n);
    pl->noise_sigma = sigma;
    pl->any_bad = 0;
    for (size_t i = 0; i < n && !pl->any_bad; ++i) pl->any_bad |= mask[i];
}

/* ---------------------------------------------------------------- step 2a */
typedef struct {
    Pipeline *pl;
    const float *y, *sm, *gx, *gy;
    float c, scale, alpha;
} NucCtx;

static void nuc_rows(void *vc, int y0, int y1) {
    NucCtx *c = vc;
    const int w = c->pl->w;
    float *restrict o = c->pl->nuc_offset;
    const uint8_t *restrict bad = c->pl->bad;
    IVDEP
    for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) {
        const float e = clampf(c->y[i] - c->sm[i], -c->c, c->c);
        const float g1 = c->gx[i] / 8.f, g2 = c->gy[i] / 8.f;
        const float g = sqrtf(g1 * g1 + g2 * g2) / c->scale;
        float wt = 1.f / (1.f + g * g);
        if (bad[i]) wt = 0.f;
        o[i] += c->alpha * wt * e;
    }
}

/* x = input, y = x - offset, sum |x - prev|, prev = x */
static void nuc_pre_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    float *restrict x = pl->s[0], *restrict y = pl->img, *restrict prev = pl->nuc_prev;
    const float *restrict off = pl->nuc_offset;
    const int has_prev = pl->nuc_has_prev;
    double sum = 0;
    for (size_t i = a; i < b; ++i) {
        const float v = y[i];
        x[i] = v;
        y[i] = v - off[i];
        if (has_prev) sum += fabsf(v - prev[i]);
        prev[i] = v;
    }
    pl->part1[k] = sum;
}
static void nuc_sum_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    double s = 0;
    for (size_t i = a; i < b; ++i) s += pl->nuc_offset[i];
    pl->part1[k] = s;
}
static void nuc_post_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    const float mf = pl->k_f[0];
    float *restrict off = pl->nuc_offset, *restrict y = pl->img;
    const float *restrict x = pl->s[0];
    for (size_t i = a; i < b; ++i) {
        off[i] -= mf;
        y[i] = x[i] - off[i];
    }
}

static void residual_nuc(Pipeline *pl) {
    const int w = pl->w, h = pl->h;
    const size_t n = pl->n;
    const int had_prev = pl->nuc_has_prev;
    pmap(pl, nuc_pre_chunk); /* also sets prev = x for the next frame */
    pl->nuc_has_prev = 1;
    float *y = pl->img;
    const double sigma = pl->noise_sigma > 0 ? pl->noise_sigma : estimate_noise_sigma(y, w, h, NULL, pl->lap);
    if (!had_prev) return;
    const double motion = (sum_parts(pl->part1) / (double)n) / (1.128 * sigma);
    if (motion <= pl->p.nuc_motion_k) return;
    float *sm = pl->s[1], *gx = pl->s[2], *gy = pl->s[3];
    gaussian_blur(y, sm, w, h, pl->p.nuc_blur_sigma, pl->s[4]);
    sobel3(sm, gx, w, h, 1, 0);
    sobel3(sm, gy, w, h, 0, 1);
    NucCtx c = {pl, y, sm, gx, gy, (float)(pl->p.nuc_clip_k * sigma),
                (float)maxd(pl->p.nuc_edge_k * sigma / pl->p.nuc_blur_sigma, 0.001), (float)pl->p.nuc_alpha};
    parallel_for(h, 16, nuc_rows, &c);
    pmap(pl, nuc_sum_chunk);
    pl->k_f[0] = (float)(sum_parts(pl->part1) / (double)n);
    pmap(pl, nuc_post_chunk);
}

/* ---------------------------------------------------------------- step 2b */
/* axis 0: one offset per column; axis 1: one per row */
static void destripe_offsets(Pipeline *pl, const float *x, int axis, double sigma, float *out) {
    const int w = pl->w, h = pl->h;
    float *lp = pl->s[1], *sm = pl->s[2], *grad = pl->s[3];
    if (axis == 0) box_filter(x, lp, w, h, pl->p.ds_width, 1, pl->dtmp);
    else box_filter(x, lp, w, h, 1, pl->p.ds_width, pl->dtmp);
    gaussian_blur(x, sm, w, h, pl->p.ds_edge_sigma, pl->s[4]);
    sobel3(sm, grad, w, h, axis == 0, axis != 0);
    const float c = (float)(pl->p.ds_clip_k * sigma);
    const float lim = (float)(pl->p.ds_edge_k * sigma);
    const int len = axis == 0 ? w : h;
    double *sum = pl->ds_sum;
    int *cnt = pl->ds_cnt;
    for (int k = 0; k < len; ++k) { sum[k] = 0; cnt[k] = 0; }
    for (int y = 0; y < h; ++y) {
        const float *xr = x + (size_t)y * w, *lr = lp + (size_t)y * w, *gr = grad + (size_t)y * w;
        if (axis == 0) {
            for (int i = 0; i < w; ++i)
                if (fabsf(gr[i]) / 8.f < lim) { sum[i] += clampf(xr[i] - lr[i], -c, c); cnt[i]++; }
        } else {
            double s = 0;
            int ct = 0;
            for (int i = 0; i < w; ++i)
                if (fabsf(gr[i]) / 8.f < lim) { s += clampf(xr[i] - lr[i], -c, c); ct++; }
            sum[y] = s;
            cnt[y] = ct;
        }
    }
    const double need = pl->p.ds_min_valid * (axis == 0 ? h : w);
    double mean = 0;
    for (int k = 0; k < len; ++k) {
        sum[k] = cnt[k] >= need ? sum[k] / (cnt[k] > 1 ? cnt[k] : 1) : 0.0;
        mean += sum[k];
    }
    mean /= len;
    for (int k = 0; k < len; ++k) out[k] = (float)(sum[k] - mean);
}

static void ds_sub_rows(void *vc, int y0, int y1) {
    Pipeline *pl = vc;
    const int w = pl->w;
    for (int y = y0; y < y1; ++y) {
        float *restrict r = pl->img + (size_t)y * w;
        if (pl->k_i[0] == 0) {
            const float *restrict col = pl->ds_col;
            for (int i = 0; i < w; ++i) r[i] -= col[i];
        } else {
            const float o = pl->ds_row[y];
            for (int i = 0; i < w; ++i) r[i] -= o;
        }
    }
}

static void destripe(Pipeline *pl) {
    const int w = pl->w, h = pl->h;
    float *x = pl->img;
    const double sigma = pl->noise_sigma > 0 ? pl->noise_sigma : estimate_noise_sigma(x, w, h, NULL, pl->lap);
    const float a = (float)pl->p.ds_temporal_alpha;
    float now_c[w], now_r[h];
    destripe_offsets(pl, x, 0, sigma, now_c);
    if (a <= 0 || !pl->ds_has) memcpy(pl->ds_col, now_c, sizeof(float) * (size_t)w);
    else for (int k = 0; k < w; ++k) pl->ds_col[k] += a * (now_c[k] - pl->ds_col[k]);
    pl->k_i[0] = 0;
    parallel_for(h, 16, ds_sub_rows, pl);
    destripe_offsets(pl, x, 1, sigma, now_r);
    if (a <= 0 || !pl->ds_has) memcpy(pl->ds_row, now_r, sizeof(float) * (size_t)h);
    else for (int k = 0; k < h; ++k) pl->ds_row[k] += a * (now_r[k] - pl->ds_row[k]);
    pl->ds_has = 1;
    pl->k_i[0] = 1;
    parallel_for(h, 16, ds_sub_rows, pl);
    pl->noise_sigma = estimate_noise_sigma(x, w, h, pl->any_bad ? pl->bad : NULL, pl->lap);
}

/* ---------------------------------------------------------------- step 3a */
typedef struct {
    Pipeline *pl;
    const float *src;
    uint8_t *dst;
    double lo, scale;
} To8Ctx;
static void to8_rows(void *vc, int y0, int y1) {
    To8Ctx *c = vc;
    const int w = c->pl->w;
    for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) {
        double t = ((double)c->src[i] - c->lo) * c->scale;
        t = t < 0 ? 0 : (t > 255 ? 255 : t);
        c->dst[i] = (uint8_t)t; /* truncation, as numpy astype(uint8) */
    }
}

typedef struct {
    Pipeline *pl;
    const float *x, *p, *nw, *vw, *local;
    float *d, *s2, *dd;
    float s2base, maxf;
} TnrCtx;

static void tnr_pre_rows(void *vc, int y0, int y1) {
    TnrCtx *c = vc;
    Pipeline *pl = c->pl;
    const int w = pl->w, h = pl->h;
    float *vw = (float *)c->vw, *nw = (float *)c->nw;
    for (int y = y0; y < y1; ++y)
        for (int i = 0; i < w; ++i) {
            const size_t k = (size_t)y * w + i;
            const float a = pl->mapx[k], b = pl->mapy[k];
            const int inside = a >= 0 && a <= w - 1 && b >= 0 && b <= h - 1;
            if (!inside) { nw[k] = 0.f; vw[k] = 1.f; }
            c->d[k] = c->x[k] - c->p[k];
            c->s2[k] = c->s2base * (1.f + vw[k]);
            c->dd[k] = c->d[k] * c->d[k];
        }
}

static void tnr_mix_rows(void *vc, int y0, int y1) {
    TnrCtx *c = vc;
    Pipeline *pl = c->pl;
    const int w = pl->w, h = pl->h;
    const float llo = (float)pl->p.tnr_local_lo, lhi = (float)pl->p.tnr_local_hi;
    const float plo = (float)pl->p.tnr_pixel_lo, phi = (float)pl->p.tnr_pixel_hi;
    float *out = pl->tnr_prev, *nn = pl->tnr_n, *vv = pl->tnr_v;
    for (int y = y0; y < y1; ++y)
        for (int i = 0; i < w; ++i) {
            const size_t k = (size_t)y * w + i;
            const float a = pl->mapx[k], b = pl->mapy[k];
            const int inside = a >= 0 && a <= w - 1 && b >= 0 && b <= h - 1;
            float alpha = ramp(c->local[k] / c->s2[k], llo, lhi) * ramp(fabsf(c->d[k]) / sqrtf(c->s2[k]), plo, phi);
            if (!inside) alpha = 0.f;
            const float nf = minf(alpha * c->nw[k] + 1.f, c->maxf);
            const float gain = 1.f / nf;
            out[k] = c->p[k] + gain * c->d[k];
            nn[k] = nf;
            vv[k] = (1.f - gain) * (1.f - gain) * c->vw[k] + gain * gain;
        }
}

static void tnr_map_rows(void *vc, int y0, int y1) {
    Pipeline *pl = vc;
    const int w = pl->w;
    for (int y = y0; y < y1; ++y)
        for (int i = 0; i < w; ++i) {
            const size_t k = (size_t)y * w + i;
            pl->mapx[k] = (float)i + pl->flowx[k];
            pl->mapy[k] = (float)y + pl->flowy[k];
        }
}
static void tnr_out_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    const float sg = pl->k_f[0];
    for (size_t i = a; i < b; ++i) {
        pl->noise_map[i] = sg * sqrtf(pl->tnr_v[i]);
        pl->img[i] = pl->tnr_prev[i];
    }
}

static void temporal_denoise(Pipeline *pl) {
    const int w = pl->w, h = pl->h;
    const size_t n = pl->n;
    const float *x = pl->img;
    const double sigma = pl->noise_sigma > 0 ? pl->noise_sigma : estimate_noise_sigma(pl->img, w, h, NULL, pl->lap);
    if (!pl->tnr_has_prev) {
        memcpy(pl->tnr_prev, x, sizeof(float) * n);
        for (size_t i = 0; i < n; ++i) { pl->tnr_n[i] = 1.f; pl->tnr_v[i] = 1.f; }
        pl->tnr_has_prev = 1;
    } else {
        /* optical flow from the current frame to the previous result (8 bit) */
        float *cur_s = pl->s[0];
        gaussian_blur(x, cur_s, w, h, pl->p.tnr_flow_smooth, pl->s[1]);
        float *sub = pl->s[1];
        int ns = 0;
        for (int y = 0; y < h; y += 4)
            for (int i = 0; i < w; i += 4) sub[ns++] = cur_s[(size_t)y * w + i];
        const double lo = percentile(sub, ns, 0.5), hi = percentile(sub, ns, 99.5);
        const double scale = 255.0 / maxd(hi - lo, 0.001);
        To8Ctx t8 = {pl, cur_s, pl->cur8, lo, scale};
        parallel_for(h, 16, to8_rows, &t8);
        t8.src = pl->tnr_prev;
        t8.dst = pl->prev8;
        parallel_for(h, 16, to8_rows, &t8);
        dis_calc(pl->dis, pl->cur8, pl->prev8, pl->flowx, pl->flowy);
        parallel_for(h, 16, tnr_map_rows, pl);
        float *p = pl->s[2], *nw = pl->s[3], *vw = pl->s[4];
        remap_cubic_f32(pl->tnr_prev, w, h, pl->mapx, pl->mapy, p, w, h);
        remap_linear_f32(pl->tnr_n, w, h, pl->mapx, pl->mapy, nw, w, h);
        remap_linear_f32(pl->tnr_v, w, h, pl->mapx, pl->mapy, vw, w, h);
        float *d = pl->s[5], *s2 = pl->s[6], *dd = pl->s[7], *local = pl->s[8];
        TnrCtx c = {pl, x, p, nw, vw, local, d, s2, dd, (float)(sigma * sigma), (float)pl->p.tnr_max_frames};
        parallel_for(h, 16, tnr_pre_rows, &c);
        box_filter(dd, local, w, h, pl->p.tnr_window, pl->p.tnr_window, pl->dtmp);
        parallel_for(h, 16, tnr_mix_rows, &c); /* writes tnr_prev (= out), tnr_n, tnr_v */
    }
    pl->k_f[0] = (float)sigma;
    pmap(pl, tnr_out_chunk);
    pl->noise_sigma = sigma;
}

/* ---------------------------------------------------------------- step 3b */
typedef struct {
    Pipeline *pl;
    const float *m, *m2;
    float *a, *b;
    float st;
} GuideCtx;

static void sp_ab_rows(void *vc, int y0, int y1) {
    GuideCtx *c = vc;
    const int w = c->pl->w;
    const float *restrict m = c->m, *restrict q = c->m2, *restrict s = c->pl->noise_map;
    float *restrict a = c->a, *restrict b = c->b;
    IVDEP
    for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) {
        const float var = maxf(q[i] - m[i] * m[i], 0.f);
        const float e = (c->st * s[i]) * (c->st * s[i]);
        a[i] = var / (var + e + 1e-6f);
        b[i] = m[i] - a[i] * m[i];
    }
}

static void sp_sq_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    const float *restrict x = pl->img;
    float *restrict q = pl->s[1];
    for (size_t i = a; i < b; ++i) q[i] = x[i] * x[i];
}
/* out = a*x + b, and the sums of (out - x) for meanStdDev */
static void sp_out_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    float *restrict x = pl->img;
    const float *restrict A = pl->s[3], *restrict B = pl->s[4];
    double s1 = 0, s2 = 0;
    for (size_t i = a; i < b; ++i) {
        const float o = A[i] * x[i] + B[i];
        const double df = (double)(o - x[i]);
        s1 += df;
        s2 += df * df;
        x[i] = o;
    }
    pl->part1[k] = s1;
    pl->part2[k] = s2;
}
static void scale_noise_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    const float f = pl->k_f[0];
    float *restrict m = pl->noise_map;
    for (size_t i = a; i < b; ++i) m[i] *= f;
}

static void spatial_denoise(Pipeline *pl) {
    const int w = pl->w, h = pl->h;
    const size_t n = pl->n;
    float *x = pl->img;
    const double sigma = median(pl->noise_map, (int)n); /* large array: not reordered */
    const int k = 2 * pl->p.sp_radius + 1;
    float *m = pl->s[0], *xx = pl->s[1], *m2 = pl->s[2], *a = pl->s[3], *b = pl->s[4];
    box_filter(x, m, w, h, k, k, pl->dtmp);
    pmap(pl, sp_sq_chunk);
    box_filter(xx, m2, w, h, k, k, pl->dtmp);
    GuideCtx c = {pl, m, m2, a, b, (float)pl->p.sp_strength};
    parallel_for(h, 16, sp_ab_rows, &c);
    box_filter(a, a, w, h, k, k, pl->dtmp);
    box_filter(b, b, w, h, k, k, pl->dtmp);
    pmap(pl, sp_out_chunk);
    const double mu = sum_parts(pl->part1) / (double)n;
    const double removed = maxd(sum_parts(pl->part2) / (double)n - mu * mu, 0.0);
    const double factor = sqrt(maxd(1.0 - removed / maxd(sigma * sigma, 1e-6), 0.1));
    pl->k_f[0] = (float)factor;
    pmap(pl, scale_noise_chunk);
    pl->noise_sigma = sigma * factor;
}

/* ---------------------------------------------------------------- step 5 */
typedef struct {
    Pipeline *pl;
    const float *m, *m2;
    float *a, *b;
    float eps;
} BDCtx;
static void bd_ab_rows(void *vc, int y0, int y1) {
    BDCtx *c = vc;
    const int w = c->pl->w;
    const float *restrict m = c->m, *restrict q = c->m2;
    float *restrict a = c->a, *restrict b = c->b;
    IVDEP
    for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) {
        const float var = maxf(q[i] - m[i] * m[i], 0.f);
        a[i] = var / (var + c->eps);
        b[i] = m[i] - a[i] * m[i];
    }
}

static void bd_sum_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    double s = 0;
    for (size_t i = a; i < b; ++i) s += pl->img[i];
    pl->part1[k] = s;
}
static void bd_center_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    const float mean = pl->k_f[0];
    const float *restrict x = pl->img;
    float *restrict xc = pl->s[0], *restrict q = pl->s[1];
    for (size_t i = a; i < b; ++i) {
        xc[i] = x[i] - mean;
        q[i] = xc[i] * xc[i];
    }
}
static void bd_out_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    const float mean = pl->k_f[0];
    const float *restrict x = pl->img, *restrict xc = pl->s[0], *restrict A = pl->s[4], *restrict B = pl->s[5];
    float *restrict base = pl->base, *restrict det = pl->detail;
    for (size_t i = a; i < b; ++i) {
        base[i] = A[i] * xc[i] + B[i] + mean;
        det[i] = x[i] - base[i];
    }
}

static void base_detail_split(Pipeline *pl) {
    const int w = pl->w, h = pl->h;
    const size_t n = pl->n;
    pmap(pl, bd_sum_chunk);
    pl->k_f[0] = (float)(sum_parts(pl->part1) / (double)n);
    pmap(pl, bd_center_chunk);
    float *xc = pl->s[0], *q = pl->s[1], *m = pl->s[2], *m2 = pl->s[3], *a = pl->s[4], *b = pl->s[5];
    const int k = 2 * pl->p.bd_radius + 1;
    box_filter(xc, m, w, h, k, k, pl->dtmp);
    box_filter(q, m2, w, h, k, k, pl->dtmp);
    BDCtx c = {pl, m, m2, a, b, (float)(pl->p.bd_edge_dn * pl->p.bd_edge_dn)};
    parallel_for(h, 16, bd_ab_rows, &c);
    box_filter(a, a, w, h, k, k, pl->dtmp);
    box_filter(b, b, w, h, k, k, pl->dtmp);
    pmap(pl, bd_out_chunk);
}

/* ---------------------------------------------------------------- step 6 */
static double anchored_lo(Pipeline *pl, const float *sub, int ns, double lo_t) {
    /* black point moves at most lo_slew DN per frame, except on a real scene change */
    const double old = pl->lo;
    const double span = maxd(pl->hi - old, 1.0);
    int fast;
    if (lo_t < old) {
        size_t below = 0;
        for (int i = 0; i < ns; ++i) below += sub[i] < old;
        fast = (double)below / (double)ns > pl->p.tm_fast_frac;
    } else {
        fast = (lo_t - old) / span > pl->p.tm_fast_gap;
    }
    if (fast) {
        const double a = lo_t < old ? pl->p.tm_alpha_grow : pl->p.tm_alpha_shrink;
        return old + a * (lo_t - old);
    }
    double dlt = lo_t - old;
    if (dlt < -pl->p.tm_lo_slew) dlt = -pl->p.tm_lo_slew;
    if (dlt > pl->p.tm_lo_slew) dlt = pl->p.tm_lo_slew;
    return old + dlt;
}

static void peak(Pipeline *pl, double *old, double now, int *hold, int up) {
    if (pl->tm_first) { *old = now; *hold = pl->p.tm_peak_hold; return; }
    if (up ? now > *old : now < *old) { *old += pl->p.tm_peak_attack * (now - *old); *hold = pl->p.tm_peak_hold; }
    else if (*hold > 0) { --*hold; }
    else { *old += pl->p.tm_alpha_shrink * (now - *old); *hold = 0; }
}

static void tonemap_update_stats(Pipeline *pl) {
    const int w = pl->w, h = pl->h;
    float *keep = pl->s[6], *sub = pl->s[7];
    int ns = 0;
    for (int y = 0; y < h; y += 2)
        for (int x = 0; x < w; x += 2) keep[ns++] = pl->base[(size_t)y * w + x];
    memcpy(sub, keep, sizeof(float) * (size_t)ns);
    const double lo = percentile(sub, ns, pl->p.tm_p_low);
    const double hi = percentile(sub, ns, pl->p.tm_p_high);
    float smax = keep[0], smin = keep[0];
    for (int i = 0; i < ns; ++i) { smax = maxf(smax, keep[i]); smin = minf(smin, keep[i]); }
    if (pl->tm_first) {
        pl->lo = lo;
        pl->hi = pl->hi_hold = hi;
    } else {
        pl->lo = anchored_lo(pl, keep, ns, lo);
        if (hi > pl->hi_hold) pl->hi_hold += pl->p.tm_hi_attack * (hi - pl->hi_hold);
        else pl->hi_hold = maxd(hi, pl->hi_hold - pl->p.tm_hi_decay);
        pl->hi = pl->hi_hold;
    }
    if (pl->p.tm_min_span_dn > 0 && pl->hi - pl->lo < pl->p.tm_min_span_dn) pl->hi = pl->lo + pl->p.tm_min_span_dn;
    /* peak hold of the frame max / min (scale of the knee and the foot) */
    peak(pl, &pl->mx, smax, &pl->hold_max, 1);
    peak(pl, &pl->mn, smin, &pl->hold_min, 0);
    if (pl->hi - pl->lo < 4 * pl->p.tm_bin_dn) {
        const double c = 0.5 * (pl->hi + pl->lo);
        pl->lo = c - 2 * pl->p.tm_bin_dn;
        pl->hi = c + 2 * pl->p.tm_bin_dn;
    }
    pl->tm_first = 0;
}

typedef struct {
    Pipeline *pl;
    const float *d2, *local;
    double e0, e1, bin, k, fo, s_hi, s_lo;
    int nseg;
    float g_fix, g_cap, ad, tgt, ck, lim, wb, limit, span;
} TMCtx;

static inline float soft_top(float v, float limit, float span) {
    return v > limit ? limit + span * (1.f - expf(-(v - limit) / span)) : v;
}

static void tm_rows(void *vc, int y0, int y1) {
    TMCtx *c = vc;
    Pipeline *pl = c->pl;
    const int w = pl->w;
    const double *curve = pl->curve;
    for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) {
        /* base -> 0..1 */
        const double v = pl->base[i];
        double t;
        if (v > c->e1) {
            t = c->k + (1 - c->k) * (1 - exp(-(v - c->e1) / c->s_hi));
        } else if (v < c->e0) {
            t = c->fo * exp(-(c->e0 - v) / c->s_lo);
        } else {
            const double pos = (v - c->e0) / c->bin;
            const int j = (int)pos;
            double cv;
            if (j >= c->nseg) cv = curve[c->nseg];
            else {
                const double fr = (v - (c->e0 + j * c->bin)) / c->bin;
                cv = curve[j] + (curve[j + 1] - curve[j]) * fr;
            }
            t = cv * (c->k - c->fo) + c->fo;
        }
        /* detail (DDE): noise gate + adaptive local gain */
        const float nr = pl->noise_map[i], q = c->d2[i];
        const float nz = c->ck * nr;
        const float wgt = q / (q + nz * nz + 1e-6f);
        const float n2 = 2.f * nr;
        const float rms = sqrtf(c->local[i] + n2 * n2);
        const float gl = minf(c->tgt / rms, c->g_cap);
        const float gg = (1.f - c->ad) * c->g_fix + c->ad * gl;
        float add = gg * wgt * pl->detail[i];
        if (add > 0) add *= c->wb;
        add = add / (1.f + fabsf(add) / c->lim);
        pl->slope[i] = gg;
        /* soft limits towards 1 and 0 */
        float out = soft_top((float)(t + add), c->limit, c->span);
        out = 1.f - soft_top(1.f - out, c->limit, c->span);
        pl->img[i] = out;
    }
}

static void tm_d2_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    const float *restrict d = pl->detail;
    float *restrict q = pl->s[0];
    for (size_t i = a; i < b; ++i) q[i] = d[i] * d[i];
}

static void tonemap(Pipeline *pl) {
    const int w = pl->w, h = pl->h;
    const size_t n = pl->n;
    memcpy(pl->linear, pl->img, sizeof(float) * n);
    tonemap_update_stats(pl);

    /* curve on the grid of 2-DN bins from lo to hi: linear, then gamma */
    const double bin = pl->p.tm_bin_dn;
    const int i0 = (int)floor(pl->lo / bin), i1 = (int)ceil(pl->hi / bin);
    const int nseg = i1 + 1 - i0;
    if (nseg + 1 > pl->curve_cap) {
        xfree(pl->curve);
        pl->curve_cap = nseg + 1;
        pl->curve = ALLOC(double, pl->curve_cap);
    }
    for (int j = 0; j <= nseg; ++j) pl->curve[j] = pow((double)j / nseg, pl->p.tm_gamma);
    TMCtx c;
    c.pl = pl;
    c.bin = bin;
    c.nseg = nseg;
    c.e0 = i0 * bin;
    c.e1 = (i1 + 1) * bin;
    c.fo = pl->p.tm_foot;
    c.k = pl->p.tm_knee;
    c.s_hi = maxd((pl->mx - c.e1) / 2.0, 4 * bin);
    c.s_lo = maxd((c.e0 - pl->mn) / 2.0, 4 * bin);
    const double lin_slope = (c.k - c.fo) / (pl->hi - pl->lo);

    float *d2 = pl->s[0], *local = pl->s[1];
    pmap(pl, tm_d2_chunk);
    const int kk = 2 * pl->p.dde_radius + 1;
    box_filter(d2, local, w, h, kk, kk, pl->dtmp);
    c.d2 = d2;
    c.local = local;
    c.g_fix = (float)(pl->p.dde_gain * lin_slope);
    c.g_cap = (float)(pl->p.dde_max * lin_slope);
    c.ad = (float)pl->p.dde_adapt;
    c.tgt = (float)pl->p.dde_target;
    c.ck = (float)pl->p.dde_coring_k;
    c.lim = (float)pl->p.dde_limit;
    c.wb = (float)pl->p.warm_boost;
    c.limit = (float)pl->p.tm_limit;
    c.span = 1.f - c.limit;
    parallel_for(h, 16, tm_rows, &c);
    pl->lin_slope = lin_slope;
}

/* ---------------------------------------------------------------- step 6c */
static void th_add_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    const float thr = pl->k_f[0], gain = pl->k_f[1];
    const float *restrict th = pl->s[2];
    float *restrict add = pl->s[3];
    for (size_t i = a; i < b; ++i) add[i] = gain * maxf(th[i] - thr, 0.f);
}

static void compact_hot_boost(Pipeline *pl) {
    const int w = pl->w, h = pl->h;
    const int k = pl->p.th_size | 1;
    const int hw = w / 2, hh = h / 2;
    float *xs = pl->s[0], *op = pl->s[1], *th = pl->s[2], *add = pl->s[3];
    resize_half_area_f32(pl->linear, w, h, xs);
    morph_open(xs, op, hw, hh, k, pl->s[4]);
    for (int i = 0; i < hw * hh; ++i) op[i] = xs[i] - op[i];
    resize_linear_f32(op, hw, hh, th, w, h);
    pl->k_f[0] = (float)maxd(pl->p.th_threshold_dn, pl->p.th_sigma_k * pl->noise_sigma);
    pl->k_f[1] = (float)(pl->p.th_gain * pl->lin_slope);
    pmap(pl, th_add_chunk);
    limit_add(pl->img, add, w, h);
}

/* ---------------------------------------------------------------- step 7b */
static double band_noise_factor(double s_lo, double s_hi) {
#define ENERGY(s) ((s) <= 0 ? 1.0 : 1.0 / (4.0 * M_PI * (s) * (s)))
    const double cross = s_lo <= 0 ? 1.0 / (2.0 * M_PI * s_hi * s_hi) : 1.0 / (2.0 * M_PI * (s_lo * s_lo + s_hi * s_hi));
    const double v = ENERGY(s_lo) + ENERGY(s_hi) - 2.0 * cross;
#undef ENERGY
    return sqrt(maxd(v, 1e-6));
}

typedef struct {
    Pipeline *pl;
    const float *g1, *g2, *g3;
    float *add;
    float nf[3], gains[3], e2, ck;
} MSCtx;

static void ms_rows(void *vc, int y0, int y1) {
    MSCtx *c = vc;
    Pipeline *pl = c->pl;
    const int w = pl->w;
    const float *restrict y = pl->img, *restrict a = c->g1, *restrict b = c->g2, *restrict g = c->g3;
    const float *restrict nm = pl->noise_map, *restrict sl = pl->slope;
    float *restrict o = c->add;
    const float e2 = c->e2, ck = c->ck;
    const float n0 = c->nf[0], n1 = c->nf[1], n2 = c->nf[2];
    const float k0 = c->gains[0], k1 = c->gains[1], k2 = c->gains[2];
    IVDEP
    for (size_t i = (size_t)y0 * w; i < (size_t)y1 * w; ++i) {
        const float b0 = y[i] - a[i], b1 = a[i] - b[i], b2 = b[i] - g[i];
        const float noise = nm[i] * sl[i];
        float s = 0.f;
        {
            const float bb = b0 * b0, nn = ck * noise * n0;
            s += k0 * b0 * (bb / (bb + nn * nn + 1e-12f)) * e2 / (e2 + bb);
        }
        {
            const float bb = b1 * b1, nn = ck * noise * n1;
            s += k1 * b1 * (bb / (bb + nn * nn + 1e-12f)) * e2 / (e2 + bb);
        }
        {
            const float bb = b2 * b2, nn = ck * noise * n2;
            s += k2 * b2 * (bb / (bb + nn * nn + 1e-12f)) * e2 / (e2 + bb);
        }
        o[i] = s;
    }
}

static void multiscale_detail(Pipeline *pl) {
    const int w = pl->w, h = pl->h;
    float *g1 = pl->s[0], *g2 = pl->s[1], *g3 = pl->s[2], *add = pl->s[3], *tmp = pl->s[4];
    gaussian_blur(pl->img, g1, w, h, pl->p.ms_sigma_1, tmp);
    gaussian_blur(pl->img, g2, w, h, pl->p.ms_sigma_2, tmp);
    gaussian_blur(pl->img, g3, w, h, pl->p.ms_sigma_3, tmp);
    MSCtx c = {pl, g1, g2, g3, add,
               {(float)band_noise_factor(0.0, pl->p.ms_sigma_1), (float)band_noise_factor(pl->p.ms_sigma_1, pl->p.ms_sigma_2),
                (float)band_noise_factor(pl->p.ms_sigma_2, pl->p.ms_sigma_3)},
               {(float)pl->p.ms_fine, (float)pl->p.ms_mid, (float)pl->p.ms_coarse},
               (float)(pl->p.ms_edge * pl->p.ms_edge), (float)pl->p.ms_coring_k};
    parallel_for(h, 16, ms_rows, &c);
    limit_add(pl->img, add, w, h);
}

/* ---------------------------------------------------------------- last display step */
static void shutter_hold(Pipeline *pl) {
    if (pl->frozen) {
        if (pl->has_last) memcpy(pl->img, pl->last, sizeof(float) * pl->n);
        else for (size_t i = 0; i < pl->n; ++i) pl->img[i] = 0.5f;
        return;
    }
    memcpy(pl->last, pl->img, sizeof(float) * pl->n);
    pl->has_last = 1;
}

/* ---------------------------------------------------------------- step 9 */
/* q = floor(v + 0.5 + (t - 0.5)) with t in (0, 1); for v = 0 this gives 0 and
 * for v = 255 it gives 255, so the special cases of the C++ version need no
 * branch here and the loop vectorises. */
static uint8_t *quant_out; /* output frame of the current call */
static void quantise_chunk(Pipeline *pl, size_t a, size_t b, int k) {
    const float *restrict y = pl->img, *restrict t = pl->thr;
    uint8_t *restrict out = quant_out;
    IVDEP
    for (size_t i = a; i < b; ++i) {
        const float v = clampf(y[i], 0.f, 1.f) * 255.f;
        out[i] = (uint8_t)(int)clampf(floorf(v + 0.5f + (t[i] - 0.5f)), 0.f, 255.f);
    }
}
static void quantise(Pipeline *pl, uint8_t *out) {
    quant_out = out;
    pmap(pl, quantise_chunk);
}

/* ---------------------------------------------------------------- pipeline */
void pipeline_process(Pipeline *pl, const uint16_t *raw, uint8_t *out, double *ms) {
    double t = now_ms(), t2;
#define TICK(id)                              \
    t2 = now_ms();                            \
    if (ms) ms[id] += t2 - t;                 \
    t = t2;
    pl->frozen = 0;
    decode(pl, raw);                TICK(ST_DECODE);
    shutter_guard(pl);              TICK(ST_SHUTTER_GUARD);
    if (!pl->frozen) {              /* shutter frame: no state update */
        bad_pixels(pl);             TICK(ST_BAD_PIXELS);
        residual_nuc(pl);           TICK(ST_RESIDUAL_NUC);
        destripe(pl);               TICK(ST_DESTRIPE);
        temporal_denoise(pl);       TICK(ST_TEMPORAL_DENOISE);
        spatial_denoise(pl);        TICK(ST_SPATIAL_DENOISE);
        base_detail_split(pl);      TICK(ST_BASE_DETAIL);
        tonemap(pl);                TICK(ST_TONEMAP);
        compact_hot_boost(pl);      TICK(ST_COMPACT_HOT);
        multiscale_detail(pl);      TICK(ST_MULTISCALE);
    }
    shutter_hold(pl);               TICK(ST_SHUTTER_HOLD);
    quantise(pl, out);              TICK(ST_QUANTISE);
#undef TICK
}

/* ---------------------------------------------------------------- naive (script.py) */
void naive_agc(const uint16_t *raw, int w, int h, uint8_t *out, float *v) {
    const int n = w * h;
    for (int i = 0; i < n; ++i) v[i] = (float)raw[i];
    const double vmin = percentile(v, n, 2.0), vmax = percentile(v, n, 98.0);
    const double scale = vmax > vmin ? 255.0 / (vmax - vmin) : 0.0;
    for (int i = 0; i < n; ++i) {
        const double c = mind(maxd((double)raw[i], vmin), vmax);
        out[i] = (uint8_t)((c - vmin) * scale); /* truncation, as astype(np.uint8) */
    }
}
