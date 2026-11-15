/* DIS optical flow, port of OpenCV 4.6 modules/video/src/dis_flow.cpp and
 * variational_refinement.cpp (CPU path, mean normalisation and spatial
 * propagation on, as in every preset). Comments name the OpenCV pieces.
 */
#include "dis.h"

#include <math.h>
#include <string.h>

#include "common.h"
#include "filters.h"
#include "parallel.h"

#define MAX_LEVELS 12
#define DIS_EPS 0.001f
#define DIS_INF 1e10f
#define BSZ 16      /* border of I1_ext */
#define PSZ 8       /* patch size of every preset */
#define N_STRIPES 8 /* fixed, so the spatial propagation is reproducible */

typedef struct {
    int w, h;
    uint8_t *I0, *I1, *I1ext;
    int16_t *I0x, *I0y;
    float *Ux, *Uy;
} Level;

/* variational refinement buffers (plain grids, size of the finest level) */
typedef struct {
    float *I1f, *mapx, *mapy, *warped, *avg;
    float *Ix, *Iy, *Iz, *Ixx, *Ixy, *Iyy, *Ixz, *Iyz;
    float *A11, *A12, *A22, *b1, *b2;
    float *wgt, *uxh, *vxh, *uyv, *vyv;
    float *W0u, *W0v, *tu, *tv, *du, *dv;
    float *zero; /* one packed row of zeros */
    /* packed red/black copies for the SOR: [0] red, [1] black */
    int S; /* packed row stride */
    float *pw[2], *pia11[2], *pa12[2], *pia22[2], *pb1[2], *pb2[2], *pdu[2], *pdv[2];
} VR;

struct Dis {
    int w, h;
    int finest, coarsest;
    int pstr, gd_iter, vr_iter;
    Level L[MAX_LEVELS];
    float *Sx, *Sy, *Txx, *Tyy, *Txy, *Tx, *Ty;
    float *aux_xx, *aux_yy, *aux_xy, *aux_x, *aux_y;
    float *acc;                               /* densification: sum of weights */
    int *row_lo, *row_hi, *col_lo, *col_hi;   /* pixels of each patch row/column */
    VR vr;
};

/* ====================================================================== */
/* setup                                                                  */
/* ====================================================================== */

Dis *dis_create(int preset, int w, int h) {
    Dis *d = ALLOC(Dis, 1);
    d->w = w;
    d->h = h;
    if (preset == DIS_ULTRAFAST) { d->finest = 2; d->pstr = 4; d->gd_iter = 12; d->vr_iter = 0; }
    else if (preset == DIS_FAST) { d->finest = 2; d->pstr = 4; d->gd_iter = 16; d->vr_iter = 5; }
    else { d->finest = 1; d->pstr = 3; d->gd_iter = 25; d->vr_iter = 5; }

    const int mx = w > h ? w : h, mn = w < h ? w : h;
    int c1 = (int)(log(mx / (4.0 * PSZ)) / log(2.0) + 0.5);
    int c2 = (int)(log((double)(mn / PSZ)) / log(2.0));
    d->coarsest = c1 < c2 ? c1 : c2;
    if (d->coarsest < d->finest) { /* autoSelectPatchSizeAndScales, patch 8 */
        int c = (int)floor(log2((2.0f * (float)w) / (5.0f * (float)PSZ)));
        d->coarsest = c > 0 ? c : 0;
        d->finest = d->coarsest - 2 > 0 ? d->coarsest - 2 : 0;
    }
    if (d->coarsest >= MAX_LEVELS) d->coarsest = MAX_LEVELS - 1;

    int lw = 0, lh = 0;
    for (int i = d->finest; i <= d->coarsest; ++i) {
        if (i == d->finest) { lw = w >> i; lh = h >> i; }
        else { lw /= 2; lh /= 2; }
        Level *L = &d->L[i];
        L->w = lw;
        L->h = lh;
        L->I0 = ALLOC(uint8_t, lw * lh + 16);
        L->I1 = ALLOC(uint8_t, lw * lh + 16);
        L->I1ext = ALLOC(uint8_t, (lw + 2 * BSZ) * (lh + 2 * BSZ) + 16);
        L->I0x = ALLOC(int16_t, lw * lh + 16);
        L->I0y = ALLOC(int16_t, lw * lh + 16);
        L->Ux = ALLOC(float, lw * lh);
        L->Uy = ALLOC(float, lw * lh);
    }
    const Level *F = &d->L[d->finest];
    const int sn = (F->w / d->pstr + 1) * (F->h / d->pstr + 1);
    d->Sx = ALLOC(float, sn); d->Sy = ALLOC(float, sn);
    d->Txx = ALLOC(float, sn); d->Tyy = ALLOC(float, sn); d->Txy = ALLOC(float, sn);
    d->Tx = ALLOC(float, sn); d->Ty = ALLOC(float, sn);
    const int an = F->h * (F->w / d->pstr + 1);
    d->aux_xx = ALLOC(float, an); d->aux_yy = ALLOC(float, an); d->aux_xy = ALLOC(float, an);
    d->aux_x = ALLOC(float, an); d->aux_y = ALLOC(float, an);
    d->acc = ALLOC(float, F->w * F->h + VW);
    d->row_lo = ALLOC(int, F->h); d->row_hi = ALLOC(int, F->h);
    d->col_lo = ALLOC(int, F->w); d->col_hi = ALLOC(int, F->w);

    const int n = F->w * F->h;
    VR *v = &d->vr;
    float **bufs[] = {&v->I1f, &v->mapx, &v->mapy, &v->warped, &v->avg, &v->Ix, &v->Iy, &v->Iz, &v->Ixx,
                      &v->Ixy, &v->Iyy, &v->Ixz, &v->Iyz, &v->A11, &v->A12, &v->A22, &v->b1, &v->b2,
                      &v->wgt, &v->uxh, &v->vxh, &v->uyv, &v->vyv, &v->W0u, &v->W0v, &v->tu, &v->tv,
                      &v->du, &v->dv};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; ++i) *bufs[i] = ALLOC(float, n + VW);
    v->S = (F->w + 1) / 2 + 2 + VW;
    v->zero = ALLOC(float, v->S);
    float **pk[] = {v->pw, v->pia11, v->pa12, v->pia22, v->pb1, v->pb2, v->pdu, v->pdv};
    for (size_t i = 0; i < sizeof pk / sizeof pk[0]; ++i)
        for (int col = 0; col < 2; ++col) pk[i][col] = ALLOC(float, (size_t)v->S * F->h + VW);
    return d;
}

void dis_free(Dis *d) {
    if (!d) return;
    for (int i = d->finest; i <= d->coarsest; ++i) {
        Level *L = &d->L[i];
        xfree(L->I0); xfree(L->I1); xfree(L->I1ext); xfree(L->I0x); xfree(L->I0y); xfree(L->Ux); xfree(L->Uy);
    }
    xfree(d->Sx); xfree(d->Sy); xfree(d->Txx); xfree(d->Tyy); xfree(d->Txy); xfree(d->Tx); xfree(d->Ty);
    xfree(d->aux_xx); xfree(d->aux_yy); xfree(d->aux_xy); xfree(d->aux_x); xfree(d->aux_y);
    xfree(d->acc); xfree(d->row_lo); xfree(d->row_hi); xfree(d->col_lo); xfree(d->col_hi);
    VR *v = &d->vr;
    float *bufs[] = {v->I1f, v->mapx, v->mapy, v->warped, v->avg, v->Ix, v->Iy, v->Iz, v->Ixx, v->Ixy,
                     v->Iyy, v->Ixz, v->Iyz, v->A11, v->A12, v->A22, v->b1, v->b2, v->wgt, v->uxh,
                     v->vxh, v->uyv, v->vyv, v->W0u, v->W0v, v->tu, v->tv, v->du, v->dv, v->zero};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; ++i) xfree(bufs[i]);
    float **pk[] = {v->pw, v->pia11, v->pa12, v->pia22, v->pb1, v->pb2, v->pdu, v->pdv};
    for (size_t i = 0; i < sizeof pk / sizeof pk[0]; ++i)
        for (int col = 0; col < 2; ++col) xfree(pk[i][col]);
    xfree(d);
}

/* cv::spatialGradient: Sobel 3x3, uint8 -> int16, BORDER_REFLECT_101 */
static void spatial_gradient(const uint8_t *s, int w, int h, int16_t *dx, int16_t *dy) {
    for (int y = 0; y < h; ++y) {
        const uint8_t *u = s + (size_t)border_reflect101(y - 1, h) * w;
        const uint8_t *c = s + (size_t)y * w;
        const uint8_t *n = s + (size_t)border_reflect101(y + 1, h) * w;
        for (int x = 0; x < w; ++x) {
            const int l = border_reflect101(x - 1, w), r = border_reflect101(x + 1, w);
            dx[(size_t)y * w + x] = (int16_t)((u[r] - u[l]) + 2 * (c[r] - c[l]) + (n[r] - n[l]));
            dy[(size_t)y * w + x] = (int16_t)((n[l] + 2 * n[x] + n[r]) - (u[l] + 2 * u[x] + u[r]));
        }
    }
}

static void prepare_pyramid(Dis *d, const uint8_t *I0, const uint8_t *I1) {
    for (int i = d->finest; i <= d->coarsest; ++i) {
        Level *L = &d->L[i];
        if (i == d->finest) {
            if (i == 0) {
                memcpy(L->I0, I0, (size_t)L->w * L->h);
                memcpy(L->I1, I1, (size_t)L->w * L->h);
            } else {
                resize_area_u8(I0, d->w, d->h, 1 << i, L->I0);
                resize_area_u8(I1, d->w, d->h, 1 << i, L->I1);
            }
        } else {
            const Level *P = &d->L[i - 1];
            resize_area_u8(P->I0, P->w, P->h, 2, L->I0);
            resize_area_u8(P->I1, P->w, P->h, 2, L->I1);
        }
        /* I1 with a replicated border of 16 */
        const int we = L->w + 2 * BSZ;
        for (int y = 0; y < L->h + 2 * BSZ; ++y) {
            const uint8_t *src = L->I1 + (size_t)clampi(y - BSZ, 0, L->h - 1) * L->w;
            uint8_t *dst = L->I1ext + (size_t)y * we;
            memset(dst, src[0], BSZ);
            memcpy(dst + BSZ, src, (size_t)L->w);
            memset(dst + BSZ + L->w, src[L->w - 1], BSZ);
        }
        spatial_gradient(L->I0, L->w, L->h, L->I0x, L->I0y);
    }
}

/* precomputeStructureTensor: patch sums of Ix^2, Iy^2, IxIy, Ix, Iy on the sparse grid */
static void structure_tensor(Dis *d, const Level *L, int ws) {
    const int w = L->w, h = L->h, psz = PSZ, pstr = d->pstr;
    for (int i = 0; i < h; ++i) {
        float sxx = 0, syy = 0, sxy = 0, sx = 0, sy = 0;
        const int16_t *xr = L->I0x + (size_t)i * w, *yr = L->I0y + (size_t)i * w;
        for (int j = 0; j < psz; ++j) {
            sxx += xr[j] * xr[j]; syy += yr[j] * yr[j]; sxy += xr[j] * yr[j]; sx += xr[j]; sy += yr[j];
        }
        d->aux_xx[i * ws] = sxx; d->aux_yy[i * ws] = syy; d->aux_xy[i * ws] = sxy;
        d->aux_x[i * ws] = sx; d->aux_y[i * ws] = sy;
        int js = 1;
        for (int j = psz; j < w; ++j) {
            sxx += (xr[j] * xr[j] - xr[j - psz] * xr[j - psz]);
            syy += (yr[j] * yr[j] - yr[j - psz] * yr[j - psz]);
            sxy += (xr[j] * yr[j] - xr[j - psz] * yr[j - psz]);
            sx += (xr[j] - xr[j - psz]);
            sy += (yr[j] - yr[j - psz]);
            if ((j - psz + 1) % pstr == 0) {
                d->aux_xx[i * ws + js] = sxx; d->aux_yy[i * ws + js] = syy; d->aux_xy[i * ws + js] = sxy;
                d->aux_x[i * ws + js] = sx; d->aux_y[i * ws + js] = sy;
                js++;
            }
        }
    }
    float sxx[ws], syy[ws], sxy[ws], sx[ws], sy[ws];
    for (int j = 0; j < ws; ++j) sxx[j] = syy[j] = sxy[j] = sx[j] = sy[j] = 0.f;
    for (int i = 0; i < psz; ++i)
        for (int j = 0; j < ws; ++j) {
            sxx[j] += d->aux_xx[i * ws + j]; syy[j] += d->aux_yy[i * ws + j]; sxy[j] += d->aux_xy[i * ws + j];
            sx[j] += d->aux_x[i * ws + j]; sy[j] += d->aux_y[i * ws + j];
        }
    for (int j = 0; j < ws; ++j) {
        d->Txx[j] = sxx[j]; d->Tyy[j] = syy[j]; d->Txy[j] = sxy[j]; d->Tx[j] = sx[j]; d->Ty[j] = sy[j];
    }
    int is = 1;
    for (int i = psz; i < h; ++i) {
        for (int j = 0; j < ws; ++j) {
            sxx[j] += (d->aux_xx[i * ws + j] - d->aux_xx[(i - psz) * ws + j]);
            syy[j] += (d->aux_yy[i * ws + j] - d->aux_yy[(i - psz) * ws + j]);
            sxy[j] += (d->aux_xy[i * ws + j] - d->aux_xy[(i - psz) * ws + j]);
            sx[j] += (d->aux_x[i * ws + j] - d->aux_x[(i - psz) * ws + j]);
            sy[j] += (d->aux_y[i * ws + j] - d->aux_y[(i - psz) * ws + j]);
        }
        if ((i - psz + 1) % pstr == 0) {
            for (int j = 0; j < ws; ++j) {
                d->Txx[is * ws + j] = sxx[j]; d->Tyy[is * ws + j] = syy[j]; d->Txy[is * ws + j] = sxy[j];
                d->Tx[is * ws + j] = sx[j]; d->Ty[is * ws + j] = sy[j];
            }
            is++;
        }
    }
}

/* ====================================================================== */
/* patch inverse search                                                   */
/* ====================================================================== */

/* processPatchMeanNorm for an 8x8 patch: one SIMD vector per patch row */
static inline float patch_step(float *dUx, float *dUy, const uint8_t *I0, const uint8_t *I1, const int16_t *I0x,
                               const int16_t *I0y, int s0, int s1, float w00, float w01, float w10, float w11,
                               float xg, float yg) {
    v8f ax = v8f_set1(0), ay = v8f_set1(0), sd = v8f_set1(0), sq = v8f_set1(0);
    v8f r0 = v8f_from_u8(I1), r0s = v8f_from_u8(I1 + 1);
    I1 += s1;
    for (int r = 0; r < PSZ; ++r) {
        const v8f r1 = v8f_from_u8(I1), r1s = v8f_from_u8(I1 + 1);
        const v8f diff = r0 * w00 + r0s * w01 + r1 * w10 + r1s * w11 - v8f_from_u8(I0);
        ax += diff * v8f_from_i16(I0x);
        ay += diff * v8f_from_i16(I0y);
        sq += diff * diff;
        sd += diff;
        I0 += s0; I0x += s0; I0y += s0; I1 += s1;
        r0 = r1; r0s = r1s;
    }
    const float n = (float)(PSZ * PSZ);
    const float sum_diff = v8f_sum(sd);
    *dUx = v8f_sum(ax) - sum_diff * xg / n;
    *dUy = v8f_sum(ay) - sum_diff * yg / n;
    return v8f_sum(sq) - sum_diff * sum_diff / n;
}

/* computeSSDMeanNorm, 8x8 */
static inline float patch_ssd(const uint8_t *I0, const uint8_t *I1, int s0, int s1, float w00, float w01,
                              float w10, float w11) {
    v8f sd = v8f_set1(0), sq = v8f_set1(0);
    v8f r0 = v8f_from_u8(I1), r0s = v8f_from_u8(I1 + 1);
    I1 += s1;
    for (int r = 0; r < PSZ; ++r) {
        const v8f r1 = v8f_from_u8(I1), r1s = v8f_from_u8(I1 + 1);
        const v8f diff = r0 * w00 + r0s * w01 + r1 * w10 + r1s * w11 - v8f_from_u8(I0);
        sq += diff * diff;
        sd += diff;
        I0 += s0; I1 += s1;
        r0 = r1; r0s = r1s;
    }
    const float s = v8f_sum(sd);
    return v8f_sum(sq) - s * s / (float)(PSZ * PSZ);
}

typedef struct {
    Dis *d;
    const Level *L;
    int ws, hs, stripe_sz, num_iter;
} ISCtx;

static void inverse_search_stripes(void *vc, int s_begin, int s_end) {
    ISCtx *c = vc;
    Dis *d = c->d;
    const Level *L = c->L;
    const int w = L->w, h = L->h, ws = c->ws, hs = c->hs, pstr = d->pstr;
    const int psz2 = PSZ / 2, w_ext = w + 2 * BSZ;
    float *Sx = d->Sx, *Sy = d->Sy;
    const float *Ux = L->Ux, *Uy = L->Uy;
    const uint8_t *I0 = L->I0, *I1 = L->I1ext;
    const int16_t *I0x = L->I0x, *I0y = L->I0y;
    const float i_lo = BSZ - PSZ + 1.0f, i_hi = BSZ + h - 1.0f;
    const float j_lo = BSZ - PSZ + 1.0f, j_hi = BSZ + w - 1.0f;
    const int num_inner = (int)floorf(d->gd_iter / (float)c->num_iter);

    for (int stripe = s_begin; stripe < s_end; ++stripe) {
        for (int iter = 0; iter < c->num_iter; ++iter) {
            int dir, start_is, end_is, start_js, end_js, start_i, start_j;
            if (iter % 2 == 0) {
                dir = 1;
                start_is = stripe * c->stripe_sz < hs ? stripe * c->stripe_sz : hs;
                end_is = (stripe + 1) * c->stripe_sz < hs ? (stripe + 1) * c->stripe_sz : hs;
                start_js = 0;
                end_js = ws;
                start_i = start_is * pstr;
                start_j = 0;
            } else {
                dir = -1;
                start_is = ((stripe + 1) * c->stripe_sz < hs ? (stripe + 1) * c->stripe_sz : hs) - 1;
                end_is = (stripe * c->stripe_sz < hs ? stripe * c->stripe_sz : hs) - 1;
                start_js = ws - 1;
                end_js = -1;
                start_i = start_is * pstr;
                start_j = (ws - 1) * pstr;
            }
            int i = start_i;
            for (int is = start_is; dir * is < dir * end_is; is += dir) {
                int j = start_j;
                for (int js = start_js; dir * js < dir * end_js; js += dir) {
                    const int k = is * ws + js;
                    if (iter == 0) {
                        Sx[k] = Ux[(i + psz2) * w + j + psz2];
                        Sy[k] = Uy[(i + psz2) * w + j + psz2];
                    }
                    const uint8_t *p0 = I0 + i * w + j;
                    float i_I1, j_I1, w00, w01, w10, w11;
#define INIT_WEIGHTS(ux, uy)                                         \
    i_I1 = fminf(fmaxf(i + (uy) + BSZ, i_lo), i_hi);                 \
    j_I1 = fminf(fmaxf(j + (ux) + BSZ, j_lo), j_hi);                 \
    {                                                                \
        const float di = i_I1 - floorf(i_I1), dj = j_I1 - floorf(j_I1); \
        w11 = di * dj;                                               \
        w10 = di * (1 - dj);                                         \
        w01 = (1 - di) * dj;                                         \
        w00 = (1 - di) * (1 - dj);                                   \
    }
#define SSD_AT(ux, uy) \
    (INIT_WEIGHTS(ux, uy) patch_ssd(p0, I1 + (int)i_I1 * w_ext + (int)j_I1, w, w_ext, w00, w01, w10, w11))
                    float min_ssd, cur;
                    INIT_WEIGHTS(Sx[k], Sy[k]);
                    min_ssd = patch_ssd(p0, I1 + (int)i_I1 * w_ext + (int)j_I1, w, w_ext, w00, w01, w10, w11);
                    /* spatial candidates */
                    if (dir * js > dir * start_js) {
                        INIT_WEIGHTS(Sx[k - dir], Sy[k - dir]);
                        cur = patch_ssd(p0, I1 + (int)i_I1 * w_ext + (int)j_I1, w, w_ext, w00, w01, w10, w11);
                        if (cur < min_ssd) { min_ssd = cur; Sx[k] = Sx[k - dir]; Sy[k] = Sy[k - dir]; }
                    }
                    if (dir * is > dir * start_is) {
                        const int kk = (is - dir) * ws + js;
                        INIT_WEIGHTS(Sx[kk], Sy[kk]);
                        cur = patch_ssd(p0, I1 + (int)i_I1 * w_ext + (int)j_I1, w, w_ext, w00, w01, w10, w11);
                        if (cur < min_ssd) { min_ssd = cur; Sx[k] = Sx[kk]; Sy[k] = Sy[kk]; }
                    }
                    (void)min_ssd;
                    /* gradient descent from the best candidate */
                    float cur_ux = Sx[k], cur_uy = Sy[k];
                    float detH = d->Txx[k] * d->Tyy[k] - d->Txy[k] * d->Txy[k];
                    if (fabsf(detH) < DIS_EPS) detH = DIS_EPS;
                    const float iH11 = d->Tyy[k] / detH, iH12 = -d->Txy[k] / detH, iH22 = d->Txx[k] / detH;
                    const float xg = d->Tx[k], yg = d->Ty[k];
                    float prev_ssd = DIS_INF;
                    for (int t = 0; t < num_inner; ++t) {
                        float dUx, dUy;
                        INIT_WEIGHTS(cur_ux, cur_uy);
                        const float ssd = patch_step(&dUx, &dUy, p0, I1 + (int)i_I1 * w_ext + (int)j_I1,
                                                     I0x + i * w + j, I0y + i * w + j, w, w_ext, w00, w01, w10,
                                                     w11, xg, yg);
                        cur_ux -= iH11 * dUx + iH12 * dUy;
                        cur_uy -= iH12 * dUx + iH22 * dUy;
                        if (ssd >= prev_ssd) break;
                        prev_ssd = ssd;
                    }
#undef INIT_WEIGHTS
#undef SSD_AT
                    const float ex = cur_ux - Sx[k], ey = cur_uy - Sy[k];
                    if (sqrtf(ex * ex + ey * ey) <= PSZ) { Sx[k] = cur_ux; Sy[k] = cur_uy; }
                    j += dir * pstr;
                }
                i += dir * pstr;
            }
        }
    }
}

/* ====================================================================== */
/* densification                                                          */
/* ====================================================================== */

/* Each pixel gets the weighted mean of the flow of the patches that cover it
 * (OpenCV: Densification_ParBody). Here the loop runs over the patches and
 * adds each patch to its pixels, 8 pixels of a row at a time; every pixel
 * still receives the patches in the same order (is, then js), so the sums
 * are the same as in the per-pixel loop of OpenCV. */
typedef struct {
    Dis *d;
    Level *L;
    int ws, hs;
} DenCtx;

/* Patches used by each pixel row (OpenCV's UPDATE_SPARSE_*_COORDINATES),
 * turned into the pixel range [lo[s], hi[s]] of each patch s. */
static void patch_pixel_ranges(int n, int pstr, int ns, int *lo, int *hi) {
    for (int s = 0; s < ns; ++s) { lo[s] = n; hi[s] = -1; }
    int start = 0, end = -1;
    for (int i = 0; i < n; ++i) {
        if (i % pstr == 0 && i + PSZ <= n) end++;
        if (i - PSZ >= 0 && (i - PSZ) % pstr == 0 && start < end) start++;
        for (int s = start; s <= end; ++s) {
            if (i < lo[s]) lo[s] = i;
            if (i > hi[s]) hi[s] = i;
        }
    }
}

static void densify_rows(void *vc, int y0, int y1) {
    DenCtx *c = vc;
    Dis *d = c->d;
    Level *L = c->L;
    const int w = L->w, h = L->h, ws = c->ws, hs = c->hs;
    const float *Sx = d->Sx, *Sy = d->Sy;
    const uint8_t *I0 = L->I0, *I1 = L->I1;
    float *ax = L->Ux, *ay = L->Uy, *ac = d->acc;
    for (int i = y0; i < y1; ++i) {
        memset(ax + (size_t)i * w, 0, sizeof(float) * (size_t)w);
        memset(ay + (size_t)i * w, 0, sizeof(float) * (size_t)w);
        memset(ac + (size_t)i * w, 0, sizeof(float) * (size_t)w);
    }
    const float jmax = w - 1.0f - DIS_EPS, imax = h - 1.0f - DIS_EPS;
    const v8f lane = {0, 1, 2, 3, 4, 5, 6, 7};
    const v8i abs_mask = v8f_set1(0) == v8f_set1(0); /* all bits set */
    const v8i sign_off = abs_mask ^ (v8i)v8f_set1(-0.0f);
    for (int is = 0; is < hs; ++is) {
        const int r0 = d->row_lo[is] > y0 ? d->row_lo[is] : y0;
        const int r1 = d->row_hi[is] < y1 - 1 ? d->row_hi[is] : y1 - 1;
        if (r0 > r1) continue;
        for (int js = 0; js < ws; ++js) {
            const float sx = Sx[is * ws + js], sy = Sy[is * ws + js];
            const int c0 = d->col_lo[js], c1 = d->col_hi[js];
            for (int i = r0; i <= r1; ++i) {
                const float i_m = fminf(fmaxf(i + sy, 0.0f), imax);
                const int i_l = (int)i_m, i_u = i_l + 1;
                const float di = i_m - i_l, ei = i_u - i_m;
                const uint8_t *rl = I1 + (size_t)i_l * w, *ru = I1 + (size_t)i_u * w;
                const uint8_t *r0p = I0 + (size_t)i * w;
                float *px = ax + (size_t)i * w, *py = ay + (size_t)i * w, *pc = ac + (size_t)i * w;
                int j = c0;
                for (; j + VW <= c1 + 1; j += VW) {
                    const v8f jm = v8f_min(v8f_max(lane + (float)j + sx, v8f_set1(0.0f)), v8f_set1(jmax));
                    const v8i jl = __builtin_convertvector(jm, v8i);
                    const v8f jlf = __builtin_convertvector(jl, v8f);
                    const v8f ju = jlf + 1.0f;
                    v8f vuu, vul, vlu, vll;
                    const int j0 = jl[0];
                    const v8i expect = (v8i){0, 1, 2, 3, 4, 5, 6, 7} + j0;
                    const v8i same = jl == expect;
                    if ((same[0] & same[1] & same[2] & same[3] & same[4] & same[5] & same[6] & same[7]) != 0) {
                        vll = v8f_from_u8(rl + j0); vlu = v8f_from_u8(rl + j0 + 1);
                        vul = v8f_from_u8(ru + j0); vuu = v8f_from_u8(ru + j0 + 1);
                    } else {
                        for (int k = 0; k < VW; ++k) {
                            vll[k] = rl[jl[k]]; vlu[k] = rl[jl[k] + 1];
                            vul[k] = ru[jl[k]]; vuu[k] = ru[jl[k] + 1];
                        }
                    }
                    const v8f dj = jm - jlf, ej = ju - jm;
                    const v8f diff = dj * di * vuu + ej * di * vul + dj * ei * vlu + ej * ei * vll - v8f_from_u8(r0p + j);
                    const v8f ad = (v8f)((v8i)diff & sign_off);
                    const v8f coef = 1.0f / v8f_max(v8f_set1(1.0f), ad);
                    v8f_store(px + j, v8f_load(px + j) + coef * sx);
                    v8f_store(py + j, v8f_load(py + j) + coef * sy);
                    v8f_store(pc + j, v8f_load(pc + j) + coef);
                }
                for (; j <= c1; ++j) {
                    const float j_m = fminf(fmaxf(j + sx, 0.0f), jmax);
                    const int j_l = (int)j_m, j_u = j_l + 1;
                    const float diff = (j_m - j_l) * di * ru[j_u] + (j_u - j_m) * di * ru[j_l] +
                                       (j_m - j_l) * ei * rl[j_u] + (j_u - j_m) * ei * rl[j_l] - r0p[j];
                    const float coef = 1 / fmaxf(1.0f, fabsf(diff));
                    px[j] += coef * sx;
                    py[j] += coef * sy;
                    pc[j] += coef;
                }
            }
        }
    }
    for (size_t p = (size_t)y0 * w; p < (size_t)y1 * w; ++p) {
        ax[p] = ax[p] / ac[p];
        ay[p] = ay[p] / ac[p];
    }
}

/* ====================================================================== */
/* variational refinement (plain grid, red-black SOR)                     */
/* ====================================================================== */

#define VR_ALPHA 20.0f
#define VR_DELTA 5.0f
#define VR_GAMMA 10.0f
#define VR_OMEGA 1.6f
#define VR_ZETA 0.1f
#define VR_EPSILON 0.001f
#define VR_SOR_ITERS 5

typedef struct {
    VR *v;
    int w, h;
    const uint8_t *I0;
    float *Wu, *Wv;
    int color; /* SOR: 0 = red ((x+y) even), 1 = black */
} VRCtx;

/* Sobel ksize 1 ([-1 0 1]), BORDER_REPLICATE */
static void deriv_x(const float *s, float *d, int w, int h) {
    for (int y = 0; y < h; ++y) {
        const float *r = s + (size_t)y * w;
        float *o = d + (size_t)y * w;
        o[0] = r[w > 1 ? 1 : 0] - r[0];
        for (int x = 1; x < w - 1; ++x) o[x] = r[x + 1] - r[x - 1];
        if (w > 1) o[w - 1] = r[w - 1] - r[w - 2];
    }
}
static void deriv_y(const float *s, float *d, int w, int h) {
    for (int y = 0; y < h; ++y) {
        const float *u = s + (size_t)clampi(y - 1, 0, h - 1) * w, *n = s + (size_t)clampi(y + 1, 0, h - 1) * w;
        float *o = d + (size_t)y * w;
        for (int x = 0; x < w; ++x) o[x] = n[x] - u[x];
    }
}

static void vr_prepare_rows(void *vc, int y0, int y1) {
    VRCtx *c = vc;
    VR *v = c->v;
    const int w = c->w;
    for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t p = (size_t)y * w + x;
            const float i0 = c->I0[p], wp = v->warped[p];
            v->avg[p] = i0 * 0.5f + wp * 0.5f;
            v->Iz[p] = wp - i0;
        }
}

static void vr_prepare(VRCtx *c, const uint8_t *I1) {
    VR *v = c->v;
    const int w = c->w, h = c->h, n = w * h;
    for (int i = 0; i < n; ++i) v->I1f[i] = I1[i];
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t p = (size_t)y * w + x;
            v->mapx[p] = x + c->Wu[p];
            v->mapy[p] = y + c->Wv[p];
        }
    remap_linear_f32(v->I1f, w, h, v->mapx, v->mapy, v->warped, w, h);
    parallel_for(h, 8, vr_prepare_rows, c);
    deriv_x(v->avg, v->Ix, w, h);
    deriv_y(v->avg, v->Iy, w, h);
    deriv_x(v->Iz, v->Ixz, w, h);
    deriv_y(v->Iz, v->Iyz, w, h);
    deriv_x(v->Ix, v->Ixx, w, h);
    deriv_y(v->Ix, v->Ixy, w, h);
    deriv_y(v->Iy, v->Iyy, w, h);
}

/* ComputeDataTerm: colour and gradient constancy */
static void vr_data_rows(void *vc, int y0, int y1) {
    VRCtx *c = vc;
    VR *v = c->v;
    const float zeta2 = VR_ZETA * VR_ZETA, eps2 = VR_EPSILON * VR_EPSILON;
    const float gamma2 = VR_GAMMA / 2, delta2 = VR_DELTA / 2;
    const size_t a = (size_t)y0 * c->w, b = (size_t)y1 * c->w;
    const float *restrict Ix = v->Ix, *restrict Iy = v->Iy, *restrict Iz = v->Iz;
    const float *restrict Ixx = v->Ixx, *restrict Ixy = v->Ixy, *restrict Iyy = v->Iyy;
    const float *restrict Ixz = v->Ixz, *restrict Iyz = v->Iyz, *restrict dU = v->du, *restrict dV = v->dv;
    float *restrict a11 = v->A11, *restrict a12 = v->A12, *restrict a22 = v->A22;
    float *restrict b1 = v->b1, *restrict b2 = v->b2;
    IVDEP
    for (size_t j = a; j < b; ++j) {
        float derivNorm = Ix[j] * Ix[j] + Iy[j] * Iy[j] + zeta2;
        const float Ik1z = Iz[j] + Ix[j] * dU[j] + Iy[j] * dV[j];
        float weight = (delta2 / sqrtf(Ik1z * Ik1z / derivNorm + eps2)) / derivNorm;
        float A11 = weight * (Ix[j] * Ix[j]) + zeta2;
        float A12 = weight * (Ix[j] * Iy[j]);
        float A22 = weight * (Iy[j] * Iy[j]) + zeta2;
        float B1 = -weight * (Iz[j] * Ix[j]);
        float B2 = -weight * (Iz[j] * Iy[j]);
        derivNorm = Ixx[j] * Ixx[j] + Ixy[j] * Ixy[j] + zeta2;
        const float derivNorm2 = Iyy[j] * Iyy[j] + Ixy[j] * Ixy[j] + zeta2;
        const float Ik1zx = Ixz[j] + Ixx[j] * dU[j] + Ixy[j] * dV[j];
        const float Ik1zy = Iyz[j] + Ixy[j] * dU[j] + Iyy[j] * dV[j];
        weight = gamma2 / sqrtf(Ik1zx * Ik1zx / derivNorm + Ik1zy * Ik1zy / derivNorm2 + eps2);
        A11 += weight * (Ixx[j] * Ixx[j] / derivNorm + Ixy[j] * Ixy[j] / derivNorm2);
        A12 += weight * (Ixx[j] * Ixy[j] / derivNorm + Ixy[j] * Iyy[j] / derivNorm2);
        A22 += weight * (Ixy[j] * Ixy[j] / derivNorm + Iyy[j] * Iyy[j] / derivNorm2);
        B1 += -weight * (Ixx[j] * Ixz[j] / derivNorm + Ixy[j] * Iyz[j] / derivNorm2);
        B2 += -weight * (Ixy[j] * Ixz[j] / derivNorm + Iyy[j] * Iyz[j] / derivNorm2);
        a11[j] = A11; a12[j] = A12; a22[j] = A22; b1[j] = B1; b2[j] = B2;
    }
}

/* Smoothness weights from the current flow, and the weighted differences of
 * the input flow to the right (h) and to the row below (v). */
static void vr_weights_rows(void *vc, int y0, int y1) {
    VRCtx *c = vc;
    VR *v = c->v;
    const int w = c->w, h = c->h;
    const float alpha2 = VR_ALPHA / 2, eps2 = VR_EPSILON * VR_EPSILON;
    for (int y = y0; y < y1; ++y) {
        const size_t r = (size_t)y * w;
        const size_t rn = (size_t)(y + 1 < h ? y + 1 : y) * w; /* replicated bottom */
        const float *restrict tu = v->tu + r, *restrict tv = v->tv + r;
        const float *restrict tun = v->tu + rn, *restrict tvn = v->tv + rn;
        const float *restrict W0u = v->W0u + r, *restrict W0v = v->W0v + r;
        const float *restrict W0un = v->W0u + rn, *restrict W0vn = v->W0v + rn;
        float *restrict wg = v->wgt + r, *restrict uxh = v->uxh + r, *restrict vxh = v->vxh + r;
        float *restrict uyv = v->uyv + r, *restrict vyv = v->vyv + r;
        IVDEP
        for (int x = 0; x < w - 1; ++x) {
            const float ux = tu[x + 1] - tu[x], vx = tv[x + 1] - tv[x];
            const float uy = tun[x] - tu[x], vy = tvn[x] - tv[x];
            const float wt = alpha2 / sqrtf(ux * ux + vx * vx + uy * uy + vy * vy + eps2);
            wg[x] = wt;
            uxh[x] = wt * (W0u[x + 1] - W0u[x]);
            vxh[x] = wt * (W0v[x + 1] - W0v[x]);
            uyv[x] = wt * (W0un[x] - W0u[x]);
            vyv[x] = wt * (W0vn[x] - W0v[x]);
        }
        {
            const int x = w - 1; /* right border: x gradient 0 */
            const float uy = tun[x] - tu[x], vy = tvn[x] - tv[x];
            const float wt = alpha2 / sqrtf(0.f * 0.f + 0.f * 0.f + uy * uy + vy * vy + eps2);
            wg[x] = wt;
            uxh[x] = 0.f;
            vxh[x] = 0.f;
            uyv[x] = wt * (W0un[x] - W0u[x]);
            vyv[x] = wt * (W0vn[x] - W0v[x]);
        }
    }
}

/* Adds the smoothness terms to A11, A22, b1, b2 in the order OpenCV's
 * red/black horizontal and vertical passes add them. */
static void vr_accum_rows(void *vc, int y0, int y1) {
    VRCtx *c = vc;
    VR *v = c->v;
    const int w = c->w, h = c->h;
    for (int y = y0; y < y1; ++y) {
        const int up = y > 0, self_v = y < h - 1;
        for (int x = 0; x < w; ++x) {
            const size_t p = (size_t)y * w + x;
            const int left = x > 0, self_h = x < w - 1;
            const int red = ((x + y) & 1) == 0;
            float A11 = v->A11[p], A22 = v->A22[p], B1 = v->b1[p], B2 = v->b2[p];
            const float ws = v->wgt[p];
            const float wl = left ? v->wgt[p - 1] : 0.f, wu = up ? v->wgt[p - w] : 0.f;
            if (red) {
                if (self_h) { B1 += v->uxh[p]; A11 += ws; B2 += v->vxh[p]; A22 += ws; }
                if (left) { B1 -= v->uxh[p - 1]; A11 += wl; B2 -= v->vxh[p - 1]; A22 += wl; }
                if (self_v) { B1 += v->uyv[p]; A11 += ws; B2 += v->vyv[p]; A22 += ws; }
                if (up) { B1 -= v->uyv[p - w]; A11 += wu; B2 -= v->vyv[p - w]; A22 += wu; }
            } else {
                if (left) { B1 -= v->uxh[p - 1]; A11 += wl; B2 -= v->vxh[p - 1]; A22 += wl; }
                if (self_h) { B1 += v->uxh[p]; A11 += ws; B2 += v->vxh[p]; A22 += ws; }
                if (up) { B1 -= v->uyv[p - w]; A11 += wu; B2 -= v->vyv[p - w]; A22 += wu; }
                if (self_v) { B1 += v->uyv[p]; A11 += ws; B2 += v->vyv[p]; A22 += ws; }
            }
            v->A11[p] = A11; v->A22[p] = A22; v->b1[p] = B1; v->b2[p] = B2;
        }
    }
}

/* ---- red-black SOR on packed rows (the layout OpenCV uses) ----
 * Row y of colour c (0 = red: (x+y) even, 1 = black) keeps the pixels
 * x = 2k + off, off = (y+c)&1, at index 1+k of a row of stride S. Index 0
 * and the slots after the last pixel stay 0, so missing neighbours at the
 * borders read 0 (zero weight / zero increment), as in OpenCV. In this
 * layout the left/right neighbours of a pixel are the other colour at
 * k+off-1 / k+off, the neighbours above/below are the other colour at k. */

static inline int rb_len(int w, int off) { return (w - off + 1) / 2; }

typedef struct {
    VRCtx *c;
    const float *src;
    float *dst[2];
    int inv; /* store 1/value */
} PackCtx;

static void pack_rows(void *vc, int y0, int y1) {
    PackCtx *p = vc;
    const int w = p->c->w, S = p->c->v->S;
    for (int y = y0; y < y1; ++y)
        for (int col = 0; col < 2; ++col) {
            const int off = (y + col) & 1, len = rb_len(w, off);
            const float *s = p->src + (size_t)y * w + off;
            float *d = p->dst[col] + (size_t)y * S + 1;
            if (p->inv) for (int k = 0; k < len; ++k) d[k] = 1.f / s[2 * k];
            else for (int k = 0; k < len; ++k) d[k] = s[2 * k];
            d[-1] = 0.f; /* border slots read as 0 */
            d[len] = 0.f;
        }
}

static void unpack_rows(void *vc, int y0, int y1) {
    PackCtx *p = vc; /* dst[0..1] -> src (used as output) */
    const int w = p->c->w, S = p->c->v->S;
    float *out = (float *)p->src;
    for (int y = y0; y < y1; ++y)
        for (int col = 0; col < 2; ++col) {
            const int off = (y + col) & 1, len = rb_len(w, off);
            const float *s = p->dst[col] + (size_t)y * S + 1;
            float *d = out + (size_t)y * w + off;
            for (int k = 0; k < len; ++k) d[2 * k] = s[k];
        }
}

static void vr_pack(VRCtx *c, const float *src, float *dst[2], int inv, int grain) {
    PackCtx p = {c, src, {dst[0], dst[1]}, inv};
    parallel_for(c->h, grain, pack_rows, &p);
}
static void vr_unpack(VRCtx *c, float *dst, float *src[2], int grain) {
    PackCtx p = {c, dst, {src[0], src[1]}, 0};
    parallel_for(c->h, grain, unpack_rows, &p);
}

static void vr_sor_rows(void *vc, int y0, int y1) {
    VRCtx *c = vc;
    VR *v = c->v;
    const int w = c->w, h = c->h, S = v->S;
    const int me = c->color, ot = 1 - me;
    const v8f om = v8f_set1(VR_OMEGA);
    for (int y = y0; y < y1; ++y) {
        const int off = (y + me) & 1, len = rb_len(w, off);
        const size_t r = (size_t)y * S + 1;
        float *du = v->pdu[me] + r, *dv = v->pdv[me] + r;
        const float *W = v->pw[me] + r;
        const float *duO = v->pdu[ot] + r + off, *dvO = v->pdv[ot] + r + off; /* [k-1] left, [k] right */
        const float *WO = v->pw[ot] + r + off;
        const float *WU = y > 0 ? v->pw[ot] + r - S : v->zero + 1;
        const float *duU = y > 0 ? v->pdu[ot] + r - S : v->zero + 1, *dvU = y > 0 ? v->pdv[ot] + r - S : v->zero + 1;
        const float *duD = y < h - 1 ? v->pdu[ot] + r + S : v->zero + 1;
        const float *dvD = y < h - 1 ? v->pdv[ot] + r + S : v->zero + 1;
        const float *ia11 = v->pia11[me] + r, *a12 = v->pa12[me] + r, *ia22 = v->pia22[me] + r;
        const float *b1 = v->pb1[me] + r, *b2 = v->pb2[me] + r;
        int k = 0;
        for (; k + VW <= len; k += VW) {
            const v8f wl = v8f_load(WO + k - 1), wc = v8f_load(W + k), wu = v8f_load(WU + k);
            const v8f sU = wl * v8f_load(duO + k - 1) + wc * v8f_load(duO + k) + wu * v8f_load(duU + k) +
                           wc * v8f_load(duD + k);
            const v8f sV = wl * v8f_load(dvO + k - 1) + wc * v8f_load(dvO + k) + wu * v8f_load(dvU + k) +
                           wc * v8f_load(dvD + k);
            const v8f u0 = v8f_load(du + k), v0 = v8f_load(dv + k), A12 = v8f_load(a12 + k);
            const v8f u1 = u0 + om * ((sU + v8f_load(b1 + k) - v0 * A12) * v8f_load(ia11 + k) - u0);
            const v8f v1 = v0 + om * ((sV + v8f_load(b2 + k) - u1 * A12) * v8f_load(ia22 + k) - v0);
            v8f_store(du + k, u1);
            v8f_store(dv + k, v1);
        }
        for (; k < len; ++k) {
            const float wl = WO[k - 1], wc = W[k], wu = WU[k];
            const float sU = wl * duO[k - 1] + wc * duO[k] + wu * duU[k] + wc * duD[k];
            const float sV = wl * dvO[k - 1] + wc * dvO[k] + wu * dvU[k] + wc * dvD[k];
            du[k] += VR_OMEGA * ((sU + b1[k] - dv[k] * a12[k]) * ia11[k] - du[k]);
            dv[k] += VR_OMEGA * ((sV + b2[k] - du[k] * a12[k]) * ia22[k] - dv[k]);
        }
    }
}

static void vr_update_rows(void *vc, int y0, int y1) {
    VRCtx *c = vc;
    VR *v = c->v;
    const size_t a = (size_t)y0 * c->w, b = (size_t)y1 * c->w;
    IVDEP
    for (size_t p = a; p < b; ++p) {
        v->tu[p] = v->W0u[p] + v->du[p];
        v->tv[p] = v->W0v[p] + v->dv[p];
    }
}

static void variational_refinement(Dis *d, Level *L) {
    VR *v = &d->vr;
    const int w = L->w, h = L->h, n = w * h;
    VRCtx c = {v, w, h, L->I0, L->Ux, L->Uy, 0};
    vr_prepare(&c, L->I1);
    memcpy(v->W0u, L->Ux, sizeof(float) * (size_t)n);
    memcpy(v->W0v, L->Uy, sizeof(float) * (size_t)n);
    memcpy(v->tu, L->Ux, sizeof(float) * (size_t)n);
    memcpy(v->tv, L->Uy, sizeof(float) * (size_t)n);
    memset(v->du, 0, sizeof(float) * (size_t)n);
    memset(v->dv, 0, sizeof(float) * (size_t)n);
    memset(v->zero, 0, sizeof(float) * (size_t)v->S);
    for (int col = 0; col < 2; ++col) {
        memset(v->pdu[col], 0, sizeof(float) * (size_t)v->S * h);
        memset(v->pdv[col], 0, sizeof(float) * (size_t)v->S * h);
    }
    /* rows per thread: at least ~4096 pixels, small levels run on one thread */
    int grain = 4096 / w;
    if (grain < 4) grain = 4;
    for (int it = 0; it < d->vr_iter; ++it) {
        parallel_for(h, grain, vr_data_rows, &c);
        parallel_for(h, grain, vr_weights_rows, &c);
        parallel_for(h, grain, vr_accum_rows, &c);
        vr_pack(&c, v->wgt, v->pw, 0, grain);
        vr_pack(&c, v->A11, v->pia11, 1, grain);
        vr_pack(&c, v->A12, v->pa12, 0, grain);
        vr_pack(&c, v->A22, v->pia22, 1, grain);
        vr_pack(&c, v->b1, v->pb1, 0, grain);
        vr_pack(&c, v->b2, v->pb2, 0, grain);
        for (int s = 0; s < VR_SOR_ITERS; ++s) {
            c.color = 0;
            parallel_for(h, grain, vr_sor_rows, &c);
            c.color = 1;
            parallel_for(h, grain, vr_sor_rows, &c);
        }
        vr_unpack(&c, v->du, v->pdu, grain);
        vr_unpack(&c, v->dv, v->pdv, grain);
        parallel_for(h, grain, vr_update_rows, &c);
    }
    memcpy(L->Ux, v->tu, sizeof(float) * (size_t)n);
    memcpy(L->Uy, v->tv, sizeof(float) * (size_t)n);
}

/* ====================================================================== */
/* main loop over the pyramid                                             */
/* ====================================================================== */

void dis_calc(Dis *d, const uint8_t *I0, const uint8_t *I1, float *flow_x, float *flow_y) {
    prepare_pyramid(d, I0, I1);
    Level *C = &d->L[d->coarsest];
    memset(C->Ux, 0, sizeof(float) * (size_t)C->w * C->h);
    memset(C->Uy, 0, sizeof(float) * (size_t)C->w * C->h);

    for (int i = d->coarsest; i >= d->finest; --i) {
        Level *L = &d->L[i];
        const int ws = 1 + (L->w - PSZ) / d->pstr;
        const int hs = 1 + (L->h - PSZ) / d->pstr;
        structure_tensor(d, L, ws);

        ISCtx is = {d, L, ws, hs, (int)ceil(hs / (double)N_STRIPES), 2};
        parallel_for(N_STRIPES, 1, inverse_search_stripes, &is);

        patch_pixel_ranges(L->h, d->pstr, hs, d->row_lo, d->row_hi);
        patch_pixel_ranges(L->w, d->pstr, ws, d->col_lo, d->col_hi);
        DenCtx dc = {d, L, ws, hs};
        parallel_for(L->h, 8, densify_rows, &dc);

        if (d->vr_iter > 0) variational_refinement(d, L);

        if (i > d->finest) {
            Level *N = &d->L[i - 1];
            resize_linear_f32(L->Ux, L->w, L->h, N->Ux, N->w, N->h);
            resize_linear_f32(L->Uy, L->w, L->h, N->Uy, N->w, N->h);
            const int n = N->w * N->h;
            for (int k = 0; k < n; ++k) { N->Ux[k] *= 2; N->Uy[k] *= 2; }
        }
    }
    Level *F = &d->L[d->finest];
    resize_linear_f32(F->Ux, F->w, F->h, flow_x, d->w, d->h);
    resize_linear_f32(F->Uy, F->w, F->h, flow_y, d->w, d->h);
    const float s = (float)(1 << d->finest);
    const int n = d->w * d->h;
    for (int k = 0; k < n; ++k) { flow_x[k] *= s; flow_y[k] *= s; }
}
