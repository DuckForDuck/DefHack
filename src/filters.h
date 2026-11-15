/* Image filters used by the pipeline, each matching the OpenCV call that the
 * C++ version made (same kernel sizes, borders and rounding), written in C
 * with SIMD vectors and run in parallel over rows.
 *
 * All images are dense row-major arrays: float (w*h), uint8 or int16.
 */
#ifndef THERMAL_FILTERS_H
#define THERMAL_FILTERS_H

#include <stdint.h>

/* ---- statistics ---- */
/* k-th smallest value (0-based); v is reordered (nth_element) */
float select_kth(float *v, int n, int k);
/* numpy.percentile(method="linear"); v may be reordered */
double percentile(float *v, int n, double p);
/* numpy.median; v may be reordered */
double median(float *v, int n);
/* Robust std of the random noise (MAD of a 4-neighbour Laplacian on every
 * second pixel); pixels with mask != 0 are left out (mask may be NULL).
 * scratch: at least (w/2)*(h/2) floats. */
double estimate_noise_sigma(const float *img, int w, int h, const uint8_t *mask, float *scratch);

/* ---- linear filters ---- */
/* cv::GaussianBlur(sigma, ksize from sigma, BORDER_REFLECT); tmp: w*h floats.
 * src and dst may be the same buffer. */
void gaussian_blur(const float *src, float *dst, int w, int h, double sigma, float *tmp);
/* cv::blur, box kx x ky, BORDER_REFLECT, sums in double like OpenCV;
 * tmp: w*h doubles. src and dst may be the same buffer. */
void box_filter(const float *src, float *dst, int w, int h, int kx, int ky, double *tmp);
/* cv::Sobel ksize 3, float -> float, BORDER_REFLECT_101; dx=1 or dy=1 */
void sobel3(const float *src, float *dst, int w, int h, int dx, int dy);

/* ---- morphology ---- */
/* max and min of the 8 neighbours (centre left out), BORDER_REFLECT */
void max_min_8(const float *src, float *mx, float *mn, int w, int h);
/* cv::morphologyEx(MORPH_OPEN) with a k x k rectangle (outside ignored);
 * tmp: w*h floats */
void morph_open(const float *src, float *dst, int w, int h, int k, float *tmp);

/* ---- resampling ---- */
/* cv::resize INTER_AREA by 1/2, float */
void resize_half_area_f32(const float *src, int w, int h, float *dst);
/* cv::resize INTER_AREA by an integer factor, uint8 */
void resize_area_u8(const uint8_t *src, int w, int h, int factor, uint8_t *dst);
/* cv::resize INTER_LINEAR, float */
void resize_linear_f32(const float *src, int sw, int sh, float *dst, int dw, int dh);
/* cv::remap with float maps (1/32 pixel steps), BORDER_REPLICATE */
void remap_linear_f32(const float *src, int sw, int sh, const float *mapx, const float *mapy, float *dst,
                      int dw, int dh);
void remap_cubic_f32(const float *src, int sw, int sh, const float *mapx, const float *mapy, float *dst, int dw,
                     int dh);

#endif
