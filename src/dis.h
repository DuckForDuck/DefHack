/* DIS optical flow (Kroeger et al. 2016) in C: the same algorithm, presets
 * and parameters as cv::DISOpticalFlow of OpenCV 4.6 (dis_flow.cpp and
 * variational_refinement.cpp), with SIMD patch processing and a vectorised
 * red-black SOR.
 */
#ifndef THERMAL_DIS_H
#define THERMAL_DIS_H

#include <stdint.h>

enum { DIS_ULTRAFAST = 0, DIS_FAST = 1, DIS_MEDIUM = 2 };

typedef struct Dis Dis;

Dis *dis_create(int preset, int w, int h);
void dis_free(Dis *d);
/* Flow from I0 to I1 (8-bit, w x h): I0(x) ~ I1(x + flow(x)) */
void dis_calc(Dis *d, const uint8_t *I0, const uint8_t *I1, float *flow_x, float *flow_y);

#endif
