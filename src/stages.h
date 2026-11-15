/* The processing stages of preset michael-features, in pipeline order.
 * Each stage is the C version of a class in the C++ version (stages.cpp)
 * and of a module in the Python version (project/thermal/stages/).
 */
#ifndef THERMAL_STAGES_H
#define THERMAL_STAGES_H

#include <stdint.h>

#include "params.h"

enum {
    ST_DECODE, ST_SHUTTER_GUARD, ST_BAD_PIXELS, ST_RESIDUAL_NUC, ST_DESTRIPE, ST_TEMPORAL_DENOISE,
    ST_SPATIAL_DENOISE, ST_BASE_DETAIL, ST_TONEMAP, ST_COMPACT_HOT, ST_MULTISCALE, ST_SHUTTER_HOLD,
    ST_QUANTISE, ST_COUNT
};
extern const char *const kStageNames[ST_COUNT];

typedef struct Pipeline Pipeline;

Pipeline *pipeline_create(const Params *p, int w, int h);
void pipeline_free(Pipeline *pl);
/* One raw 16-bit frame -> one 8-bit frame (out: w*h bytes).
 * stage_ms (may be NULL) gets the time of each stage added. */
void pipeline_process(Pipeline *pl, const uint16_t *raw, uint8_t *out, double *stage_ms);

/* The original script.py: percentiles 2 / 98 of the raw frame, clip, 8 bit.
 * scratch: w*h floats. */
void naive_agc(const uint16_t *raw, int w, int h, uint8_t *out, float *scratch);

#endif
