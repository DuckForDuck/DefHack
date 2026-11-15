/* Parameters of the pipeline = preset presets/michael-features.json of the
 * Python version (project/). Each value names its preset key.
 * Only the modules that are ON in that preset are implemented.
 */
#ifndef THERMAL_PARAMS_H
#define THERMAL_PARAMS_H

typedef struct {
    /* step 0 - Decode */
    int bit_depth;

    /* step 0b - ShutterGuard (shutter / FFC frames) */
    double shutter_ratio;   /* std < ratio x median of history */
    int shutter_max_freeze; /* max. frozen frames in a row */
    int shutter_history;    /* past normal frames for the reference std */

    /* step 1 - BadPixels */
    double bp_k, bp_k_replace, bp_persist_ratio, bp_ema, bp_stuck_ratio;
    int bp_warmup;

    /* step 2a - ResidualNUC (online only) */
    double nuc_alpha, nuc_blur_sigma, nuc_clip_k, nuc_edge_k, nuc_motion_k;

    /* step 2b - Destripe */
    int ds_width;
    double ds_clip_k, ds_edge_k, ds_edge_sigma, ds_min_valid;
    double ds_temporal_alpha; /* Destripe.temporal_alpha */

    /* step 3a - TemporalDenoise */
    int tnr_max_frames; /* TemporalDenoise.max_frames */
    int tnr_window;
    double tnr_local_lo, tnr_local_hi;
    double tnr_pixel_lo, tnr_pixel_hi; /* TemporalDenoise.pixel_hi */
    double tnr_flow_smooth;
    int tnr_flow_preset; /* DIS: 2 = medium (preset), 1 = fast, 0 = ultrafast */

    /* step 3b - SpatialDenoise (guided) */
    double sp_strength; /* SpatialDenoise.strength */
    int sp_radius;

    /* step 5 - BaseDetailSplit */
    int bd_radius;    /* BaseDetailSplit.radius */
    double bd_edge_dn; /* BaseDetailSplit.edge_dn */

    /* step 6 - ToneMap */
    double tm_p_low, tm_p_high, tm_alpha_grow, tm_alpha_shrink, tm_foot, tm_knee, tm_limit, tm_gamma, tm_bin_dn;
    double tm_peak_attack;
    int tm_peak_hold;
    double tm_lo_slew, tm_fast_frac, tm_fast_gap, tm_hi_attack, tm_hi_decay, tm_min_span_dn;
    /* DDE detail */
    double dde_gain, dde_adapt, dde_target, dde_max;
    int dde_radius;
    double dde_coring_k, dde_limit, warm_boost;

    /* step 6c - CompactHotBoost (white top-hat, soft) */
    double th_gain;
    int th_size;
    double th_threshold_dn, th_sigma_k;

    /* step 7b - MultiScaleDetail */
    double ms_fine, ms_mid, ms_coarse, ms_sigma_1, ms_sigma_2, ms_sigma_3, ms_edge, ms_coring_k;

    /* step 10 - video */
    double fps;
    int crf;
    const char *x264_preset, *x264_tune;
} Params;

static inline Params params_default(void) {
    Params p = {
        .bit_depth = 14,
        .shutter_ratio = 0.35, .shutter_max_freeze = 30, .shutter_history = 16,
        .bp_k = 8.0, .bp_k_replace = 15.0, .bp_persist_ratio = 0.5, .bp_ema = 0.05, .bp_stuck_ratio = 0.05,
        .bp_warmup = 30,
        .nuc_alpha = 0.02, .nuc_blur_sigma = 2.0, .nuc_clip_k = 5.0, .nuc_edge_k = 3.0, .nuc_motion_k = 3.0,
        .ds_width = 9, .ds_clip_k = 2.0, .ds_edge_k = 0.5, .ds_edge_sigma = 1.5, .ds_min_valid = 0.2,
        .ds_temporal_alpha = 0.1,
        .tnr_max_frames = 8, .tnr_window = 5, .tnr_local_lo = 1.6, .tnr_local_hi = 3.0,
        .tnr_pixel_lo = 3.0, .tnr_pixel_hi = 5.0, .tnr_flow_smooth = 1.0, .tnr_flow_preset = 2,
        .sp_strength = 1.5, .sp_radius = 2,
        .bd_radius = 8, .bd_edge_dn = 25.0,
        .tm_p_low = 0.2, .tm_p_high = 99.8, .tm_alpha_grow = 0.15, .tm_alpha_shrink = 0.05,
        .tm_foot = 0.05, .tm_knee = 0.75, .tm_limit = 0.97, .tm_gamma = 1.3, .tm_bin_dn = 2.0,
        .tm_peak_attack = 1.0, .tm_peak_hold = 12,
        .tm_lo_slew = 1.5, .tm_fast_frac = 0.15, .tm_fast_gap = 0.25, .tm_hi_attack = 0.15,
        .tm_hi_decay = 0.15, .tm_min_span_dn = 500.0,
        .dde_gain = 2.5, .dde_adapt = 0.5, .dde_target = 0.03, .dde_max = 8.0, .dde_radius = 7,
        .dde_coring_k = 2.0, .dde_limit = 0.2, .warm_boost = 1.5,
        .th_gain = 1.1, .th_size = 7, .th_threshold_dn = 150.0, .th_sigma_k = 25.0,
        .ms_fine = 0.6, .ms_mid = 1.0, .ms_coarse = 0.6, .ms_sigma_1 = 1.0, .ms_sigma_2 = 2.5,
        .ms_sigma_3 = 6.0, .ms_edge = 0.06, .ms_coring_k = 3.0,
        .fps = 25.0, .crf = 10, .x264_preset = "slow", .x264_tune = "grain",
    };
    return p;
}

#endif
