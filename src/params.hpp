// Parameters of the pipeline = preset presets/michael-features.json of the
// Python version (project/). Each value names its preset key.
// Only the modules that are ON in that preset are implemented.
#pragma once

namespace thermal {

struct Params {
    // step 0 – Decode
    int bit_depth = 14;

    // step 0b – ShutterGuard (shutter / FFC frames)
    double shutter_ratio = 0.35;     // ShutterGuard: std < ratio x median of history
    int shutter_max_freeze = 30;     // max. frozen frames in a row
    int shutter_history = 16;        // past normal frames for the reference std

    // step 1 – BadPixels
    double bp_k = 8.0;               // candidate (learned mask), in noise sigmas
    double bp_k_replace = 15.0;      // replaced at once, in noise sigmas
    double bp_persist_ratio = 0.5;   // share of frames as impulse -> permanent
    double bp_ema = 0.05;            // speed of the learned statistics
    double bp_stuck_ratio = 0.05;    // temporal change < 5 % of median -> stuck
    int bp_warmup = 30;              // frames before the learned mask is used

    // step 2a – ResidualNUC (online only: no calibration in advance)
    double nuc_alpha = 0.02;
    double nuc_blur_sigma = 2.0;
    double nuc_clip_k = 5.0;
    double nuc_edge_k = 3.0;
    double nuc_motion_k = 3.0;

    // step 2b – Destripe
    int ds_width = 9;
    double ds_clip_k = 2.0;
    double ds_edge_k = 0.5;
    double ds_edge_sigma = 1.5;
    double ds_min_valid = 0.2;
    double ds_temporal_alpha = 0.1;  // Destripe.temporal_alpha

    // step 3a – TemporalDenoise
    int tnr_max_frames = 8;          // TemporalDenoise.max_frames
    int tnr_window = 5;
    double tnr_local_lo = 1.6, tnr_local_hi = 3.0;
    double tnr_pixel_lo = 3.0, tnr_pixel_hi = 5.0;   // TemporalDenoise.pixel_hi
    double tnr_flow_smooth = 1.0;
    // DIS optical-flow preset: 2 = medium (as in the preset), 1 = fast, 0 = ultrafast.
    // fast/ultrafast are NOT in michael-features: quicker, less exact motion.
    int tnr_flow_preset = 2;

    // step 3b – SpatialDenoise (guided)
    double sp_strength = 1.5;        // SpatialDenoise.strength
    int sp_radius = 2;

    // step 5 – BaseDetailSplit
    int bd_radius = 8;               // BaseDetailSplit.radius
    double bd_edge_dn = 25.0;        // BaseDetailSplit.edge_dn

    // step 6 – ToneMap
    double tm_p_low = 0.2, tm_p_high = 99.8;
    double tm_alpha_grow = 0.15, tm_alpha_shrink = 0.05;
    double tm_foot = 0.05, tm_knee = 0.75;
    double tm_limit = 0.97;
    double tm_gamma = 1.3;
    double tm_bin_dn = 2.0;
    double tm_peak_attack = 1.0;
    int tm_peak_hold = 12;
    // anchored black point / white peak hold (ideas from Michael's thermal_isp)
    double tm_lo_slew = 1.5;
    double tm_fast_frac = 0.15;
    double tm_fast_gap = 0.25;
    double tm_hi_attack = 0.15;
    double tm_hi_decay = 0.15;
    double tm_min_span_dn = 500.0;
    // DDE detail
    double dde_gain = 2.5;
    double dde_adapt = 0.5;
    double dde_target = 0.03;
    double dde_max = 8.0;
    int dde_radius = 7;
    double dde_coring_k = 2.0;
    double dde_limit = 0.2;
    double warm_boost = 1.5;
    // (use_plateau = false -> linear curve; use_knee = true; temporal = true;
    //  white = 1.0; shadow_lift = 0)

    // step 6c – CompactHotBoost (white top-hat)
    double th_gain = 1.1;
    int th_size = 7;
    double th_threshold_dn = 150.0;
    double th_sigma_k = 25.0;        // soft = true

    // step 7b – MultiScaleDetail
    double ms_fine = 0.6, ms_mid = 1.0, ms_coarse = 0.6;
    double ms_sigma_1 = 1.0, ms_sigma_2 = 2.5, ms_sigma_3 = 6.0;
    double ms_edge = 0.06;
    double ms_coring_k = 3.0;

    // step 10 – video
    double fps = 25.0;
    int crf = 10;
    const char* x264_preset = "slow";
    const char* x264_tune = "grain";
};

}  // namespace thermal
