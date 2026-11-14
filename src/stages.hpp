// The processing stages, in the order of the pipeline.
// Each class is the C++ version of a Python module in project/thermal/stages/
// with the values of preset michael-features (params.hpp).
#pragma once

#include <deque>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>
#include <vector>

#include "params.hpp"

namespace thermal {

// Everything that travels with one frame through the pipeline
// (FrameData in the Python version).
struct Frame {
    int index = 0;
    cv::Mat raw;          // CV_16U, as read from the sensor
    cv::Mat img;          // CV_32F: DN (steps 0-5), 0..1 (steps 6-7); CV_8U after Quantise
    cv::Mat bad;          // CV_8U, 255 = bad pixel (step 1)
    cv::Mat noise_map;    // CV_32F, remaining noise std per pixel, DN (step 3)
    cv::Mat base, detail; // CV_32F, step 5
    cv::Mat linear;       // CV_32F, image before tone mapping (step 6)
    cv::Mat slope;        // CV_32F, display units per DN of detail (step 6)
    double noise_sigma = 0.0;  // std of the random noise of img, DN
    double lin_slope = 0.0;    // display units per DN of the tone map
    bool frozen = false;       // shutter / FFC frame (step 0b)
};

class Stage {
public:
    virtual ~Stage() = default;
    virtual const char* name() const = 0;
    virtual void process(Frame& f) = 0;
    // true = also runs on a frozen (shutter) frame
    virtual bool run_when_frozen() const { return false; }
};

// step 0: keep the 14 valid bits, convert to float32
class Decode : public Stage {
public:
    explicit Decode(const Params& p) : p_(p) {}
    const char* name() const override { return "Decode"; }
    void process(Frame& f) override;
private:
    const Params& p_;
};

// step 0b: mark flat shutter / FFC frames as frozen
class ShutterGuard : public Stage {
public:
    explicit ShutterGuard(const Params& p) : p_(p) {}
    const char* name() const override { return "ShutterGuard"; }
    void process(Frame& f) override;
private:
    const Params& p_;
    std::deque<double> stds_;
    int freeze_ = 0;
};

// step 1: impulse test per frame + learned mask, median replacement
class BadPixels : public Stage {
public:
    explicit BadPixels(const Params& p) : p_(p) {}
    const char* name() const override { return "BadPixels"; }
    void process(Frame& f) override;
private:
    void update_learned(const cv::Mat& img, const cv::Mat& impulses);
    const Params& p_;
    int n_ = 0;
    cv::Mat prev_, impulse_rate_, change_, learned_;
};

// step 2a: per-pixel offset (residual NUC), learned while the camera moves
class ResidualNUC : public Stage {
public:
    explicit ResidualNUC(const Params& p) : p_(p) {}
    const char* name() const override { return "ResidualNUC"; }
    void process(Frame& f) override;
private:
    const Params& p_;
    cv::Mat offset_, prev_;
};

// step 2b: column and row stripes, smoothed over time
class Destripe : public Stage {
public:
    explicit Destripe(const Params& p) : p_(p) {}
    const char* name() const override { return "Destripe"; }
    void process(Frame& f) override;
private:
    std::vector<float> offsets(const cv::Mat& x, int axis, double sigma) const;
    const Params& p_;
    std::vector<float> col_, row_;
};

// step 3a: motion-compensated temporal averaging (DIS optical flow)
class TemporalDenoise : public Stage {
public:
    explicit TemporalDenoise(const Params& p);
    const char* name() const override { return "TemporalDenoise"; }
    void process(Frame& f) override;
private:
    const Params& p_;
    cv::Ptr<cv::DISOpticalFlow> dis_;
    cv::Mat prev_, n_, v_;
};

// step 3b: guided filter, strength from the noise map
class SpatialDenoise : public Stage {
public:
    explicit SpatialDenoise(const Params& p) : p_(p) {}
    const char* name() const override { return "SpatialDenoise"; }
    void process(Frame& f) override;
private:
    const Params& p_;
};

// step 5: base (large areas) + detail (texture)
class BaseDetailSplit : public Stage {
public:
    explicit BaseDetailSplit(const Params& p) : p_(p) {}
    const char* name() const override { return "BaseDetailSplit"; }
    void process(Frame& f) override;
private:
    const Params& p_;
};

// step 6: DN -> 0..1 (anchored range, linear curve + gamma, knee, foot) + DDE detail
class ToneMap : public Stage {
public:
    explicit ToneMap(const Params& p) : p_(p) {}
    const char* name() const override { return "ToneMap"; }
    void process(Frame& f) override;
private:
    void update_stats(const cv::Mat& base);
    double anchored_lo(const std::vector<float>& sub, double lo_t);
    const Params& p_;
    bool first_ = true;
    double lo_ = 0, hi_ = 0, hi_hold_ = 0, min_ = 0, max_ = 0;
    int hold_max_ = 0, hold_min_ = 0;
};

// step 6c: compact hot objects brighter (white top-hat), soft approach to white
class CompactHotBoost : public Stage {
public:
    explicit CompactHotBoost(const Params& p) : p_(p) {}
    const char* name() const override { return "CompactHotBoost"; }
    void process(Frame& f) override;
private:
    const Params& p_;
};

// step 7b: texture on three scales, noise gate, no halos
class MultiScaleDetail : public Stage {
public:
    explicit MultiScaleDetail(const Params& p) : p_(p) {}
    const char* name() const override { return "MultiScaleDetail"; }
    void process(Frame& f) override;
private:
    const Params& p_;
};

// last display step: frozen frame -> last good image
class ShutterHold : public Stage {
public:
    const char* name() const override { return "ShutterHold"; }
    bool run_when_frozen() const override { return true; }
    void process(Frame& f) override;
private:
    cv::Mat last_;
};

// step 9: 0..1 -> uint8 with a blue-noise dither
class Quantise : public Stage {
public:
    Quantise();
    const char* name() const override { return "Quantise"; }
    bool run_when_frozen() const override { return true; }
    void process(Frame& f) override;
private:
    cv::Mat threshold_;  // tiled dither thresholds, made on the first frame
};

// The whole processing chain of preset michael-features (steps 0-9).
std::vector<std::unique_ptr<Stage>> build_pipeline(const Params& p);

// The original script.py: percentiles 2 / 98 of the raw frame, clip, 8 bit.
cv::Mat naive_agc(const cv::Mat& raw16);

}  // namespace thermal
