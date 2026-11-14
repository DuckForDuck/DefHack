#include "stages.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

#include "blue_noise.hpp"
#include "utils.hpp"

namespace thermal {

namespace {

// Apply fn(y) to every row y in parallel (OpenCV thread pool).
template <class F>
void rows(int n, F fn) {
    cv::parallel_for_(cv::Range(0, n), [&](const cv::Range& r) {
        for (int y = r.start; y < r.end; ++y) fn(y);
    });
}

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// clip((hi - x) / (hi - lo), 0, 1): 1 below lo, 0 above hi
inline float ramp(float x, float lo, float hi) { return clampf((hi - x) / (hi - lo), 0.f, 1.f); }

}  // namespace

// ---------------------------------------------------------------- step 0
void Decode::process(Frame& f) {
    const uint16_t mask = uint16_t((1u << p_.bit_depth) - 1u);
    f.img.create(f.raw.size(), CV_32F);
    rows(f.raw.rows, [&](int y) {
        const uint16_t* r = f.raw.ptr<uint16_t>(y);
        float* o = f.img.ptr<float>(y);
        for (int x = 0; x < f.raw.cols; ++x) o[x] = float(r[x] & mask);
    });
}

// ---------------------------------------------------------------- step 0b
void ShutterGuard::process(Frame& f) {
    // std of every 4th pixel in both directions
    double s1 = 0, s2 = 0;
    size_t n = 0;
    for (int y = 0; y < f.img.rows; y += 4) {
        const float* r = f.img.ptr<float>(y);
        for (int x = 0; x < f.img.cols; x += 4) { s1 += r[x]; s2 += double(r[x]) * r[x]; ++n; }
    }
    const double mean = s1 / double(n);
    const double s = std::sqrt(std::max(s2 / double(n) - mean * mean, 0.0));
    bool frozen = false;
    if (stds_.size() >= 4) {
        std::vector<float> h(stds_.begin(), stds_.end());
        if (s < p_.shutter_ratio * median(h)) {
            ++freeze_;
            if (freeze_ <= p_.shutter_max_freeze) frozen = true;
            else stds_.clear();  // the scene really changed: accept it
        }
    }
    if (!frozen) {
        freeze_ = 0;
        stds_.push_back(s);
        if (int(stds_.size()) > p_.shutter_history) stds_.pop_front();
    }
    f.frozen = frozen;
}

// ---------------------------------------------------------------- step 1
void BadPixels::update_learned(const cv::Mat& img, const cv::Mat& candidates) {
    const float a = float(p_.bp_ema);
    cv::Mat cand_f;
    candidates.convertTo(cand_f, CV_32F, 1.0 / 255.0);
    if (impulse_rate_.empty()) impulse_rate_ = cand_f;
    else impulse_rate_ += a * (cand_f - impulse_rate_);
    if (!prev_.empty()) {
        cv::Mat change = cv::abs(img - prev_);
        if (change_.empty()) change_ = change;
        else change_ += a * (change - change_);
    }
    prev_ = img.clone();
    if (n_ < p_.bp_warmup || change_.empty()) return;
    std::vector<float> all = values(change_);
    const double scene_change = median(all);
    cv::Mat persistent = impulse_rate_ > p_.bp_persist_ratio;
    cv::Mat stuck = scene_change > 0 ? (change_ < p_.bp_stuck_ratio * scene_change)
                                     : cv::Mat::zeros(img.size(), CV_8U);
    learned_ = persistent | stuck;
}

void BadPixels::process(Frame& f) {
    const cv::Mat& img = f.img;
    const int h = img.rows, w = img.cols;
    cv::Mat permanent = learned_.empty() ? cv::Mat::zeros(img.size(), CV_8U) : learned_;
    const bool any_perm = cv::countNonZero(permanent) > 0;
    const double sigma = estimate_noise_sigma(img, any_perm ? permanent : cv::Mat());

    // max and min of the 8 neighbours (centre left out)
    cv::Mat k8 = cv::Mat::ones(3, 3, CV_8U);
    k8.at<uint8_t>(1, 1) = 0;
    cv::Mat mx, mn;
    cv::dilate(img, mx, k8, cv::Point(-1, -1), 1, cv::BORDER_REFLECT);
    cv::erode(img, mn, k8, cv::Point(-1, -1), 1, cv::BORDER_REFLECT);

    const float inv = 1.f / float(std::max(sigma, 0.001));
    cv::Mat candidates(img.size(), CV_8U), impulses(img.size(), CV_8U);
    rows(h, [&](int y) {
        const float* v = img.ptr<float>(y);
        const float* a = mx.ptr<float>(y);
        const float* b = mn.ptr<float>(y);
        const uint8_t* pm = permanent.ptr<uint8_t>(y);
        uint8_t* c = candidates.ptr<uint8_t>(y);
        uint8_t* im = impulses.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            const float e = std::max(v[x] - a[x], b[x] - v[x]) * inv;
            c[x] = (e > p_.bp_k && !pm[x]) ? 255 : 0;
            im[x] = (e > p_.bp_k_replace && !pm[x]) ? 255 : 0;
        }
    });
    update_learned(img, candidates);
    ++n_;

    cv::Mat mask = permanent | impulses;
    cv::Mat out = img.clone();
    std::vector<cv::Point> pts;
    cv::findNonZero(mask, pts);
    if (!pts.empty()) {
        auto refl = [](int i, int n) {  // numpy pad mode "reflect" (edge not repeated)
            if (i < 0) return -i;
            if (i >= n) return 2 * n - 2 - i;
            return i;
        };
        std::vector<cv::Point> left;
        for (const auto& p : pts) {
            bool done = false;
            for (int r = 1; r <= 2 && !done; ++r) {
                std::vector<float> vals;
                for (int dy = -r; dy <= r; ++dy)
                    for (int dx = -r; dx <= r; ++dx) {
                        if (!dy && !dx) continue;
                        const int yy = refl(p.y + dy, h), xx = refl(p.x + dx, w);
                        if (!mask.at<uint8_t>(yy, xx)) vals.push_back(img.at<float>(yy, xx));
                    }
                if (!vals.empty()) {
                    out.at<float>(p) = float(median(vals));
                    done = true;
                }
            }
            if (!done) left.push_back(p);
        }
        if (!left.empty()) {
            std::vector<float> good;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    if (!mask.at<uint8_t>(y, x)) good.push_back(img.at<float>(y, x));
            const float fill = good.empty() ? 0.f : float(median(good));
            for (const auto& p : left) out.at<float>(p) = fill;
        }
    }
    f.img = out;
    f.bad = mask;
    f.noise_sigma = sigma;
}

// ---------------------------------------------------------------- step 2a
void ResidualNUC::process(Frame& f) {
    const cv::Mat x = f.img;
    if (offset_.empty()) offset_ = cv::Mat::zeros(x.size(), CV_32F);
    cv::Mat y = x - offset_;
    const double sigma = f.noise_sigma > 0 ? f.noise_sigma : estimate_noise_sigma(y);
    if (!prev_.empty()) {
        const double motion = cv::mean(cv::abs(x - prev_))[0] / (1.128 * sigma);
        if (motion > p_.nuc_motion_k) {
            // residual (pattern) and its weight: low on scene edges
            cv::Mat smooth = gaussian(y, p_.nuc_blur_sigma);
            cv::Mat gx, gy;
            cv::Sobel(smooth, gx, CV_32F, 1, 0, 3);
            cv::Sobel(smooth, gy, CV_32F, 0, 1, 3);
            const float c = float(p_.nuc_clip_k * sigma);
            const float scale = float(std::max(p_.nuc_edge_k * sigma / p_.nuc_blur_sigma, 0.001));
            const float alpha = float(p_.nuc_alpha);
            rows(x.rows, [&](int r) {
                const float* yr = y.ptr<float>(r);
                const float* sr = smooth.ptr<float>(r);
                const float* ax = gx.ptr<float>(r);
                const float* ay = gy.ptr<float>(r);
                const uint8_t* br = f.bad.empty() ? nullptr : f.bad.ptr<uint8_t>(r);
                float* o = offset_.ptr<float>(r);
                for (int i = 0; i < x.cols; ++i) {
                    const float e = clampf(yr[i] - sr[i], -c, c);
                    const float g1 = ax[i] / 8.f, g2 = ay[i] / 8.f;
                    const float g = std::sqrt(g1 * g1 + g2 * g2) / scale;
                    float w = 1.f / (1.f + g * g);
                    if (br && br[i]) w = 0.f;
                    o[i] += alpha * w * e;
                }
            });
            offset_ -= cv::mean(offset_)[0];
            y = x - offset_;
        }
    }
    prev_ = x.clone();
    f.img = y;
}

// ---------------------------------------------------------------- step 2b
std::vector<float> Destripe::offsets(const cv::Mat& x, int axis, double sigma) const {
    // axis 0: one offset per column; axis 1: one per row
    const int h = x.rows, w = x.cols;
    cv::Mat lp = axis == 0 ? box(x, p_.ds_width, 1) : box(x, 1, p_.ds_width);
    cv::Mat smooth = gaussian(x, p_.ds_edge_sigma);
    cv::Mat grad;
    cv::Sobel(smooth, grad, CV_32F, axis == 0 ? 1 : 0, axis == 0 ? 0 : 1, 3);
    const float c = float(p_.ds_clip_k * sigma);
    const float lim = float(p_.ds_edge_k * sigma);
    const int len = axis == 0 ? w : h;
    std::vector<double> sum(len, 0.0);
    std::vector<int> cnt(len, 0);
    for (int y = 0; y < h; ++y) {
        const float* xr = x.ptr<float>(y);
        const float* lr = lp.ptr<float>(y);
        const float* gr = grad.ptr<float>(y);
        for (int i = 0; i < w; ++i) {
            if (std::fabs(gr[i]) / 8.f < lim) {
                const float hp = clampf(xr[i] - lr[i], -c, c);
                const int k = axis == 0 ? i : y;
                sum[k] += hp;
                cnt[k] += 1;
            }
        }
    }
    const double need = p_.ds_min_valid * (axis == 0 ? h : w);
    std::vector<double> off(len);
    double mean = 0;
    for (int k = 0; k < len; ++k) {
        off[k] = cnt[k] >= need ? sum[k] / std::max(cnt[k], 1) : 0.0;
        mean += off[k];
    }
    mean /= len;
    std::vector<float> out(len);
    for (int k = 0; k < len; ++k) out[k] = float(off[k] - mean);
    return out;
}

void Destripe::process(Frame& f) {
    cv::Mat x = f.img.clone();
    const double sigma = f.noise_sigma > 0 ? f.noise_sigma : estimate_noise_sigma(x);
    const float a = float(p_.ds_temporal_alpha);
    auto smooth = [&](std::vector<float>& old, const std::vector<float>& now) {
        if (a <= 0 || old.size() != now.size()) { old = now; return; }
        for (size_t k = 0; k < now.size(); ++k) old[k] += a * (now[k] - old[k]);
    };
    smooth(col_, offsets(x, 0, sigma));
    for (int y = 0; y < x.rows; ++y) {
        float* r = x.ptr<float>(y);
        for (int i = 0; i < x.cols; ++i) r[i] -= col_[i];
    }
    smooth(row_, offsets(x, 1, sigma));
    for (int y = 0; y < x.rows; ++y) {
        float* r = x.ptr<float>(y);
        const float o = row_[y];
        for (int i = 0; i < x.cols; ++i) r[i] -= o;
    }
    f.img = x;
    const bool any_bad = !f.bad.empty() && cv::countNonZero(f.bad) > 0;
    f.noise_sigma = estimate_noise_sigma(x, any_bad ? f.bad : cv::Mat());
}

// ---------------------------------------------------------------- step 3a
TemporalDenoise::TemporalDenoise(const Params& p)
    : p_(p), dis_(cv::DISOpticalFlow::create(p.tnr_flow_preset)) {}

void TemporalDenoise::process(Frame& f) {
    const cv::Mat x = f.img;
    const int h = x.rows, w = x.cols;
    const double sigma = f.noise_sigma > 0 ? f.noise_sigma : estimate_noise_sigma(x);
    cv::Mat out, n, v;
    if (prev_.empty() || prev_.size() != x.size()) {
        out = x.clone();
        n = cv::Mat::ones(x.size(), CV_32F);
        v = cv::Mat::ones(x.size(), CV_32F);
    } else {
        // optical flow from the current frame to the previous result (8 bit)
        cv::Mat cur_s = gaussian(x, p_.tnr_flow_smooth);
        std::vector<float> sub = subsample(cur_s, 4);
        const auto pc = percentiles(sub, {0.5, 99.5});
        const double lo = pc[0], scale = 255.0 / std::max(pc[1] - pc[0], 0.001);
        auto to8 = [&](const cv::Mat& a) {
            cv::Mat o(a.size(), CV_8U);
            rows(a.rows, [&](int y) {
                const float* s = a.ptr<float>(y);
                uint8_t* d = o.ptr<uint8_t>(y);
                for (int i = 0; i < a.cols; ++i) {
                    double t = (double(s[i]) - lo) * scale;
                    t = t < 0 ? 0 : (t > 255 ? 255 : t);
                    d[i] = uint8_t(t);  // truncation, as numpy astype(uint8)
                }
            });
            return o;
        };
        cv::Mat flow;
        dis_->calc(to8(cur_s), to8(prev_), flow);

        cv::Mat mx(x.size(), CV_32F), my(x.size(), CV_32F);
        rows(h, [&](int y) {
            const cv::Point2f* fl = flow.ptr<cv::Point2f>(y);
            float* a = mx.ptr<float>(y);
            float* b = my.ptr<float>(y);
            for (int i = 0; i < w; ++i) { a[i] = float(i) + fl[i].x; b[i] = float(y) + fl[i].y; }
        });
        cv::Mat p, n_w, v_w;
        cv::remap(prev_, p, mx, my, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        cv::remap(n_, n_w, mx, my, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        cv::remap(v_, v_w, mx, my, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        const float s2base = float(sigma * sigma);
        cv::Mat d(x.size(), CV_32F), s2(x.size(), CV_32F), dd(x.size(), CV_32F);
        rows(h, [&](int y) {
            const float* xr = x.ptr<float>(y);
            const float* pr = p.ptr<float>(y);
            const float* a = mx.ptr<float>(y);
            const float* b = my.ptr<float>(y);
            float* vw = v_w.ptr<float>(y);
            float* nw = n_w.ptr<float>(y);
            float* dr = d.ptr<float>(y);
            float* sr = s2.ptr<float>(y);
            float* q = dd.ptr<float>(y);
            for (int i = 0; i < w; ++i) {
                const bool inside = a[i] >= 0 && a[i] <= w - 1 && b[i] >= 0 && b[i] <= h - 1;
                if (!inside) { nw[i] = 0.f; vw[i] = 1.f; }
                dr[i] = xr[i] - pr[i];
                sr[i] = s2base * (1.f + vw[i]);
                q[i] = dr[i] * dr[i];
            }
        });
        cv::Mat local = box(dd, p_.tnr_window, p_.tnr_window);
        out.create(x.size(), CV_32F);
        n.create(x.size(), CV_32F);
        v.create(x.size(), CV_32F);
        const float maxf = float(p_.tnr_max_frames);
        rows(h, [&](int y) {
            const float* a = mx.ptr<float>(y);
            const float* b = my.ptr<float>(y);
            const float* pr = p.ptr<float>(y);
            const float* dr = d.ptr<float>(y);
            const float* sr = s2.ptr<float>(y);
            const float* lr = local.ptr<float>(y);
            const float* nw = n_w.ptr<float>(y);
            const float* vw = v_w.ptr<float>(y);
            float* o = out.ptr<float>(y);
            float* nn = n.ptr<float>(y);
            float* vv = v.ptr<float>(y);
            for (int i = 0; i < w; ++i) {
                const bool inside = a[i] >= 0 && a[i] <= w - 1 && b[i] >= 0 && b[i] <= h - 1;
                float alpha = ramp(lr[i] / sr[i], float(p_.tnr_local_lo), float(p_.tnr_local_hi)) *
                              ramp(std::fabs(dr[i]) / std::sqrt(sr[i]), float(p_.tnr_pixel_lo),
                                   float(p_.tnr_pixel_hi));
                if (!inside) alpha = 0.f;
                const float nf = std::min(alpha * nw[i] + 1.f, maxf);
                const float gain = 1.f / nf;
                o[i] = pr[i] + gain * dr[i];
                nn[i] = nf;
                vv[i] = (1.f - gain) * (1.f - gain) * vw[i] + gain * gain;
            }
        });
    }
    prev_ = out;
    n_ = n;
    v_ = v;
    cv::Mat sm(x.size(), CV_32F);
    const float sg = float(sigma);
    rows(h, [&](int y) {
        const float* vv = v.ptr<float>(y);
        float* s = sm.ptr<float>(y);
        for (int i = 0; i < w; ++i) s[i] = sg * std::sqrt(vv[i]);
    });
    f.img = out.clone();  // prev_ keeps its own copy (next stages change img)
    f.noise_map = sm;
    f.noise_sigma = sigma;  // replaced by SpatialDenoise (median of the map)
}

// ---------------------------------------------------------------- step 3b
void SpatialDenoise::process(Frame& f) {
    const cv::Mat x = f.img;
    cv::Mat sig = f.noise_map;
    if (sig.empty()) sig = cv::Mat(x.size(), CV_32F, cv::Scalar(f.noise_sigma));
    std::vector<float> all = values(sig);
    const double sigma = median(all);
    const int r = p_.sp_radius;
    cv::Mat m = box(x, r);
    cv::Mat m2 = box(x.mul(x), r);
    cv::Mat a(x.size(), CV_32F), b(x.size(), CV_32F);
    const float st = float(p_.sp_strength);
    rows(x.rows, [&](int y) {
        const float* mr = m.ptr<float>(y);
        const float* qr = m2.ptr<float>(y);
        const float* sr = sig.ptr<float>(y);
        float* ar = a.ptr<float>(y);
        float* br = b.ptr<float>(y);
        for (int i = 0; i < x.cols; ++i) {
            const float var = std::max(qr[i] - mr[i] * mr[i], 0.f);
            const float e = (st * sr[i]) * (st * sr[i]);
            ar[i] = var / (var + e + 1e-6f);
            br[i] = mr[i] - ar[i] * mr[i];
        }
    });
    cv::Mat ab = box(a, r), bb = box(b, r);
    cv::Mat out = ab.mul(x) + bb;
    cv::Scalar mu, sd;
    cv::meanStdDev(out - x, mu, sd);
    const double removed = sd[0] * sd[0];
    const double factor = std::sqrt(std::max(1.0 - removed / std::max(sigma * sigma, 1e-6), 0.1));
    f.img = out;
    f.noise_map = sig * factor;
    f.noise_sigma = sigma * factor;
}

// ---------------------------------------------------------------- step 5
void BaseDetailSplit::process(Frame& f) {
    const cv::Mat x = f.img;
    const double mean = cv::mean(x)[0];
    cv::Mat xc = x - mean;
    const int r = p_.bd_radius;
    const float eps = float(p_.bd_edge_dn * p_.bd_edge_dn);
    cv::Mat m = box(xc, r), m2 = box(xc.mul(xc), r);
    cv::Mat a(x.size(), CV_32F), b(x.size(), CV_32F);
    rows(x.rows, [&](int y) {
        const float* mr = m.ptr<float>(y);
        const float* qr = m2.ptr<float>(y);
        float* ar = a.ptr<float>(y);
        float* br = b.ptr<float>(y);
        for (int i = 0; i < x.cols; ++i) {
            const float var = std::max(qr[i] - mr[i] * mr[i], 0.f);
            ar[i] = var / (var + eps);
            br[i] = mr[i] - ar[i] * mr[i];
        }
    });
    cv::Mat base = box(a, r).mul(xc) + box(b, r) + mean;
    f.base = base;
    f.detail = x - base;
}

// ---------------------------------------------------------------- step 6
double ToneMap::anchored_lo(const std::vector<float>& sub, double lo_t) {
    // black point moves at most lo_slew DN per frame, except on a real scene change
    const double old = lo_;
    const double span = std::max(hi_ - old, 1.0);
    bool fast;
    if (lo_t < old) {
        size_t below = 0;
        for (float v : sub) below += v < old;
        fast = double(below) / double(sub.size()) > p_.tm_fast_frac;
    } else {
        fast = (lo_t - old) / span > p_.tm_fast_gap;
    }
    if (fast) {
        const double a = lo_t < old ? p_.tm_alpha_grow : p_.tm_alpha_shrink;
        return old + a * (lo_t - old);
    }
    return old + std::clamp(lo_t - old, -p_.tm_lo_slew, p_.tm_lo_slew);
}

void ToneMap::update_stats(const cv::Mat& base) {
    std::vector<float> sub = subsample(base, 2);
    const std::vector<float> keep = sub;  // percentiles reorder sub
    const auto pc = percentiles(sub, {p_.tm_p_low, p_.tm_p_high});
    const double lo = pc[0], hi = pc[1];
    float smax = keep[0], smin = keep[0];
    for (float v : keep) { smax = std::max(smax, v); smin = std::min(smin, v); }

    if (first_) {
        lo_ = lo;
        hi_ = hi_hold_ = hi;
    } else {
        lo_ = anchored_lo(keep, lo);
        if (hi > hi_hold_) hi_hold_ += p_.tm_hi_attack * (hi - hi_hold_);
        else hi_hold_ = std::max(hi, hi_hold_ - p_.tm_hi_decay);
        hi_ = hi_hold_;
    }
    if (p_.tm_min_span_dn > 0 && hi_ - lo_ < p_.tm_min_span_dn) hi_ = lo_ + p_.tm_min_span_dn;

    // peak hold of the frame max / min (scale of the knee and the foot)
    auto peak = [&](double& old, double now, int& hold, bool up) {
        if (first_) { old = now; hold = p_.tm_peak_hold; return; }
        if (up ? now > old : now < old) { old += p_.tm_peak_attack * (now - old); hold = p_.tm_peak_hold; }
        else if (hold > 0) { --hold; }
        else { old += p_.tm_alpha_shrink * (now - old); hold = 0; }
    };
    peak(max_, smax, hold_max_, true);
    peak(min_, smin, hold_min_, false);
    if (hi_ - lo_ < 4 * p_.tm_bin_dn) {
        const double c = 0.5 * (hi_ + lo_);
        lo_ = c - 2 * p_.tm_bin_dn;
        hi_ = c + 2 * p_.tm_bin_dn;
    }
    first_ = false;
}

void ToneMap::process(Frame& f) {
    const cv::Mat x = f.img;
    update_stats(f.base);

    // curve on the grid of 2-DN bins from lo to hi: linear, then gamma
    const double bin = p_.tm_bin_dn;
    const int i0 = int(std::floor(lo_ / bin)), i1 = int(std::ceil(hi_ / bin));
    const int nseg = i1 + 1 - i0;
    const double e0 = i0 * bin, e1 = (i1 + 1) * bin;
    std::vector<double> curve(nseg + 1);
    for (int j = 0; j <= nseg; ++j) curve[j] = std::pow(double(j) / nseg, p_.tm_gamma);
    const double fo = p_.tm_foot, k = p_.tm_knee;
    const double s_hi = std::max((max_ - e1) / 2.0, 4 * bin);
    const double s_lo = std::max((e0 - min_) / 2.0, 4 * bin);
    const double lin_slope = (k - fo) / (hi_ - lo_);

    // DDE: detail with its own gain (noise gate + adaptive local gain)
    const cv::Mat& d = f.detail;
    cv::Mat d2 = d.mul(d);
    cv::Mat local = box(d2, p_.dde_radius);
    cv::Mat g(x.size(), CV_32F);
    cv::Mat y(x.size(), CV_32F);
    const float g_fix = float(p_.dde_gain * lin_slope);
    const float g_cap = float(p_.dde_max * lin_slope);
    const float ad = float(p_.dde_adapt), tgt = float(p_.dde_target);
    const float ck = float(p_.dde_coring_k), lim = float(p_.dde_limit), wb = float(p_.warm_boost);
    const float limit = float(p_.tm_limit), span = 1.f - limit;
    auto soft_top = [&](float v) {
        return v > limit ? limit + span * (1.f - std::exp(-(v - limit) / span)) : v;
    };
    rows(x.rows, [&](int r) {
        const float* br = f.base.ptr<float>(r);
        const float* dr = d.ptr<float>(r);
        const float* qr = d2.ptr<float>(r);
        const float* lr = local.ptr<float>(r);
        const float* nr = f.noise_map.ptr<float>(r);
        float* gr = g.ptr<float>(r);
        float* yr = y.ptr<float>(r);
        for (int i = 0; i < x.cols; ++i) {
            // base -> 0..1
            const double v = br[i];
            double t;
            if (v > e1) {
                t = k + (1 - k) * (1 - std::exp(-(v - e1) / s_hi));
            } else if (v < e0) {
                t = fo * std::exp(-(e0 - v) / s_lo);
            } else {
                const double pos = (v - e0) / bin;
                int j = int(pos);
                double c;
                if (j >= nseg) c = curve[nseg];
                else {
                    const double fr = (v - (e0 + j * bin)) / bin;
                    c = curve[j] + (curve[j + 1] - curve[j]) * fr;
                }
                t = c * (k - fo) + fo;
            }
            // detail
            const float nz = ck * nr[i];
            const float w = qr[i] / (qr[i] + nz * nz + 1e-6f);
            const float n2 = 2.f * nr[i];
            const float rms = std::sqrt(lr[i] + n2 * n2);
            const float gl = std::min(tgt / rms, g_cap);
            const float gg = (1.f - ad) * g_fix + ad * gl;
            float add = gg * w * dr[i];
            if (add > 0) add *= wb;
            add = add / (1.f + std::fabs(add) / lim);
            gr[i] = gg;
            // soft limits towards 1 and 0
            float out = soft_top(float(t + add));
            out = 1.f - soft_top(1.f - out);
            yr[i] = out;
        }
    });
    f.linear = x;
    f.img = y;
    f.slope = g;
    f.lin_slope = lin_slope;
}

// ---------------------------------------------------------------- step 6c
void CompactHotBoost::process(Frame& f) {
    const cv::Mat& x = f.linear;
    const int h = x.rows, w = x.cols;
    const int k = p_.th_size | 1;
    cv::Mat xs, op, th;
    cv::resize(x, xs, cv::Size(w / 2, h / 2), 0, 0, cv::INTER_AREA);
    cv::morphologyEx(xs, op, cv::MORPH_OPEN, cv::Mat::ones(k, k, CV_8U));
    cv::resize(xs - op, th, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    const float thr = float(std::max(p_.th_threshold_dn, p_.th_sigma_k * f.noise_sigma));
    const float gain = float(p_.th_gain * f.lin_slope);
    cv::Mat add(x.size(), CV_32F);
    rows(h, [&](int r) {
        const float* t = th.ptr<float>(r);
        float* a = add.ptr<float>(r);
        for (int i = 0; i < w; ++i) a[i] = gain * std::max(t[i] - thr, 0.f);
    });
    limit_add_inplace(f.img, add);
}

// ---------------------------------------------------------------- step 7b
namespace {
double band_noise_factor(double s_lo, double s_hi) {
    auto energy = [](double s) { return s <= 0 ? 1.0 : 1.0 / (4.0 * M_PI * s * s); };
    auto cross = [](double a, double b) {
        return a <= 0 ? 1.0 / (2.0 * M_PI * b * b) : 1.0 / (2.0 * M_PI * (a * a + b * b));
    };
    const double v = energy(s_lo) + energy(s_hi) - 2.0 * cross(s_lo, s_hi);
    return std::sqrt(std::max(v, 1e-6));
}
}  // namespace

void MultiScaleDetail::process(Frame& f) {
    cv::Mat& y = f.img;
    cv::Mat g1 = gaussian(y, p_.ms_sigma_1), g2 = gaussian(y, p_.ms_sigma_2), g3 = gaussian(y, p_.ms_sigma_3);
    const float nf[3] = {float(band_noise_factor(0.0, p_.ms_sigma_1)),
                         float(band_noise_factor(p_.ms_sigma_1, p_.ms_sigma_2)),
                         float(band_noise_factor(p_.ms_sigma_2, p_.ms_sigma_3))};
    const float gains[3] = {float(p_.ms_fine), float(p_.ms_mid), float(p_.ms_coarse)};
    const float e2 = float(p_.ms_edge * p_.ms_edge), ck = float(p_.ms_coring_k);
    const bool have_noise = !f.noise_map.empty() && !f.slope.empty();
    cv::Mat add(y.size(), CV_32F);
    rows(y.rows, [&](int r) {
        const float* yr = y.ptr<float>(r);
        const float* a = g1.ptr<float>(r);
        const float* b = g2.ptr<float>(r);
        const float* c = g3.ptr<float>(r);
        const float* nr = have_noise ? f.noise_map.ptr<float>(r) : nullptr;
        const float* sr = have_noise ? f.slope.ptr<float>(r) : nullptr;
        float* o = add.ptr<float>(r);
        for (int i = 0; i < y.cols; ++i) {
            const float bands[3] = {yr[i] - a[i], a[i] - b[i], b[i] - c[i]};
            const float noise = have_noise ? nr[i] * sr[i] : 0.f;
            float s = 0.f;
            for (int q = 0; q < 3; ++q) {
                if (gains[q] <= 0) continue;
                const float bb = bands[q] * bands[q];
                float w = 1.f;
                if (have_noise) {
                    const float n = ck * noise * nf[q];
                    w = bb / (bb + n * n + 1e-12f);
                }
                s += gains[q] * bands[q] * w * e2 / (e2 + bb);
            }
            o[i] = s;
        }
    });
    limit_add_inplace(y, add);
}

// ---------------------------------------------------------------- last display step
void ShutterHold::process(Frame& f) {
    if (f.frozen) {
        f.img = last_.empty() ? cv::Mat(f.raw.size(), CV_32F, cv::Scalar(0.5)) : last_.clone();
        return;
    }
    last_ = f.img.clone();
}

// ---------------------------------------------------------------- step 9
Quantise::Quantise() {}

void Quantise::process(Frame& f) {
    const cv::Mat& y = f.img;
    if (threshold_.size() != y.size()) {
        threshold_.create(y.size(), CV_32F);
        for (int r = 0; r < y.rows; ++r)
            for (int c = 0; c < y.cols; ++c)
                threshold_.at<float>(r, c) =
                    (kBlueNoiseRanks[(r % kBlueNoiseSize) * kBlueNoiseSize + c % kBlueNoiseSize] + 0.5f) /
                    float(kBlueNoiseSize * kBlueNoiseSize);
    }
    cv::Mat out(y.size(), CV_8U);
    rows(y.rows, [&](int r) {
        const float* yr = y.ptr<float>(r);
        const float* tr = threshold_.ptr<float>(r);
        uint8_t* o = out.ptr<uint8_t>(r);
        for (int c = 0; c < y.cols; ++c) {
            const float v = clampf(yr[c], 0.f, 1.f) * 255.f;
            float q;
            if (v <= 0.f) q = 0.f;
            else if (v >= 255.f) q = 255.f;
            else q = std::floor(v + 0.5f + (tr[c] - 0.5f));
            o[c] = uint8_t(clampf(q, 0.f, 255.f));
        }
    });
    f.img = out;
}

// ---------------------------------------------------------------- pipeline
std::vector<std::unique_ptr<Stage>> build_pipeline(const Params& p) {
    std::vector<std::unique_ptr<Stage>> s;
    s.emplace_back(new Decode(p));
    s.emplace_back(new ShutterGuard(p));
    s.emplace_back(new BadPixels(p));
    s.emplace_back(new ResidualNUC(p));
    s.emplace_back(new Destripe(p));
    s.emplace_back(new TemporalDenoise(p));
    s.emplace_back(new SpatialDenoise(p));
    s.emplace_back(new BaseDetailSplit(p));
    s.emplace_back(new ToneMap(p));
    s.emplace_back(new CompactHotBoost(p));
    s.emplace_back(new MultiScaleDetail(p));
    s.emplace_back(new ShutterHold());
    s.emplace_back(new Quantise());
    return s;
}

// ---------------------------------------------------------------- naive (script.py)
cv::Mat naive_agc(const cv::Mat& raw16) {
    std::vector<float> v;
    v.reserve(raw16.total());
    for (int y = 0; y < raw16.rows; ++y) {
        const uint16_t* r = raw16.ptr<uint16_t>(y);
        for (int x = 0; x < raw16.cols; ++x) v.push_back(float(r[x]));
    }
    const auto pc = percentiles(v, {2.0, 98.0});
    const double vmin = pc[0], vmax = pc[1];
    const double scale = vmax > vmin ? 255.0 / (vmax - vmin) : 0.0;
    cv::Mat out(raw16.size(), CV_8U);
    for (int y = 0; y < raw16.rows; ++y) {
        const uint16_t* r = raw16.ptr<uint16_t>(y);
        uint8_t* o = out.ptr<uint8_t>(y);
        for (int x = 0; x < raw16.cols; ++x) {
            double c = std::min(std::max(double(r[x]), vmin), vmax);
            o[x] = uint8_t((c - vmin) * scale);  // truncation, as astype(np.uint8)
        }
    }
    return out;
}

}  // namespace thermal
