#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace thermal {

std::vector<double> percentiles(std::vector<float>& v, const std::vector<double>& ps) {
    std::vector<double> out;
    const size_t n = v.size();
    if (n == 0) return std::vector<double>(ps.size(), 0.0);
    for (double p : ps) {
        const double idx = p / 100.0 * double(n - 1);
        const size_t lo = size_t(std::floor(idx));
        const double t = idx - double(lo);
        std::nth_element(v.begin(), v.begin() + lo, v.end());
        const double a = v[lo];
        double b = a;
        if (lo + 1 < n) b = *std::min_element(v.begin() + lo + 1, v.end());
        // numpy's lerp (exact at both ends)
        const double d = b - a;
        out.push_back(t >= 0.5 ? b - d * (1.0 - t) : a + d * t);
    }
    return out;
}

double percentile(std::vector<float>& v, double p) { return percentiles(v, {p})[0]; }

double median(std::vector<float>& v) {
    const size_t n = v.size();
    if (n == 0) return 0.0;
    const size_t m = n / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    const double hi = v[m];
    if (n % 2) return hi;
    const double lo = *std::max_element(v.begin(), v.begin() + m);
    return 0.5 * (lo + hi);
}

std::vector<float> subsample(const cv::Mat& img, int step, int y0, int x0, int y_end, int x_end) {
    if (y_end < 0) y_end = img.rows;
    if (x_end < 0) x_end = img.cols;
    std::vector<float> out;
    out.reserve(size_t((y_end - y0 + step - 1) / step) * size_t((x_end - x0 + step - 1) / step));
    for (int y = y0; y < y_end; y += step) {
        const float* r = img.ptr<float>(y);
        for (int x = x0; x < x_end; x += step) out.push_back(r[x]);
    }
    return out;
}

std::vector<float> values(const cv::Mat& img) {
    std::vector<float> out;
    out.reserve(img.total());
    for (int y = 0; y < img.rows; ++y) {
        const float* r = img.ptr<float>(y);
        out.insert(out.end(), r, r + img.cols);
    }
    return out;
}

double estimate_noise_sigma(const cv::Mat& img, const cv::Mat& mask) {
    // lap = x - 0.25 * (4 neighbours), only at [1:-1:2, 1:-1:2]
    std::vector<float> lap;
    lap.reserve(size_t(img.rows / 2) * size_t(img.cols / 2));
    for (int y = 1; y < img.rows - 1; y += 2) {
        const float* up = img.ptr<float>(y - 1);
        const float* r = img.ptr<float>(y);
        const float* dn = img.ptr<float>(y + 1);
        const uint8_t* m = mask.empty() ? nullptr : mask.ptr<uint8_t>(y);
        for (int x = 1; x < img.cols - 1; x += 2) {
            if (m && m[x]) continue;
            lap.push_back(r[x] - 0.25f * up[x] - 0.25f * dn[x] - 0.25f * r[x - 1] - 0.25f * r[x + 1]);
        }
    }
    if (lap.empty()) return 0.0;
    std::vector<float> tmp = lap;
    const double med = median(tmp);
    for (auto& v : lap) v = std::fabs(float(v - med));
    const double mad = median(lap);
    return 1.4826 * mad / std::sqrt(1.25);
}

cv::Mat gaussian(const cv::Mat& img, double sigma) {
    cv::Mat out;
    cv::GaussianBlur(img, out, cv::Size(0, 0), sigma, 0, cv::BORDER_REFLECT);
    return out;
}

cv::Mat box(const cv::Mat& img, int r) { return box(img, 2 * r + 1, 2 * r + 1); }

cv::Mat box(const cv::Mat& img, int kx, int ky) {
    cv::Mat out;
    cv::blur(img, out, cv::Size(kx, ky), cv::Point(-1, -1), cv::BORDER_REFLECT);
    return out;
}

void limit_add_inplace(cv::Mat& y, const cv::Mat& add) {
    const float eps = 1e-6f;
    cv::parallel_for_(cv::Range(0, y.rows), [&](const cv::Range& rr) {
        for (int i = rr.start; i < rr.end; ++i) {
            float* yr = y.ptr<float>(i);
            const float* ar = add.ptr<float>(i);
            for (int x = 0; x < y.cols; ++x) {
                const float v = yr[x], a = ar[x];
                yr[x] = a >= 0 ? v + a * (1.f - v) / (1.f - v + a + eps)
                               : v + a * v / (v - a + eps);
            }
        }
    });
}

}  // namespace thermal
