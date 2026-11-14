// Small helpers shared by the stages (same maths as thermal/utils.py).
#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace thermal {

// numpy.percentile(..., method="linear") of the values in v (v is reordered).
double percentile(std::vector<float>& v, double p);
// Several percentiles of the same values (p ascending).
std::vector<double> percentiles(std::vector<float>& v, const std::vector<double>& ps);
// numpy.median (mean of the two middle values for an even count).
double median(std::vector<float>& v);

// Values img[y0::step, x0::step] (rows/cols < y_end/x_end) as a flat vector.
std::vector<float> subsample(const cv::Mat& img, int step, int y0 = 0, int x0 = 0,
                             int y_end = -1, int x_end = -1);
// All values of a float matrix.
std::vector<float> values(const cv::Mat& img);

// Robust std of the random noise: MAD of a 4-neighbour Laplacian on every
// 2nd pixel, pixels where mask != 0 left out (empty mask = none).
double estimate_noise_sigma(const cv::Mat& img, const cv::Mat& mask = cv::Mat());

// Gaussian blur, border "reflect" as in the Python code.
cv::Mat gaussian(const cv::Mat& img, double sigma);
// Box mean (2r+1)x(2r+1), border "reflect".
cv::Mat box(const cv::Mat& img, int r);
// Box mean of size kx x ky, border "reflect".
cv::Mat box(const cv::Mat& img, int kx, int ky);

// y + add, approaching 0 and 1 smoothly (limit_add in multiscale.py).
void limit_add_inplace(cv::Mat& y, const cv::Mat& add);

}  // namespace thermal
