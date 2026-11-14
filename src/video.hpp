// H.264 .mp4 writer: 8-bit grey frames are piped to ffmpeg
// (same settings as VideoWriter in project/thermal/stages/video.py).
#pragma once

#include <cstdio>
#include <opencv2/core.hpp>
#include <string>

namespace thermal {

class VideoWriter {
public:
    VideoWriter(std::string path, double fps, int crf, std::string preset, std::string tune);
    ~VideoWriter();
    void write(const cv::Mat& gray8);  // CV_8U, same size every frame
    void close();                      // finishes the file; throws if ffmpeg failed
    int frames() const { return frames_; }
    const std::string& path() const { return path_; }

private:
    std::string path_, preset_, tune_;
    double fps_;
    int crf_;
    FILE* pipe_ = nullptr;
    int w_ = 0, h_ = 0, frames_ = 0;
};

}  // namespace thermal
