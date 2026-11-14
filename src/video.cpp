#include "video.hpp"

#include <sstream>
#include <stdexcept>

namespace thermal {

VideoWriter::VideoWriter(std::string path, double fps, int crf, std::string preset, std::string tune)
    : path_(std::move(path)), preset_(std::move(preset)), tune_(std::move(tune)), fps_(fps), crf_(crf) {}

VideoWriter::~VideoWriter() {
    try { close(); } catch (...) {}
}

void VideoWriter::write(const cv::Mat& img) {
    if (img.type() != CV_8U) throw std::runtime_error("VideoWriter needs 8-bit grey frames");
    if (!pipe_) {
        w_ = img.cols;
        h_ = img.rows;
        std::ostringstream cmd;
        // full range 0..255 kept end to end (no TV range 16..235)
        cmd << "ffmpeg -y -hide_banner -loglevel error -f rawvideo -pix_fmt gray -s " << w_ << "x" << h_
            << " -r " << fps_ << " -color_range pc -i - "
            << (tune_.empty() ? "" : "-tune " + tune_ + " ")
            << "-vf scale=in_range=full:out_range=full,format=yuv420p -c:v libx264 -preset " << preset_
            << " -crf " << crf_
            << " -profile:v high -color_range pc -x264-params fullrange=on -movflags +faststart -tag:v avc1 \""
            << path_ << "\"";
        pipe_ = popen(cmd.str().c_str(), "w");
        if (!pipe_) throw std::runtime_error("cannot start ffmpeg (install it, e.g. 'brew install ffmpeg')");
    }
    if (img.cols != w_ || img.rows != h_) throw std::runtime_error("frame size changed");
    for (int y = 0; y < img.rows; ++y) {
        if (fwrite(img.ptr<uint8_t>(y), 1, size_t(img.cols), pipe_) != size_t(img.cols))
            throw std::runtime_error("ffmpeg stopped accepting frames");
    }
    ++frames_;
}

void VideoWriter::close() {
    if (!pipe_) return;
    const int rc = pclose(pipe_);
    pipe_ = nullptr;
    if (rc != 0) throw std::runtime_error("ffmpeg failed while writing " + path_);
}

}  // namespace thermal
