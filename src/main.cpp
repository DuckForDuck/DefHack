// thermal_isp – raw 14-bit thermal frames -> 8-bit white-hot H.264 video.
// C++ version of project/ (Python) with preset michael-features.
//
//   thermal_isp INPUT.bin [--out DIR] [--compare] [--frames N] [--no-video]
//               [--width 640] [--height 512] [--fps 25] [--profile]
//               [--raw-out FILE] [--flow medium|fast|ultrafast]
//
// INPUT is read frame by frame (streaming); "-" reads from stdin, so a camera
// stream can be piped in. Each frame is processed using only this frame and
// the state kept from earlier frames.
//
// --compare : writes a side-by-side video instead: our result (left) vs the
//             original script.py (right: percentiles 2/98, clip, 8 bit).
// --no-video: process only (speed test); --profile prints time per stage.
// --raw-out : also writes the 8-bit frames as raw bytes (for comparisons).
// --flow    : DIS optical-flow preset; medium = as in michael-features (default),
//             fast / ultrafast = quicker but less exact motion (not in the preset).

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "params.hpp"
#include "stages.hpp"
#include "video.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace thermal;

namespace {

struct Options {
    std::string input;
    std::string out_dir = "output";
    std::string raw_out;
    bool compare = false, no_video = false, profile = false;
    int width = 640, height = 512, frames = 0;
    double fps = 25.0;
    int flow = 2;
};

void usage() {
    std::cerr << "usage: thermal_isp INPUT.bin|- [--out DIR] [--compare] [--frames N] [--no-video]\n"
                 "                   [--width 640] [--height 512] [--fps 25] [--profile] [--raw-out FILE]\n"
                 "                   [--flow medium|fast|ultrafast]\n";
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--out") o.out_dir = next();
        else if (a == "--compare") o.compare = true;
        else if (a == "--no-video") o.no_video = true;
        else if (a == "--profile") o.profile = true;
        else if (a == "--frames") o.frames = std::stoi(next());
        else if (a == "--width") o.width = std::stoi(next());
        else if (a == "--height") o.height = std::stoi(next());
        else if (a == "--fps") o.fps = std::stod(next());
        else if (a == "--raw-out") o.raw_out = next();
        else if (a == "--flow") {
            const std::string v = next();
            if (v == "medium") o.flow = 2;
            else if (v == "fast") o.flow = 1;
            else if (v == "ultrafast") o.flow = 0;
            else { std::cerr << "--flow must be medium, fast or ultrafast\n"; std::exit(2); }
        }
        else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        else if (!a.empty() && a[0] == '-' && a != "-") { std::cerr << "unknown option " << a << "\n"; usage(); std::exit(2); }
        else o.input = a;
    }
    if (o.input.empty()) { usage(); std::exit(2); }
    return o;
}

// Panel with a title bar (same layout as Compare in the Python version).
cv::Mat panel(const cv::Mat& img8, const std::string& title) {
    cv::Mat head(44, img8.cols, CV_8U, cv::Scalar(32));
    const double black = 100.0 * double(cv::countNonZero(img8 == 0)) / double(img8.total());
    const double white = 100.0 * double(cv::countNonZero(img8 == 255)) / double(img8.total());
    char stats[96];
    std::snprintf(stats, sizeof stats, "black %.2f%%  white %.2f%%", black, white);
    cv::putText(head, title, {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv::LINE_AA);
    cv::putText(head, stats, {8, 37}, cv::FONT_HERSHEY_SIMPLEX, 0.45, 200, 1, cv::LINE_AA);
    cv::Mat out;
    cv::vconcat(head, img8, out);
    return out;
}

cv::Mat compose(const cv::Mat& ours, const cv::Mat& naive, int index) {
    cv::Mat left = panel(ours, "ours (C++, preset michael-features)");
    cv::Mat right = panel(naive, "naive (script.py)");
    cv::Mat sep(left.rows, 6, CV_8U, cv::Scalar(255));
    cv::Mat body;
    cv::hconcat(std::vector<cv::Mat>{left, sep, right}, body);
    cv::Mat foot(26, body.cols, CV_8U, cv::Scalar(32));
    cv::putText(foot, "frame #" + std::to_string(index), {8, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5, 255, 1,
                cv::LINE_AA);
    cv::Mat out;
    cv::vconcat(body, foot, out);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    Params p;
    p.fps = opt.fps;
    p.tnr_flow_preset = opt.flow;

    std::ifstream file;
    std::istream* in = &std::cin;
    if (opt.input != "-") {
        file.open(opt.input, std::ios::binary);
        if (!file) { std::cerr << "cannot open " << opt.input << "\n"; return 1; }
        in = &file;
    }
    const std::string stem = opt.input == "-" ? "stdin" : fs::path(opt.input).stem().string();

    std::unique_ptr<VideoWriter> video;
    if (!opt.no_video) {
        fs::create_directories(opt.out_dir);
        const std::string name = stem + (opt.compare ? "__compare__cpp.mp4" : "__michael-features__cpp.mp4");
        video = std::make_unique<VideoWriter>((fs::path(opt.out_dir) / name).string(), p.fps, p.crf,
                                              p.x264_preset, p.x264_tune);
    }
    std::ofstream raw_out;
    if (!opt.raw_out.empty()) raw_out.open(opt.raw_out, std::ios::binary);

    auto stages = build_pipeline(p);
    std::map<std::string, double> stage_ms;
    std::vector<double> frame_ms;
    const size_t frame_bytes = size_t(opt.width) * size_t(opt.height) * 2;
    std::vector<char> buf(frame_bytes);

    int index = 0;
    while (opt.frames <= 0 || index < opt.frames) {
        in->read(buf.data(), std::streamsize(frame_bytes));
        if (in->gcount() != std::streamsize(frame_bytes)) break;  // end of stream

        Frame f;
        f.index = index;
        f.raw = cv::Mat(opt.height, opt.width, CV_16U);
        std::memcpy(f.raw.data, buf.data(), frame_bytes);  // little-endian uint16, as recorded

        const auto t0 = Clock::now();
        for (auto& s : stages) {
            if (f.frozen && !s->run_when_frozen()) continue;  // shutter frame: no state update
            const auto ts = Clock::now();
            s->process(f);
            stage_ms[s->name()] += std::chrono::duration<double, std::milli>(Clock::now() - ts).count();
        }
        frame_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());

        if (raw_out) raw_out.write(reinterpret_cast<const char*>(f.img.data), std::streamsize(f.img.total()));
        if (video) {
            if (opt.compare) video->write(compose(f.img, naive_agc(f.raw), index));
            else video->write(f.img);
        }
        ++index;
    }
    if (video) {
        video->close();
        std::cout << "wrote " << video->frames() << " frames -> " << video->path() << "\n";
    }

    if (frame_ms.size() > 1) {
        std::vector<double> t(frame_ms.begin() + 1, frame_ms.end());  // frame 0 = warm-up
        double mean = 0;
        for (double v : t) mean += v;
        mean /= double(t.size());
        std::sort(t.begin(), t.end());
        const double p95 = t[size_t(0.95 * double(t.size() - 1))];
        std::printf("frames: %d   processing per frame: mean %.1f ms, p95 %.1f ms, max %.1f ms\n", index, mean,
                    p95, t.back());
        std::printf("=> %.1f frames per second (%.1f fps in the slowest 5 %% of frames)\n", 1000.0 / mean,
                    1000.0 / p95);
        if (opt.profile) {
            double total = 0;
            for (auto& kv : stage_ms) total += kv.second;
            std::vector<std::pair<std::string, double>> v(stage_ms.begin(), stage_ms.end());
            std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
            std::printf("\nper stage (ms per frame, share):\n");
            for (auto& [name, ms] : v)
                std::printf("  %-18s %7.2f ms  %5.1f %%\n", name.c_str(), ms / index, 100.0 * ms / total);
        }
    }
    return 0;
}
