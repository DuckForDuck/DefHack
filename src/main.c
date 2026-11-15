/* thermal_isp (C) - raw 14-bit thermal frames -> 8-bit white-hot H.264 video.
 * C version of project/ (Python) and cpp/ (C++) with preset michael-features.
 *
 *   thermal_isp INPUT.bin|- [--out DIR] [--compare] [--frames N] [--no-video]
 *               [--width 640] [--height 512] [--fps 25] [--profile]
 *               [--raw-out FILE] [--flow medium|fast|ultrafast] [--threads N]
 *
 * INPUT is read frame by frame (streaming); "-" reads from stdin, so a camera
 * stream can be piped in. Each frame is processed using only this frame and
 * the state kept from earlier frames.
 *
 * --compare : writes a side-by-side video instead: our result (left) vs the
 *             original script.py (right: percentiles 2/98, clip, 8 bit).
 * --no-video: process only (speed test); --profile prints time per stage.
 * --raw-out : also writes the 8-bit frames as raw bytes (for comparisons).
 * --flow    : DIS optical-flow preset; medium = as in michael-features (default),
 *             fast / ultrafast = quicker but less exact motion (not in the preset).
 * --threads : worker threads. Default: all CPU cores with --no-video, half of
 *             them when a video is written (ffmpeg's x264 needs the rest).
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "common.h"
#include "font_data.h"
#include "parallel.h"
#include "params.h"
#include "stages.h"
#include "video.h"

typedef struct {
    const char *input, *out_dir, *raw_out;
    int compare, no_video, profile;
    int width, height, frames, flow, threads;
    double fps;
} Options;

static void usage(void) {
    fprintf(stderr,
            "usage: thermal_isp INPUT.bin|- [--out DIR] [--compare] [--frames N] [--no-video]\n"
            "                   [--width 640] [--height 512] [--fps 25] [--profile] [--raw-out FILE]\n"
            "                   [--flow medium|fast|ultrafast] [--threads N]\n");
}

static Options parse(int argc, char **argv) {
    Options o = {NULL, "output", NULL, 0, 0, 0, 640, 512, 0, 2, 0, 25.0};
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (usage(), exit(2), (char *)NULL))
        if (!strcmp(a, "--out")) o.out_dir = NEXT();
        else if (!strcmp(a, "--compare")) o.compare = 1;
        else if (!strcmp(a, "--no-video")) o.no_video = 1;
        else if (!strcmp(a, "--profile")) o.profile = 1;
        else if (!strcmp(a, "--frames")) o.frames = atoi(NEXT());
        else if (!strcmp(a, "--width")) o.width = atoi(NEXT());
        else if (!strcmp(a, "--height")) o.height = atoi(NEXT());
        else if (!strcmp(a, "--fps")) o.fps = atof(NEXT());
        else if (!strcmp(a, "--raw-out")) o.raw_out = NEXT();
        else if (!strcmp(a, "--threads")) o.threads = atoi(NEXT());
        else if (!strcmp(a, "--flow")) {
            const char *v = NEXT();
            if (!strcmp(v, "medium")) o.flow = 2;
            else if (!strcmp(v, "fast")) o.flow = 1;
            else if (!strcmp(v, "ultrafast")) o.flow = 0;
            else { fprintf(stderr, "--flow must be medium, fast or ultrafast\n"); exit(2); }
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); exit(0); }
        else if (a[0] == '-' && a[1] != '\0') { fprintf(stderr, "unknown option %s\n", a); usage(); exit(2); }
        else o.input = a;
#undef NEXT
    }
    if (!o.input || o.width < 16 || o.height < 16) { usage(); exit(2); }
    return o;
}

static void make_dirs(const char *path) {
    char buf[4096];
    snprintf(buf, sizeof buf, "%s", path);
    for (char *p = buf + 1; *p; ++p)
        if (*p == '/') { *p = 0; mkdir(buf, 0755); *p = '/'; }
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) perror(buf);
}

static void file_stem(const char *path, char *out, size_t n) {
    const char *b = strrchr(path, '/');
    b = b ? b + 1 : path;
    snprintf(out, n, "%s", b);
    char *dot = strrchr(out, '.');
    if (dot && dot != out) *dot = 0;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* ---------------------------------------------------------------- compare frame */

typedef struct {
    uint8_t *px;
    int w, h;
} Canvas;

static void fill(Canvas *c, int x0, int y0, int w, int h, uint8_t v) {
    for (int y = y0; y < y0 + h; ++y) memset(c->px + (size_t)y * c->w + x0, v, (size_t)w);
}

static void text(Canvas *c, const Font *f, int x, int baseline, const char *s, uint8_t color) {
    const int top = baseline - f->ascent;
    for (; *s; ++s) {
        const int ch = (unsigned char)*s;
        if (ch < 32 || ch > 126) continue;
        const Glyph *g = &f->g[ch - 32];
        for (int gy = 0; gy < f->height; ++gy) {
            const int y = top + gy;
            if (y < 0 || y >= c->h) continue;
            for (int gx = 0; gx < g->w; ++gx) {
                const int xx = x + gx;
                if (xx < 0 || xx >= c->w) continue;
                const int a = f->px[g->off + gy * g->w + gx];
                if (!a) continue;
                uint8_t *d = c->px + (size_t)y * c->w + xx;
                *d = (uint8_t)((*d * (255 - a) + color * a + 127) / 255);
            }
        }
        x += g->adv;
    }
}

/* panel with a title bar (same layout as Compare in the Python version) */
static void panel(Canvas *c, int x0, const uint8_t *img, int w, int h, const char *title) {
    fill(c, x0, 0, w, 44, 32);
    size_t black = 0, white = 0;
    for (size_t i = 0; i < (size_t)w * h; ++i) { black += img[i] == 0; white += img[i] == 255; }
    char stats[96];
    snprintf(stats, sizeof stats, "black %.2f%%  white %.2f%%", 100.0 * black / ((double)w * h),
             100.0 * white / ((double)w * h));
    text(c, &font_big, x0 + 8, 18, title, 255);
    text(c, &font_small, x0 + 8, 37, stats, 200);
    for (int y = 0; y < h; ++y) memcpy(c->px + (size_t)(44 + y) * c->w + x0, img + (size_t)y * w, (size_t)w);
}

static void compose(Canvas *c, const uint8_t *ours, const uint8_t *naive, int w, int h, int index) {
    panel(c, 0, ours, w, h, "ours (C, preset michael-features)");
    fill(c, w, 0, 6, 44 + h, 255);
    panel(c, w + 6, naive, w, h, "naive (script.py)");
    fill(c, 0, 44 + h, c->w, 26, 32);
    char foot[64];
    snprintf(foot, sizeof foot, "frame #%d", index);
    text(c, &font_big, 8, 44 + h + 18, foot, 255);
}

/* ---------------------------------------------------------------- main */

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
    const Options opt = parse(argc, argv);
    Params p = params_default();
    p.fps = opt.fps;
    p.tnr_flow_preset = opt.flow;
    int threads = opt.threads;
    if (threads <= 0) {
        const long cores = sysconf(_SC_NPROCESSORS_ONLN);
        threads = cores > 0 ? (int)cores : 1;
        if (!opt.no_video && threads > 1) threads /= 2; /* leave cores to the encoder */
    }
    pool_init(threads);

    FILE *in = stdin;
    if (strcmp(opt.input, "-") != 0) {
        in = fopen(opt.input, "rb");
        if (!in) { fprintf(stderr, "cannot open %s\n", opt.input); return 1; }
    }
    char stem[1024];
    if (!strcmp(opt.input, "-")) snprintf(stem, sizeof stem, "stdin");
    else file_stem(opt.input, stem, sizeof stem);

    const int W = opt.width, H = opt.height;
    VideoWriter *video = NULL;
    char out_path[8192] = "";
    if (!opt.no_video) {
        make_dirs(opt.out_dir);
        snprintf(out_path, sizeof out_path, "%s/%s%s", opt.out_dir, stem,
                 opt.compare ? "__compare__c.mp4" : "__michael-features__c.mp4");
        video = video_open(out_path, p.fps, p.crf, p.x264_preset, p.x264_tune);
    }
    FILE *raw_out = NULL;
    if (opt.raw_out) {
        raw_out = fopen(opt.raw_out, "wb");
        if (!raw_out) { fprintf(stderr, "cannot write %s\n", opt.raw_out); return 1; }
    }

    Pipeline *pl = pipeline_create(&p, W, H);
    const size_t n = (size_t)W * H;
    uint16_t *raw = ALLOC(uint16_t, n);
    uint8_t *out8 = ALLOC(uint8_t, n), *naive8 = ALLOC(uint8_t, n);
    float *scratch = ALLOC(float, n);
    Canvas canvas = {NULL, 2 * W + 6, 44 + H + 26};
    if (opt.compare) canvas.px = ALLOC(uint8_t, (size_t)canvas.w * canvas.h);

    double stage_ms[ST_COUNT] = {0};
    int cap = 1024, index = 0, failed = 0;
    double *frame_ms = malloc(sizeof(double) * (size_t)cap);

    while (opt.frames <= 0 || index < opt.frames) {
        /* little-endian uint16, as recorded */
        if (fread(raw, sizeof(uint16_t), n, in) != n) break; /* end of stream */
        const double t0 = now_ms();
        pipeline_process(pl, raw, out8, stage_ms);
        if (index == cap) { cap *= 2; frame_ms = realloc(frame_ms, sizeof(double) * (size_t)cap); }
        frame_ms[index] = now_ms() - t0;

        if (raw_out) fwrite(out8, 1, n, raw_out);
        if (video && !failed) {
            if (opt.compare) {
                naive_agc(raw, W, H, naive8, scratch);
                compose(&canvas, out8, naive8, W, H, index);
                failed = video_write(video, canvas.px, canvas.w, canvas.h) != 0;
            } else {
                failed = video_write(video, out8, W, H) != 0;
            }
        }
        ++index;
    }
    if (in != stdin) fclose(in);
    if (raw_out) fclose(raw_out);
    int rc = 0;
    if (video) {
        const int frames = video_frames(video);
        if (video_close(video) != 0 || failed) rc = 1;
        else printf("wrote %d frames -> %s\n", frames, out_path);
    }

    if (index > 1) {
        const int m = index - 1; /* frame 0 = warm-up */
        double *t = frame_ms + 1, mean = 0;
        for (int i = 0; i < m; ++i) mean += t[i];
        mean /= m;
        qsort(t, (size_t)m, sizeof(double), cmp_double);
        const double p95 = t[(int)(0.95 * (m - 1))];
        printf("frames: %d   threads: %d   processing per frame: mean %.1f ms, p95 %.1f ms, max %.1f ms\n",
               index, pool_threads(), mean, p95, t[m - 1]);
        printf("=> %.1f frames per second (%.1f fps in the slowest 5 %% of frames)\n", 1000.0 / mean, 1000.0 / p95);
        if (opt.profile) {
            double total = 0;
            int order[ST_COUNT];
            for (int i = 0; i < ST_COUNT; ++i) { total += stage_ms[i]; order[i] = i; }
            for (int i = 0; i < ST_COUNT; ++i)
                for (int j = i + 1; j < ST_COUNT; ++j)
                    if (stage_ms[order[j]] > stage_ms[order[i]]) { int s = order[i]; order[i] = order[j]; order[j] = s; }
            printf("\nper stage (ms per frame, share):\n");
            for (int i = 0; i < ST_COUNT; ++i)
                printf("  %-18s %7.2f ms  %5.1f %%\n", kStageNames[order[i]], stage_ms[order[i]] / index,
                       100.0 * stage_ms[order[i]] / total);
        }
    }
    free(frame_ms);
    pipeline_free(pl);
    xfree(raw); xfree(out8); xfree(naive8); xfree(scratch);
    if (canvas.px) xfree(canvas.px);
    pool_shutdown();
    return rc;
}
