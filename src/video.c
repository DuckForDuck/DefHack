#include "video.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct VideoWriter {
    char path[4096], preset[64], tune[64];
    double fps;
    int crf, w, h, frames;
    FILE *pipe;
};

VideoWriter *video_open(const char *path, double fps, int crf, const char *preset, const char *tune) {
    VideoWriter *v = calloc(1, sizeof *v);
    if (!v) return NULL;
    snprintf(v->path, sizeof v->path, "%s", path);
    snprintf(v->preset, sizeof v->preset, "%s", preset);
    snprintf(v->tune, sizeof v->tune, "%s", tune ? tune : "");
    v->fps = fps;
    v->crf = crf;
    return v;
}

int video_write(VideoWriter *v, const uint8_t *img, int w, int h) {
    if (!v->pipe) {
        v->w = w;
        v->h = h;
        char tune[96] = "";
        if (v->tune[0]) snprintf(tune, sizeof tune, "-tune %s ", v->tune);
        char cmd[8192];
        /* full range 0..255 kept end to end (no TV range 16..235) */
        snprintf(cmd, sizeof cmd,
                 "ffmpeg -y -hide_banner -loglevel error -f rawvideo -pix_fmt gray -s %dx%d -r %g "
                 "-color_range pc -i - %s-vf scale=in_range=full:out_range=full,format=yuv420p -c:v libx264 "
                 "-preset %s -crf %d -profile:v high -color_range pc -x264-params fullrange=on "
                 "-movflags +faststart -tag:v avc1 \"%s\"",
                 w, h, v->fps, tune, v->preset, v->crf, v->path);
        v->pipe = popen(cmd, "w");
        if (!v->pipe) {
            fprintf(stderr, "cannot start ffmpeg (install it, e.g. 'brew install ffmpeg')\n");
            return -1;
        }
    }
    if (w != v->w || h != v->h) {
        fprintf(stderr, "frame size changed\n");
        return -1;
    }
    if (fwrite(img, 1, (size_t)w * h, v->pipe) != (size_t)w * h) {
        fprintf(stderr, "ffmpeg stopped accepting frames\n");
        return -1;
    }
    ++v->frames;
    return 0;
}

int video_close(VideoWriter *v) {
    if (!v) return 0;
    int rc = 0;
    if (v->pipe) {
        rc = pclose(v->pipe);
        if (rc != 0) fprintf(stderr, "ffmpeg failed while writing %s\n", v->path);
    }
    free(v);
    return rc;
}

int video_frames(const VideoWriter *v) { return v ? v->frames : 0; }
