/* H.264 .mp4 writer: 8-bit grey frames are piped to ffmpeg
 * (same settings as VideoWriter in the Python and C++ versions). */
#ifndef THERMAL_VIDEO_H
#define THERMAL_VIDEO_H

#include <stdint.h>

typedef struct VideoWriter VideoWriter;

VideoWriter *video_open(const char *path, double fps, int crf, const char *preset, const char *tune);
/* returns 0 on success */
int video_write(VideoWriter *v, const uint8_t *gray8, int w, int h);
/* finishes the file; returns 0 if ffmpeg succeeded */
int video_close(VideoWriter *v);
int video_frames(const VideoWriter *v);

#endif
