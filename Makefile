# thermal_isp (C) - build with `make`; needs only a C11 compiler and pthreads
# (ffmpeg is needed at run time to write the .mp4).
#
#   make                 # optimised for this machine (-march/-mcpu=native)
#   make PORTABLE=1      # without -march=native (binary for other machines)
#   make clean

CC      ?= cc
TARGET  := thermal_isp
SRC     := $(wildcard src/*.c)
OBJ     := $(patsubst src/%.c,obj/%.o,$(SRC))

# -O3 turns on the loop vectoriser; -fno-math-errno / -fno-trapping-math let it
# vectorise sqrt and divisions. No -ffast-math: results stay IEEE-exact.
CFLAGS  ?= -O3
CFLAGS  += -std=gnu11 -Wall -Wextra -Wno-unused-parameter -Wno-psabi -fno-math-errno -fno-trapping-math -pthread
LDLIBS  += -lm -pthread

# -march/-mcpu=native if the compiler supports it (NEON is always there on
# arm64, so Apple Silicon is vectorised even without it)
ifndef PORTABLE
  ifneq ($(filter arm64 aarch64,$(shell uname -m)),)
    NATIVE_FLAG := -mcpu=native
  else
    NATIVE_FLAG := -march=native
  endif
  CFLAGS += $(shell echo 'int main(void){return 0;}' | $(CC) $(NATIVE_FLAG) -x c - -o /dev/null >/dev/null 2>&1 && echo $(NATIVE_FLAG))
endif

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

obj/%.o: src/%.c src/*.h | obj
	$(CC) $(CFLAGS) -c -o $@ $<

obj:
	mkdir -p obj

clean:
	rm -rf obj $(TARGET)

.PHONY: clean
