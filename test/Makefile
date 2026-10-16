# Run the unit tests with no CMake, no CUDA, no TensorRT, no OpenCV and no
# network access:
#
#     make -C test
#
# Everything here is plain host C++. That is a deliberate property of the
# codebase, not a coincidence - the geometry maths, the detector decoding, the
# command shaping and the whole balance-loop logic are all written free of
# hardware dependencies so that they can be tested on any machine, including
# one whose GPU toolchain is still half-installed.
#
# `make stub-check` additionally type-checks the two files that DO depend on
# TensorRT, against the stub header in test/stubs/. That needs OpenCV dev
# headers and CUDA headers; see the note on that target below.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
INCLUDES  = -I../src -I../mcu -I../include

TESTS = test_preprocess_math test_postprocess test_command_shaper test_balance_core

.PHONY: all run clean stub-check
all: run

run: $(TESTS)
	@echo
	@fail=0; for t in $(TESTS); do \
	    echo "================ $$t ================"; \
	    ./$$t || fail=1; \
	    echo; \
	done; \
	if [ $$fail -ne 0 ]; then echo "SOME TESTS FAILED"; exit 1; fi; \
	echo "all suites passed"

test_preprocess_math: test_preprocess_math.cpp ../src/preprocess_math.cuh test_util.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ test_preprocess_math.cpp

test_postprocess: test_postprocess.cpp ../src/postprocess.cpp ../src/postprocess.h test_util.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ test_postprocess.cpp ../src/postprocess.cpp

test_command_shaper: test_command_shaper.cpp ../src/command_shaper.cpp ../src/command_shaper.h test_util.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ test_command_shaper.cpp ../src/command_shaper.cpp

test_balance_core: test_balance_core.cpp ../mcu/balance_core.h test_util.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ test_balance_core.cpp

# ---------------------------------------------------------------------------
# Type-check the TensorRT-dependent sources without TensorRT installed.
#
# Compiles src/trt_infer.cpp and src/main.cpp against test/stubs/NvInfer.h,
# which declares only the TensorRT symbols this project uses. It catches typos,
# wrong argument types and missing overrides. It does NOT prove conformance
# with the real SDK - if a signature in the stub is wrong, this passes and the
# real build fails. The real build is the authority.
#
# Needs OpenCV development headers and the CUDA runtime headers on the include
# path; skips itself with a message if pkg-config cannot find OpenCV.
# ---------------------------------------------------------------------------
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
CUDA_INC      := $(shell dirname $$(dirname $$(readlink -f $$(command -v nvcc 2>/dev/null) 2>/dev/null) 2>/dev/null) 2>/dev/null)/include

stub-check:
	@if [ -z "$(OPENCV_CFLAGS)" ]; then \
	    echo "skipping stub-check: pkg-config cannot find opencv4"; \
	    exit 0; \
	fi
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Istubs -I$(CUDA_INC) $(OPENCV_CFLAGS) \
	    -fsyntax-only ../src/trt_infer.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Istubs -I$(CUDA_INC) $(OPENCV_CFLAGS) \
	    -fsyntax-only ../src/main.cpp
	@echo "stub-check: trt_infer.cpp and main.cpp type-check"

clean:
	rm -f $(TESTS)
