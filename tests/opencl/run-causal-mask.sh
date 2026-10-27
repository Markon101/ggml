#!/data/data/com.termux/files/usr/bin/sh
set -eu
TEST_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GGML_DIR=$(CDPATH= cd -- "$TEST_DIR/../.." && pwd)
TEST_BUILD_DIR=${TEST_BUILD_DIR:-"$GGML_DIR/build-opencl-regression"}
mkdir -p "$TEST_BUILD_DIR"
"${CXX:-c++}" -O2 -std=c++17 "$TEST_DIR/test-causal-mask.cpp" -lOpenCL -o "$TEST_BUILD_DIR/test-causal-mask"
if [ -f /vendor/lib64/libOpenCL.so ]; then
  export OCL_ICD_VENDORS=${OCL_ICD_VENDORS:-/vendor/lib64/libOpenCL.so}
  export LD_LIBRARY_PATH="/system/lib64:/vendor/lib64:${LD_LIBRARY_PATH:-}"
fi
exec "$TEST_BUILD_DIR/test-causal-mask" "$TEST_DIR/diag-mask-buggy-fixture.cl" "$GGML_DIR/src/ggml-opencl/kernels/diag_mask_inf.cl"
