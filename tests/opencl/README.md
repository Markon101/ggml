# OpenCL causal-mask regression

Run `sh tests/opencl/run-causal-mask.sh` from a Termux installation with C++17 and OpenCL headers/loader installed. This executes GPU kernels; run it only when ready for device validation.

The test compares a historical buggy fixture and the current shader with a CPU-computed causal mask. It covers widths 8/16/24/32/64, past lengths 0/3/17, seven rows, three heads, nonzero offsets and guard regions. A successful test requires at least one failure from the buggy control and zero incorrect values or guard writes from the fixed shader.

The same computation previously passed all 15 fixed cases on an Adreno 830; the old shader failed 13 cases and corrupted guards in 11. The host source here adapts file paths for a standalone checkout. Its execution is intentionally separate from compilation and has not been rerun during publication preparation.
