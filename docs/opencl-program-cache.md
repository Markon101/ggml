# Optional OpenCL program-binary cache

Set `GGML_OPENCL_BINARY_CACHE_DIR` to an existing, writable directory to cache compiled OpenCL programs. The default remains source compilation. `GGML_OPENCL_BINARY_CACHE_DEBUG=1` prints hit/miss records.

Cache identity contains the full kernel source, compiler options, device/vendor identity and driver/OpenCL versions. The FNV filename is a lookup key: full identity comparison prevents a filename-hash collision from selecting a different program. Binary creation/build errors, invalid/truncated headers and unusable paths fall back to source. Successful single-device program binaries are written through a temporary file and renamed into place.

This is the local fork's setting; do not confuse it with similarly named upstream cache options. Kernel/option/driver changes invalidate the corresponding entry automatically. Keep cache files private to the executing user.

On the tested Adreno 830 setup, an initial same-route cache reuse reduced process readiness from 9.33 to 2.91 seconds. This is a startup observation, not a warm-inference or universal speed claim. A deliberately truncated header recovered through source compilation and produced finite, non-silent output.
