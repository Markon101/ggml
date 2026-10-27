# Experimental native Termux Hexagon builds

This branch adds opt-in Android vendor-namespace loading and project-private emulated SDK build tools. ARM64 host code and v79 DSP computation run natively. QEMU executes only the x86 SDK compiler/QAIC/linker on the phone.

The `sphal` loading approach and SDK wrappers were adapted from [OpenCL-S24-Ultra](https://github.com/Ishabdullah/OpenCL-S24-Ultra), under the [included MIT notice](../src/ggml-hexagon/LICENSE-OpenCL-S24-Ultra.txt). Its published hardware was Hexagon v75. The source-built v79 arithmetic and TTS checks here were run independently on an S25 Ultra.

Relevant CMake options:

- `GGML_HEXAGON_USE_SPHAL=ON`: load `/vendor/lib64/libcdsprpc.so` using Android's exported `sphal` namespace. OFF preserves ordinary driver loading.
- `GGML_HEXAGON_ARCHITECTURES=v79`: build only the selected DSP. The default retains all existing architectures.
- `GGML_HEXAGON_QAIC_DIR`: optional root containing `bin/qaic`.
- `GGML_HEXAGON_EMULATION_DIR`: optional root containing the SDK compiler/ar/link wrappers.
- `GGML_HEXAGON_DSP_BUILD_JOBS`: parallel jobs in each DSP external build (default two).

This GGML revision defaults to `GGML_HEXAGON_HOSTBUF=1`. Applications allocating Q8 weights using only the device's ordinary default buffer may then leave matrix products on CPU. The tested application explicitly sets `GGML_HEXAGON_HOSTBUF=0`, uploads canonical Q8 through the backend's set-tensor API, verifies exact get-tensor roundtrips, and checks scheduler placement. Merely selecting HTP is insufficient proof of offload.

Native prerequisites, pinned SDK preparation, matrix/model controls, measurement-only telemetry and actual end-to-end results are in the [application NPU branch](https://github.com/Markon101/qwen3-tts.cpp/blob/npu-v79-bringup/docs/NPU_V79.md). Proprietary SDK files, drivers, model assets and compiled DSP libraries are excluded from these source repositories.

The Q8 activation path also includes a bounded FP32 fallback for finite values above FP16 range whose amax/127 scale remains representable in FP16 (65,504). A captured Qwen3-TTS FFN input reached 157,183.609; converting it to FP16 before quantization created NaNs. The ordinary vector path is retained, and the large-value path normalizes in FP32 before converting quantized values. Source-built default group size 128 was tested with ordinary/large/zero synthetic activations and the real FFN replay; group sizes 32/64 share the helper but were not separately built in this audit.
