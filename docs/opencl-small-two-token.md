# Optional Q8 two-token kernel

`GGML_OPENCL_Q8_SMALL2=1` enables an Adreno-specific kernel for eligible two-column Q8 matrix multiplications. It reuses the existing FP16 activation-texture conversion and preserves the original accumulation precision. Other shapes retain the existing dispatch.

The final FP16-texture variant matched twenty paired matrix cases across four tested shapes. On a small conditioned 0.6B TTS screen, saved speech codes and PCM matched the corrected GPU control, with a 4.2% median synthesis-time reduction. The earlier faster F32-input prototype changed real output and is not included as the selected shader.

This is shape- and workload-specific evidence. The option is disabled by default in the backend. The application fork enables it only in its optional 0.6B GPU profile; 1.7B seeded request variability prevented a claim of exact parity for that screen.
