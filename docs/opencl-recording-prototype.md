# Experimental QCOM graph recordings

`GGML_OPENCL_QCOM_RECORD=1` enables the research prototype for eligible predictor-sized graphs. It is disabled by default and is not enabled by the application's optimized launchers.

The driver supported recording/replay and argument updates in a synthetic sequence test. That test improved dispatch time by roughly threefold for 160 tiny kernels. The generic graph-cache implementation did not establish an end-to-end TTS gain; the synthetic ratio must not be presented as a synthesis speedup.

The cache serializes graph operations, metadata, allocation identities, source associations and offsets, retains referenced CL buffers, limits the number of entries, and clears recordings when the backend is released. Broader graph lifetime, allocation reuse, concurrent execution and error-path validation remain research work. This is retained as an explicitly gated experiment, not a production recommendation.
