#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_conv_transpose_1d_f16_f32(
    global const half  * src0,
    ulong                offset0,
    global const float * src1,
    ulong                offset1,
    global float       * dst,
    ulong                offsetd,
    int s0,
    int p0,
    int d0,
    int output_size,
    int src0_ne0,
    int src0_ne1,
    int src0_ne2,
    int src1_ne0,
    int dst_ne0,
    int nb01_el,
    int nb02_el,
    int nb11_el,
    int nbd1_el
) {
    src0 = (global const half*)((global const char*)src0 + offset0);
    src1 = (global const float*)((global const char*)src1 + offset1);
    dst  = (global float*)((global char*)dst + offsetd);

    int global_index = get_global_id(0);
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;
    int idx = global_index % dst_ne0;

    int i_min = (idx >= src0_ne0) ? (idx - src0_ne0 + s0) / s0 : 0;
    int i_max = min(src1_ne0 - 1, idx / s0);

    float accumulator = 0.0f;

    for (int c = 0; c < src0_ne2; c++) {
        int kernel_offset = c * nb02_el + out_index * nb01_el;
        int input_offset  = c * nb11_el;

        for (int i = i_min; i <= i_max; i++) {
            int weight_idx = idx - i * s0;
            float kernel_weight = (float)src0[kernel_offset + weight_idx];
            float input_value   = src1[input_offset + i];
            accumulator += kernel_weight * input_value;
        }
    }

    dst[out_index * nbd1_el + idx] = accumulator;
}

kernel void kernel_conv_transpose_1d_f32_f32(
    global const float * src0,
    ulong                offset0,
    global const float * src1,
    ulong                offset1,
    global float       * dst,
    ulong                offsetd,
    int s0,
    int p0,
    int d0,
    int output_size,
    int src0_ne0,
    int src0_ne1,
    int src0_ne2,
    int src1_ne0,
    int dst_ne0,
    int nb01_el,
    int nb02_el,
    int nb11_el,
    int nbd1_el
) {
    src0 = (global const float*)((global const char*)src0 + offset0);
    src1 = (global const float*)((global const char*)src1 + offset1);
    dst  = (global float*)((global char*)dst + offsetd);

    int global_index = get_global_id(0);
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;
    int idx = global_index % dst_ne0;

    int i_min = (idx >= src0_ne0) ? (idx - src0_ne0 + s0) / s0 : 0;
    int i_max = min(src1_ne0 - 1, idx / s0);

    float accumulator = 0.0f;

    for (int c = 0; c < src0_ne2; c++) {
        int kernel_offset = c * nb02_el + out_index * nb01_el;
        int input_offset  = c * nb11_el;

        for (int i = i_min; i <= i_max; i++) {
            int weight_idx = idx - i * s0;
            float kernel_weight = src0[kernel_offset + weight_idx];
            float input_value   = src1[input_offset + i];
            accumulator += kernel_weight * input_value;
        }
    }

    dst[out_index * nbd1_el + idx] = accumulator;
}
