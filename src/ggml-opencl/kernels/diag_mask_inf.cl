#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// diag_mask_inf kernels
//------------------------------------------------------------------------------
kernel void kernel_diag_mask_inf(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int n_past
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i02 = get_global_id(2);
    int i01 = get_global_id(1);
    int i00 = get_global_id(0);

    if (i00 > n_past + i01) {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;
    } else {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];
    }
}

kernel void kernel_diag_mask_inf_8(
        global float4 * src0,
        ulong offset0,
        global float4 * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int n_past
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);

    int i = 2*get_global_id(0);

    int i4 = 4*i;
    int i02 = i4/(ne00*ne01); i4 -= i02*ne00*ne01;
    int i01 = i4/ne00;
    int i00 = i4 - i01*ne00;
    const int limit = n_past + i01;
    const int4 columns = (int4)(i00, i00+1, i00+2, i00+3);
    dst[i] = select(src0[i], (float4)(-INFINITY), columns > limit);
    dst[i+1] = select(src0[i+1], (float4)(-INFINITY), columns + 4 > limit);

}
