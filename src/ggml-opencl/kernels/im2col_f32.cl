#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_im2col_f32(
        global const float * src1,
        ulong offset1,
        global float       * dst,
        ulong offsetd,
        ulong batch_offset,
        ulong delta_offset,
        long IW,
        long IH,
        long IC,
        long OW,
        long OH,
        long KW,
        long KH,
        long total_elements,
        long CHW,
        int  s0,
        int  s1,
        int  p0,
        int  p1,
        int  d0,
        int  d1
) {
    long gid = get_global_id(0);
    if (gid >= total_elements) {
        return;
    }

    src1 = (global const float*)((global const char*)src1 + offset1);
    dst  = (global float*)((global char*)dst + offsetd);

    // Fast path for 1D convolution (OH == 1 && KH == 1)
    if (OH == 1 && KH == 1) {
        int chw = (int)(gid % CHW);
        int spatial = (int)(gid / CHW);
        int ix  = spatial % (int)OW;
        int batch_idx = spatial / (int)OW;
        int ic  = chw / (int)KW;
        int kx  = chw % (int)KW;

        int iiw = ix * s0 + kx * d0 - p0;
        if (iiw < 0 || iiw >= (int)IW) {
            dst[gid] = 0.0f;
        } else {
            ulong offset_src = (ulong)ic * delta_offset + (ulong)batch_idx * batch_offset;
            dst[gid] = src1[offset_src + iiw];
        }
        return;
    }

    // General 2D path with coalesced destination writes
    long chw = gid % CHW;
    long spatial = gid / CHW;
    long ix = spatial % OW;
    long batch_oh = spatial / OW;
    long oh = batch_oh % OH;
    long batch = batch_oh / OH;

    long kw_kh = KW * KH;
    long ic = chw / kw_kh;
    long k_rem = chw % kw_kh;
    long ky = k_rem / KW;
    long kx = k_rem % KW;

    long iiw = ix * s0 + kx * d0 - p0;
    long iih = oh * s1 + ky * d1 - p1;

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        dst[gid] = 0.0f;
    } else {
        ulong offset_src = (ulong)ic * delta_offset + (ulong)batch * batch_offset;
        dst[gid] = src1[offset_src + iih * IW + iiw];
    }
}
