#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define N_F16_F16 4

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f16(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3)
{
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);
    int rb = get_group_id(1)*N_F16_F16;
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    global half * x = (global half *) (src0 + offset_src0);
    int lid = get_sub_group_local_id();
    int sg_size = get_max_sub_group_size();
    ulong off_base = (ulong)i12*nb12 + (ulong)i13*nb13;

    if (rb + 3 < ne11) {
        // Fast path: all 4 rows are in bounds, compute 4 rows simultaneously in scalar registers
        int r1_0 = rb + 0;
        int r1_1 = rb + 1;
        int r1_2 = rb + 2;
        int r1_3 = rb + 3;

        if (ne00 < 128) {
            global half * y_0 = (global half *)(src1 + r1_0*nb11 + off_base);
            global half * y_1 = (global half *)(src1 + r1_1*nb11 + off_base);
            global half * y_2 = (global half *)(src1 + r1_2*nb11 + off_base);
            global half * y_3 = (global half *)(src1 + r1_3*nb11 + off_base);

            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;

            for (int i = lid; i < ne00; i += sg_size) {
                half xi = x[i];
                sum0 += (half)xi * y_0[i];
                sum1 += (half)xi * y_1[i];
                sum2 += (half)xi * y_2[i];
                sum3 += (half)xi * y_3[i];
            }

            float all0 = sub_group_reduce_add(sum0);
            float all1 = sub_group_reduce_add(sum1);
            float all2 = sub_group_reduce_add(sum2);
            float all3 = sub_group_reduce_add(sum3);

            if (lid == 0) {
                dst[im*ne1*ne0 + r1_0*ne0 + r0] = all0;
                dst[im*ne1*ne0 + r1_1*ne0 + r0] = all1;
                dst[im*ne1*ne0 + r1_2*ne0 + r0] = all2;
                dst[im*ne1*ne0 + r1_3*ne0 + r0] = all3;
            }
        } else {
            global half4 * x4 = (global half4 *)x;
            global half4 * y4_0 = (global half4 *)(src1 + r1_0*nb11 + off_base);
            global half4 * y4_1 = (global half4 *)(src1 + r1_1*nb11 + off_base);
            global half4 * y4_2 = (global half4 *)(src1 + r1_2*nb11 + off_base);
            global half4 * y4_3 = (global half4 *)(src1 + r1_3*nb11 + off_base);

            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;

            int n_chunks = ne00 / 4;
            for (int i = lid; i < n_chunks; i += sg_size) {
                half4 xi = x4[i];
                half4 k0 = y4_0[i];
                half4 k1 = y4_1[i];
                half4 k2 = y4_2[i];
                half4 k3 = y4_3[i];

                sum0 += (half)xi.s0 * k0.s0 + (half)xi.s1 * k0.s1 + (half)xi.s2 * k0.s2 + (half)xi.s3 * k0.s3;
                sum1 += (half)xi.s0 * k1.s0 + (half)xi.s1 * k1.s1 + (half)xi.s2 * k1.s2 + (half)xi.s3 * k1.s3;
                sum2 += (half)xi.s0 * k2.s0 + (half)xi.s1 * k2.s1 + (half)xi.s2 * k2.s2 + (half)xi.s3 * k2.s3;
                sum3 += (half)xi.s0 * k3.s0 + (half)xi.s1 * k3.s1 + (half)xi.s2 * k3.s2 + (half)xi.s3 * k3.s3;
            }

            float all0 = sub_group_reduce_add(sum0);
            float all1 = sub_group_reduce_add(sum1);
            float all2 = sub_group_reduce_add(sum2);
            float all3 = sub_group_reduce_add(sum3);

            if (lid == 0) {
                int tail_start = 4 * n_chunks;
                if (tail_start < ne00) {
                    global half * y_0 = (global half *)y4_0;
                    global half * y_1 = (global half *)y4_1;
                    global half * y_2 = (global half *)y4_2;
                    global half * y_3 = (global half *)y4_3;
                    for (int i = tail_start; i < ne00; ++i) {
                        half xi = x[i];
                        all0 += (half)xi * y_0[i];
                        all1 += (half)xi * y_1[i];
                        all2 += (half)xi * y_2[i];
                        all3 += (half)xi * y_3[i];
                    }
                }
                dst[im*ne1*ne0 + r1_0*ne0 + r0] = all0;
                dst[im*ne1*ne0 + r1_1*ne0 + r0] = all1;
                dst[im*ne1*ne0 + r1_2*ne0 + r0] = all2;
                dst[im*ne1*ne0 + r1_3*ne0 + r0] = all3;
            }
        }
    } else {
        // Boundary fallback: for any trailing rows < 4
        if (ne00 < 128) {
            for (int row = 0; row < N_F16_F16; ++row) {
                int r1 = rb + row;
                if (r1 >= ne11) break;
                global half * y = (global half *)(src1 + r1*nb11 + off_base);
                float sumf = 0.0f;
                for (int i = lid; i < ne00; i += sg_size) {
                    sumf += (half) x[i] * (half) y[i];
                }
                float all_sum = sub_group_reduce_add(sumf);
                if (lid == 0) {
                    dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
                }
            }
        } else {
            global half4 * x4 = (global half4 *)x;
            for (int row = 0; row < N_F16_F16; ++row) {
                int r1 = rb + row;
                if (r1 >= ne11) break;
                global half  * y  = (global half  *)(src1 + r1*nb11 + off_base);
                global half4 * y4 = (global half4 *)y;
                float sumf = 0.0f;
                int n_chunks = ne00 / 4;
                for (int i = lid; i < n_chunks; i += sg_size) {
                    sumf += (half) x4[i].s0 * y4[i].s0;
                    sumf += (half) x4[i].s1 * y4[i].s1;
                    sumf += (half) x4[i].s2 * y4[i].s2;
                    sumf += (half) x4[i].s3 * y4[i].s3;
                }
                float all_sum = sub_group_reduce_add(sumf);
                if (lid == 0) {
                    for (int i = 4 * n_chunks; i < ne00; ++i) {
                        all_sum += (half) x[i] * y[i];
                    }
                    dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
                }
            }
        }
    }
}
