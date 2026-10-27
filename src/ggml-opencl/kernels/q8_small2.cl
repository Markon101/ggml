#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void kernel_q8_small2(global const uint *q, global const half *scales,
        read_only image1d_buffer_t x, global float *out, int K, int M, ulong offsetd) {
    const int row=get_global_id(0);
    out=(global float *)((global char *)out+offsetd);
    half2 acc=(half2)(0), B;
    for(int i=0;i<K;i+=4) {
        uint bits=q[(i/4)*M+row];
        half scale=scales[(i/32)*M+row];
        half4 w=convert_half4(as_char4(bits))*scale;
        B=read_imageh(x,(i+0)*2).s01; acc+=B*w.s0;
        B=read_imageh(x,(i+1)*2).s01; acc+=B*w.s1;
        B=read_imageh(x,(i+2)*2).s01; acc+=B*w.s2;
        B=read_imageh(x,(i+3)*2).s01; acc+=B*w.s3;
    }
    out[row]=convert_float(acc.s0);
    out[M+row]=convert_float(acc.s1);
}
