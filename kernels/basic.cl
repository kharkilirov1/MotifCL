__kernel void fill_f32(__global float* x, float value, int n) {
    int gid = get_global_id(0);
    if (gid < n) x[gid] = value;
}

__kernel void copy_f32(__global const float* x, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = x[gid];
}

__kernel void add_f32(__global const float* a, __global const float* b, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = a[gid] + b[gid];
}

__kernel void sub_f32(__global const float* a, __global const float* b, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = a[gid] - b[gid];
}

__kernel void mul_f32(__global const float* a, __global const float* b, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = a[gid] * b[gid];
}

__kernel void div_f32(__global const float* a, __global const float* b, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = a[gid] / b[gid];
}

__kernel void add_scalar_f32(__global const float* x, float value, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = x[gid] + value;
}

__kernel void mul_scalar_f32(__global const float* x, float alpha, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = x[gid] * alpha;
}

__kernel void scale_f32(__global float* x, float alpha, int n) {
    int gid = get_global_id(0);
    if (gid < n) x[gid] *= alpha;
}

__kernel void axpy_f32(__global const float* x, __global float* y, float alpha, int n) {
    int gid = get_global_id(0);
    if (gid < n) y[gid] += alpha * x[gid];
}

__kernel void add_inplace_f32(__global float* dst, __global const float* src, int n) {
    int gid = get_global_id(0);
    if (gid < n) dst[gid] += src[gid];
}

__kernel void add_bias_rows_f32(__global const float* x,
                                __global const float* bias,
                                __global float* out,
                                int rows,
                                int cols) {
    int gid = get_global_id(0);
    int n = rows * cols;
    if (gid < n) {
        int col = gid % cols;
        out[gid] = x[gid] + bias[col];
    }
}
