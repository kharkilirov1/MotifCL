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

__kernel void gather_last_token_logits_f32(__global const float* logits,
                                           __global const int* positions,
                                           __global float* out,
                                           int batch,
                                           int seq_len,
                                           int vocab,
                                           int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    int b = gid / vocab;
    int v = gid - b * vocab;
    if (b >= batch) return;
    int t = positions[b];
    if (t < 0) t = 0;
    if (t >= seq_len) t = seq_len - 1;
    out[gid] = logits[(b * seq_len + t) * vocab + v];
}

inline uint mcl_hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline float mcl_uniform01(uint x) {
    return (float)(mcl_hash_u32(x) & 0x00ffffffu) * (1.0f / 16777216.0f);
}

__kernel void dropout_f32(__global const float* x,
                          __global float* out,
                          __global float* mask,
                          int n,
                          float p,
                          float keep_scale,
                          uint seed) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float keep = (mcl_uniform01(seed ^ (uint)gid) >= p) ? keep_scale : 0.0f;
    mask[gid] = keep;
    out[gid] = x[gid] * keep;
}

__kernel void masked_fill_f32_mask_f32(__global const float* x,
                                       __global const float* mask,
                                       __global float* out,
                                       float value,
                                       int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = (mask[gid] != 0.0f) ? value : x[gid];
}

__kernel void masked_fill_f32_mask_i32(__global const float* x,
                                       __global const int* mask,
                                       __global float* out,
                                       float value,
                                       int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = (mask[gid] != 0) ? value : x[gid];
}

__kernel void masked_fill_f32_mask_u8(__global const float* x,
                                      __global const uchar* mask,
                                      __global float* out,
                                      float value,
                                      int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = (mask[gid] != (uchar)0) ? value : x[gid];
}
