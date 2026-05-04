__kernel void quantize_f32_to_q8_0(__global const float* x,
                                   __global char* out,
                                   float scale,
                                   int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    int q = (int)rint(x[gid] / scale);
    q = min(127, max(-127, q));
    out[gid] = (char)q;
}

__kernel void dequantize_q8_0_to_f32(__global const char* x,
                                     __global float* out,
                                     float scale,
                                     int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    out[gid] = ((float)x[gid]) * scale;
}

__kernel void quantize_f32_to_q4_0(__global const float* x,
                                   __global uchar* out,
                                   float scale,
                                   int n) {
    int byte_id = get_global_id(0);
    int i0 = byte_id * 2;
    if (i0 >= n) return;

    int q0 = (int)rint(x[i0] / scale);
    q0 = min(7, max(-7, q0));
    uchar lo = (uchar)(q0 + 8);

    uchar hi = (uchar)8;
    int i1 = i0 + 1;
    if (i1 < n) {
        int q1 = (int)rint(x[i1] / scale);
        q1 = min(7, max(-7, q1));
        hi = (uchar)(q1 + 8);
    }

    out[byte_id] = (uchar)(lo | (hi << 4));
}

__kernel void dequantize_q4_0_to_f32(__global const uchar* x,
                                     __global float* out,
                                     float scale,
                                     int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;

    uchar packed = x[gid >> 1];
    uchar code = (gid & 1) ? ((packed >> 4) & 15) : (packed & 15);
    int q = ((int)code) - 8;
    out[gid] = ((float)q) * scale;
}

inline float quant_scale_for_index(__global const float* scales,
                                   int idx,
                                   int rows,
                                   int cols,
                                   int mode,
                                   int block_size) {
    if (mode == 1) {
        int row = idx / cols;
        return scales[row];
    }
    if (mode == 2) {
        int col = idx - (idx / cols) * cols;
        return scales[col];
    }
    if (mode == 3) {
        return scales[idx / block_size];
    }
    return 1.0f;
}

__kernel void quantize_f32_to_q8_0_scaled(__global const float* x,
                                          __global char* out,
                                          __global const float* scales,
                                          int n,
                                          int rows,
                                          int cols,
                                          int mode,
                                          int block_size) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float scale = quant_scale_for_index(scales, gid, rows, cols, mode, block_size);
    int q = (int)rint(x[gid] / scale);
    q = min(127, max(-127, q));
    out[gid] = (char)q;
}

__kernel void dequantize_q8_0_to_f32_scaled(__global const char* x,
                                            __global float* out,
                                            __global const float* scales,
                                            int n,
                                            int rows,
                                            int cols,
                                            int mode,
                                            int block_size) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float scale = quant_scale_for_index(scales, gid, rows, cols, mode, block_size);
    out[gid] = ((float)x[gid]) * scale;
}

__kernel void quantize_f32_to_q4_0_scaled(__global const float* x,
                                          __global uchar* out,
                                          __global const float* scales,
                                          int n,
                                          int rows,
                                          int cols,
                                          int mode,
                                          int block_size) {
    int byte_id = get_global_id(0);
    int i0 = byte_id * 2;
    if (i0 >= n) return;

    float scale0 = quant_scale_for_index(scales, i0, rows, cols, mode, block_size);
    int q0 = (int)rint(x[i0] / scale0);
    q0 = min(7, max(-7, q0));
    uchar lo = (uchar)(q0 + 8);

    uchar hi = (uchar)8;
    int i1 = i0 + 1;
    if (i1 < n) {
        float scale1 = quant_scale_for_index(scales, i1, rows, cols, mode, block_size);
        int q1 = (int)rint(x[i1] / scale1);
        q1 = min(7, max(-7, q1));
        hi = (uchar)(q1 + 8);
    }

    out[byte_id] = (uchar)(lo | (hi << 4));
}

__kernel void dequantize_q4_0_to_f32_scaled(__global const uchar* x,
                                            __global float* out,
                                            __global const float* scales,
                                            int n,
                                            int rows,
                                            int cols,
                                            int mode,
                                            int block_size) {
    int gid = get_global_id(0);
    if (gid >= n) return;

    uchar packed = x[gid >> 1];
    uchar code = (gid & 1) ? ((packed >> 4) & 15) : (packed & 15);
    int q = ((int)code) - 8;
    float scale = quant_scale_for_index(scales, gid, rows, cols, mode, block_size);
    out[gid] = ((float)q) * scale;
}
