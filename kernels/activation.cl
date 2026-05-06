__kernel void relu_f32(__global const float* x, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = fmax(x[gid], 0.0f);
}

__kernel void relu_backward_f32(__global const float* x,
                                __global const float* grad_out,
                                __global float* grad_x,
                                int n) {
    int gid = get_global_id(0);
    if (gid < n) grad_x[gid] = x[gid] > 0.0f ? grad_out[gid] : 0.0f;
}

__kernel void gelu_f32(__global const float* x, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float v = x[gid];
        float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        out[gid] = 0.5f * v * (1.0f + tanh(t));
    }
}

__kernel void add_bias_gelu_rows_f32(__global const float* x,
                                     __global const float* bias,
                                     __global float* out,
                                     int rows,
                                     int cols) {
    int gid = get_global_id(0);
    int n = rows * cols;
    if (gid < n) {
        int col = gid % cols;
        float v = x[gid] + bias[col];
        float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        out[gid] = 0.5f * v * (1.0f + tanh(t));
    }
}

__kernel void gelu_backward_f32(__global const float* x,
                                __global const float* grad_out,
                                __global float* grad_x,
                                int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float v = x[gid];
        float c = 0.7978845608028654f;
        float inner = c * (v + 0.044715f * v * v * v);
        float th = tanh(inner);
        float sech2 = 1.0f - th * th;
        float inner_grad = c * (1.0f + 3.0f * 0.044715f * v * v);
        float grad = 0.5f * (1.0f + th) + 0.5f * v * sech2 * inner_grad;
        grad_x[gid] = grad_out[gid] * grad;
    }
}

__kernel void silu_f32(__global const float* x, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float v = x[gid];
        out[gid] = v / (1.0f + exp(-v));
    }
}

__kernel void exp_f32(__global const float* x, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = exp(x[gid]);
}

__kernel void sqrt_f32(__global const float* x, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = sqrt(x[gid]);
}

__kernel void rsqrt_f32(__global const float* x, __global float* out, int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = rsqrt(x[gid]);
}
