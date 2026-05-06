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

__kernel void swiglu_f32(__global const float* packed,
                         __global float* out,
                         int rows,
                         int hidden) {
    int gid = get_global_id(0);
    int n = rows * hidden;
    if (gid >= n) return;
    int row = gid / hidden;
    int col = gid - row * hidden;
    int base = row * (hidden * 2);
    float gate = packed[base + col];
    float up = packed[base + hidden + col];
    float sig = 1.0f / (1.0f + exp(-gate));
    out[gid] = gate * sig * up;
}

__kernel void swiglu_backward_f32(__global const float* packed,
                                  __global const float* grad_out,
                                  __global float* grad_packed,
                                  int rows,
                                  int hidden) {
    int gid = get_global_id(0);
    int total = rows * hidden * 2;
    if (gid >= total) return;
    int row = gid / (hidden * 2);
    int col2 = gid - row * (hidden * 2);
    int col = col2 < hidden ? col2 : col2 - hidden;
    int base = row * (hidden * 2);
    float gate = packed[base + col];
    float up = packed[base + hidden + col];
    float go = grad_out[row * hidden + col];
    float sig = 1.0f / (1.0f + exp(-gate));
    float silu = gate * sig;
    if (col2 < hidden) {
        grad_packed[gid] = go * up * (sig + gate * sig * (1.0f - sig));
    } else {
        grad_packed[gid] = go * silu;
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
