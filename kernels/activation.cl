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

__kernel void gelu_mul_f32(__global const float* x,
                           __global const float* y,
                           __global float* out,
                           int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float v = x[gid];
        float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        float g = 0.5f * v * (1.0f + tanh(t));
        out[gid] = g * y[gid];
    }
}

__kernel void gelu_mul_packed_per_layer_f32(__global const float* gate,
                                            __global const float* packed,
                                            __global float* out,
                                            int token_count,
                                            int layer_count,
                                            int ple_dim,
                                            int layer,
                                            int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    int token = gid / ple_dim;
    int d = gid - token * ple_dim;
    float v = gate[gid];
    float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    float g = 0.5f * v * (1.0f + tanh(t));
    float y = packed[(token * layer_count + layer) * ple_dim + d];
    out[gid] = g * y;
}

// Decode-only fused gate projection + GELU + per-layer input multiply.
// This keeps the same output parallelism as matmul_f32_m1_f32 for the
// [1, hidden] x [hidden, ple_dim] gate projection but removes a separate
// gelu_mul_packed_per_layer_f32 launch and temporary gate tensor.
__kernel void gelu_packed_ple_gate_m1_f32(__global const float* h,
                                          __global const float* packed,
                                          __global const float* gate_w,
                                          __global float* out,
                                          int hidden,
                                          int ple_dim,
                                          int layer_count,
                                          int layer) {
    int j = get_global_id(0);
    if (j >= ple_dim) return;
    float acc = 0.0f;
    for (int i = 0; i < hidden; ++i) {
        acc += h[i] * gate_w[i * ple_dim + j];
    }
    float t = 0.7978845608028654f * (acc + 0.044715f * acc * acc * acc);
    float g = 0.5f * acc * (1.0f + tanh(t));
    out[j] = g * packed[layer * ple_dim + j];
}

// RX580/decode variant: one wave64 computes four PLE columns. Compared to
// gelu_packed_ple_gate_m1_f32, this reuses h[i] across four contiguous weights
// and parallel-reduces the hidden dimension.
__kernel void gelu_packed_ple_gate_m1_wg64x4_f32(__global const float* h,
                                                 __global const float* packed,
                                                 __global const float* gate_w,
                                                 __global float* out,
                                                 int hidden,
                                                 int ple_dim,
                                                 int layer_count,
                                                 int layer,
                                                 __local float* scratch) {
    int group = get_group_id(0);
    int lid = get_local_id(0);
    int col0 = group * 4;
    int c0 = col0;
    int c1 = col0 + 1;
    int c2 = col0 + 2;
    int c3 = col0 + 3;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (int i = lid; i < hidden; i += 64) {
        float hv = h[i];
        int base = i * ple_dim + col0;
        if (c0 < ple_dim) acc0 += hv * gate_w[base];
        if (c1 < ple_dim) acc1 += hv * gate_w[base + 1];
        if (c2 < ple_dim) acc2 += hv * gate_w[base + 2];
        if (c3 < ple_dim) acc3 += hv * gate_w[base + 3];
    }
    scratch[lid] = acc0;
    scratch[64 + lid] = acc1;
    scratch[128 + lid] = acc2;
    scratch[192 + lid] = acc3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
            scratch[64 + lid] += scratch[64 + lid + stride];
            scratch[128 + lid] += scratch[128 + lid + stride];
            scratch[192 + lid] += scratch[192 + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        int packed_base = layer * ple_dim;
        if (c0 < ple_dim) {
            float v = scratch[0];
            float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
            out[c0] = (0.5f * v * (1.0f + tanh(t))) * packed[packed_base + c0];
        }
        if (c1 < ple_dim) {
            float v = scratch[64];
            float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
            out[c1] = (0.5f * v * (1.0f + tanh(t))) * packed[packed_base + c1];
        }
        if (c2 < ple_dim) {
            float v = scratch[128];
            float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
            out[c2] = (0.5f * v * (1.0f + tanh(t))) * packed[packed_base + c2];
        }
        if (c3 < ple_dim) {
            float v = scratch[192];
            float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
            out[c3] = (0.5f * v * (1.0f + tanh(t))) * packed[packed_base + c3];
        }
    }
}

// Decode-only Gemma4 per-layer-input fusion for one token:
// out = h + rmsnorm((gelu(h @ inp_gate) * packed[layer]) @ proj, post_norm_weight)
// One workgroup handles one layer. `scratch` is laid out as:
//   mixed[ple_dim] | projected[hidden] | reduce[local_size]
__kernel void fused_packed_ple_m1_f32(__global const float* h,
                                      __global const float* packed,
                                      __global const float* gate_w,
                                      __global const float* proj_w,
                                      __global const float* norm_w,
                                      __global float* out,
                                      int hidden,
                                      int ple_dim,
                                      int layer_count,
                                      int layer,
                                      float eps,
                                      __local float* scratch) {
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    __local float* mixed = scratch;
    __local float* projected = scratch + ple_dim;
    __local float* red = projected + hidden;

    for (int j = lid; j < ple_dim; j += local_size) {
        float acc = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            acc += h[i] * gate_w[i * ple_dim + j];
        }
        float t = 0.7978845608028654f * (acc + 0.044715f * acc * acc * acc);
        float g = 0.5f * acc * (1.0f + tanh(t));
        mixed[j] = g * packed[layer * ple_dim + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float ss = 0.0f;
    for (int d = lid; d < hidden; d += local_size) {
        float acc = 0.0f;
        for (int j = 0; j < ple_dim; ++j) {
            acc += mixed[j] * proj_w[j * hidden + d];
        }
        projected[d] = acc;
        ss += acc * acc;
    }
    red[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) red[lid] += red[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = rsqrt(red[0] / (float)hidden + eps);
    for (int d = lid; d < hidden; d += local_size) {
        out[d] = h[d] + projected[d] * inv * norm_w[d];
    }
}

__kernel void silu_mul_f32(__global const float* x,
                           __global const float* y,
                           __global float* out,
                           int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float v = x[gid];
        out[gid] = (v / (1.0f + exp(-v))) * y[gid];
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

#define SWIGLU_RB_M 32
#define SWIGLU_RB_N 32
#define SWIGLU_RB_K 16
#define SWIGLU_RB_THREAD_M 4
#define SWIGLU_RB_THREAD_N 4
#define SWIGLU_RB_LX 8
#define SWIGLU_RB_LY 8

#define WRITE_SWIGLU_DOWN_GRAD(row_expr, col_expr, d_hidden_expr)                                      \
    do {                                                                                               \
        const int wr_row = (row_expr);                                                                 \
        const int wr_col = (col_expr);                                                                 \
        if (wr_row < rows && wr_col < hidden) {                                                        \
            const int wr_base = wr_row * (hidden * 2);                                                 \
            const float wr_gate = packed[wr_base + wr_col];                                           \
            const float wr_up = packed[wr_base + hidden + wr_col];                                    \
            const float wr_sig = 1.0f / (1.0f + exp(-wr_gate));                                       \
            const float wr_silu = wr_gate * wr_sig;                                                   \
            const float wr_dsilu = wr_sig + wr_gate * wr_sig * (1.0f - wr_sig);                       \
            const float wr_d_hidden = (d_hidden_expr);                                                \
            grad_packed[wr_base + wr_col] = wr_d_hidden * wr_up * wr_dsilu;                           \
            grad_packed[wr_base + hidden + wr_col] = wr_d_hidden * wr_silu;                           \
        }                                                                                              \
    } while (0)

__kernel void swiglu_down_backward_packed_f32(__global const float* packed,
                                             __global const float* down_weight,
                                             __global const float* grad_out,
                                             __global float* grad_packed,
                                             int rows,
                                             int hidden,
                                             int channels) {
    const int lcol = get_local_id(0);
    const int lrow = get_local_id(1);
    const int lid = lrow * SWIGLU_RB_LX + lcol;
    const int block_col = get_group_id(0) * SWIGLU_RB_N;
    const int block_row = get_group_id(1) * SWIGLU_RB_M;

    __local float As[SWIGLU_RB_M][SWIGLU_RB_K];
    __local float Bs[SWIGLU_RB_K][SWIGLU_RB_N];

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    for (int kt = 0; kt < channels; kt += SWIGLU_RB_K) {
        for (int idx = lid; idx < SWIGLU_RB_M * SWIGLU_RB_K; idx += SWIGLU_RB_LX * SWIGLU_RB_LY) {
            const int r = idx / SWIGLU_RB_K;
            const int kk = idx - r * SWIGLU_RB_K;
            const int gr = block_row + r;
            const int gk = kt + kk;
            As[r][kk] = (gr < rows && gk < channels) ? grad_out[gr * channels + gk] : 0.0f;
        }
        for (int idx = lid; idx < SWIGLU_RB_K * SWIGLU_RB_N; idx += SWIGLU_RB_LX * SWIGLU_RB_LY) {
            const int c = idx / SWIGLU_RB_K;
            const int kk = idx - c * SWIGLU_RB_K;
            const int gk = kt + kk;
            const int gc = block_col + c;
            Bs[kk][c] = (gk < channels && gc < hidden) ? down_weight[gc * channels + gk] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int rr = lrow * SWIGLU_RB_THREAD_M;
        const int cc = lcol * SWIGLU_RB_THREAD_N;
        for (int kk = 0; kk < SWIGLU_RB_K; ++kk) {
            const float a0 = As[rr + 0][kk];
            const float a1 = As[rr + 1][kk];
            const float a2 = As[rr + 2][kk];
            const float a3 = As[rr + 3][kk];
            const float b0 = Bs[kk][cc + 0];
            const float b1 = Bs[kk][cc + 1];
            const float b2 = Bs[kk][cc + 2];
            const float b3 = Bs[kk][cc + 3];
            acc00 += a0 * b0; acc01 += a0 * b1; acc02 += a0 * b2; acc03 += a0 * b3;
            acc10 += a1 * b0; acc11 += a1 * b1; acc12 += a1 * b2; acc13 += a1 * b3;
            acc20 += a2 * b0; acc21 += a2 * b1; acc22 += a2 * b2; acc23 += a2 * b3;
            acc30 += a3 * b0; acc31 += a3 * b1; acc32 += a3 * b2; acc33 += a3 * b3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int row0 = block_row + lrow * SWIGLU_RB_THREAD_M;
    const int col0 = block_col + lcol * SWIGLU_RB_THREAD_N;
    WRITE_SWIGLU_DOWN_GRAD(row0 + 0, col0 + 0, acc00);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 0, col0 + 1, acc01);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 0, col0 + 2, acc02);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 0, col0 + 3, acc03);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 1, col0 + 0, acc10);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 1, col0 + 1, acc11);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 1, col0 + 2, acc12);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 1, col0 + 3, acc13);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 2, col0 + 0, acc20);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 2, col0 + 1, acc21);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 2, col0 + 2, acc22);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 2, col0 + 3, acc23);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 3, col0 + 0, acc30);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 3, col0 + 1, acc31);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 3, col0 + 2, acc32);
    WRITE_SWIGLU_DOWN_GRAD(row0 + 3, col0 + 3, acc33);
}

#undef WRITE_SWIGLU_DOWN_GRAD

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
