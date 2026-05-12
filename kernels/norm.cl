inline float wg_reduce_sum_256(float value, __local float* scratch) {
    int lid = get_local_id(0);
    scratch[lid] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return scratch[0];
}

__kernel void rms_per_row_f32(__global const float* x,
                              __global float* out,
                              int rows,
                              int cols,
                              float eps) {
    int row = get_global_id(0);
    if (row >= rows) return;
    float ss = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = x[row * cols + c];
        ss += v * v;
    }
    out[row] = sqrt(ss / (float)cols + eps);
}

__kernel void rms_per_row_wg_f32(__global const float* x,
                                 __global float* out,
                                 int rows,
                                 int cols,
                                 float eps) {
    int row = get_group_id(1);
    int lid = get_local_id(0);
    if (row >= rows) return;
    __local float scratch[256];
    float ss = 0.0f;
    for (int c = lid; c < cols; c += 256) {
        float v = x[row * cols + c];
        ss += v * v;
    }
    ss = wg_reduce_sum_256(ss, scratch);
    if (lid == 0) out[row] = sqrt(ss / (float)cols + eps);
}

__kernel void rmsnorm_rowwise_f32(__global const float* x,
                                  __global const float* weight,
                                  __global float* out,
                                  int rows,
                                  int cols,
                                  float eps) {
    int row = get_global_id(0);
    if (row >= rows) return;
    float ss = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = x[row * cols + c];
        ss += v * v;
    }
    float inv = rsqrt(ss / (float)cols + eps);
    for (int c = 0; c < cols; ++c) {
        out[row * cols + c] = x[row * cols + c] * inv * weight[c];
    }
}

__kernel void rmsnorm_rowwise_wg_f32(__global const float* x,
                                     __global const float* weight,
                                     __global float* out,
                                     int rows,
                                     int cols,
                                     float eps) {
    int row = get_group_id(1);
    int lid = get_local_id(0);
    if (row >= rows) return;
    __local float scratch[256];
    float ss = 0.0f;
    for (int c = lid; c < cols; c += 256) {
        float v = x[row * cols + c];
        ss += v * v;
    }
    ss = wg_reduce_sum_256(ss, scratch);
    float inv = rsqrt(ss / (float)cols + eps);
    for (int c = lid; c < cols; c += 256) {
        out[row * cols + c] = x[row * cols + c] * inv * weight[c];
    }
}

__kernel void rmsnorm_residual_add_rowwise_wg_f32(__global const float* residual,
                                                  __global const float* x,
                                                  __global const float* weight,
                                                  __global float* out,
                                                  int rows,
                                                  int cols,
                                                  float eps) {
    int row = get_group_id(1);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    if (row >= rows) return;
    __local float scratch[256];
    float ss = 0.0f;
    for (int c = lid; c < cols; c += local_size) {
        float v = x[row * cols + c];
        ss += v * v;
    }
    scratch[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(scratch[0] / (float)cols + eps);
    for (int c = lid; c < cols; c += local_size) {
        int idx = row * cols + c;
        out[idx] = residual[idx] + x[idx] * inv * weight[c];
    }
}

__kernel void rmsnorm_residual_add_scale_rowwise_wg_f32(__global const float* residual,
                                                        __global const float* x,
                                                        __global const float* weight,
                                                        __global const float* scale,
                                                        __global float* out,
                                                        int rows,
                                                        int cols,
                                                        int scale_size,
                                                        float eps) {
    int row = get_group_id(1);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    if (row >= rows) return;
    __local float scratch[256];
    float ss = 0.0f;
    for (int c = lid; c < cols; c += local_size) {
        float v = x[row * cols + c];
        ss += v * v;
    }
    scratch[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(scratch[0] / (float)cols + eps);
    for (int c = lid; c < cols; c += local_size) {
        int idx = row * cols + c;
        float s = scale_size == 1 ? scale[0] : scale[c];
        out[idx] = (residual[idx] + x[idx] * inv * weight[c]) * s;
    }
}

__kernel void rmsnorm_row_inv_f32(__global const float* x,
                                  __global float* row_inv,
                                  int rows,
                                  int cols,
                                  float eps) {
    int row = get_global_id(0);
    if (row >= rows) return;
    float ss = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float xv = x[row * cols + c];
        ss += xv * xv;
    }
    row_inv[row] = rsqrt(ss / (float)cols + eps);
}

__kernel void rmsnorm_backward_x_f32(__global const float* x,
                                     __global const float* weight,
                                     __global const float* grad_out,
                                     __global float* grad_x,
                                     int rows,
                                     int cols,
                                     float eps) {
    int gid = get_global_id(0);
    int n = rows * cols;
    if (gid >= n) return;
    int row = gid / cols;
    int col = gid - row * cols;

    float ss = 0.0f;
    float dot = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float xv = x[row * cols + c];
        ss += xv * xv;
        dot += grad_out[row * cols + c] * weight[c] * xv;
    }
    float inv = rsqrt(ss / (float)cols + eps);
    float xval = x[gid];
    grad_x[gid] = grad_out[gid] * weight[col] * inv - xval * inv * inv * inv * dot / (float)cols;
}

__kernel void rmsnorm_backward_x_wg_f32(__global const float* x,
                                        __global const float* weight,
                                        __global const float* grad_out,
                                        __global float* grad_x,
                                        int rows,
                                        int cols,
                                        float eps) {
    int row = get_group_id(1);
    int lid = get_local_id(0);
    if (row >= rows) return;
    __local float scratch_ss[256];
    __local float scratch_dot[256];
    float ss = 0.0f;
    float dot = 0.0f;
    for (int c = lid; c < cols; c += 256) {
        float xv = x[row * cols + c];
        ss += xv * xv;
        dot += grad_out[row * cols + c] * weight[c] * xv;
    }
    ss = wg_reduce_sum_256(ss, scratch_ss);
    dot = wg_reduce_sum_256(dot, scratch_dot);
    float inv = rsqrt(ss / (float)cols + eps);
    for (int c = lid; c < cols; c += 256) {
        int idx = row * cols + c;
        float xval = x[idx];
        grad_x[idx] = grad_out[idx] * weight[c] * inv - xval * inv * inv * inv * dot / (float)cols;
    }
}

__kernel void rmsnorm_backward_x_residual_f32(__global const float* x,
                                              __global const float* weight,
                                              __global const float* grad_out,
                                              __global const float* residual_grad,
                                              __global float* grad_x,
                                              int rows,
                                              int cols,
                                              float eps) {
    int gid = get_global_id(0);
    int n = rows * cols;
    if (gid >= n) return;
    int row = gid / cols;
    int col = gid - row * cols;

    float ss = 0.0f;
    float dot = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float xv = x[row * cols + c];
        ss += xv * xv;
        dot += grad_out[row * cols + c] * weight[c] * xv;
    }
    float inv = rsqrt(ss / (float)cols + eps);
    float xval = x[gid];
    float norm_grad = grad_out[gid] * weight[col] * inv - xval * inv * inv * inv * dot / (float)cols;
    grad_x[gid] = residual_grad[gid] + norm_grad;
}

__kernel void rmsnorm_backward_x_residual_wg_f32(__global const float* x,
                                                 __global const float* weight,
                                                 __global const float* grad_out,
                                                 __global const float* residual_grad,
                                                 __global float* grad_x,
                                                 int rows,
                                                 int cols,
                                                 float eps) {
    int row = get_group_id(1);
    int lid = get_local_id(0);
    if (row >= rows) return;
    __local float scratch_ss[256];
    __local float scratch_dot[256];
    float ss = 0.0f;
    float dot = 0.0f;
    for (int c = lid; c < cols; c += 256) {
        float xv = x[row * cols + c];
        ss += xv * xv;
        dot += grad_out[row * cols + c] * weight[c] * xv;
    }
    ss = wg_reduce_sum_256(ss, scratch_ss);
    dot = wg_reduce_sum_256(dot, scratch_dot);
    float inv = rsqrt(ss / (float)cols + eps);
    for (int c = lid; c < cols; c += 256) {
        int idx = row * cols + c;
        float xval = x[idx];
        float norm_grad = grad_out[idx] * weight[c] * inv - xval * inv * inv * inv * dot / (float)cols;
        grad_x[idx] = residual_grad[idx] + norm_grad;
    }
}

__kernel void rmsnorm_backward_x_residual_cached_f32(__global const float* x,
                                                     __global const float* weight,
                                                     __global const float* row_inv,
                                                     __global const float* grad_out,
                                                     __global const float* residual_grad,
                                                     __global float* grad_x,
                                                     int rows,
                                                     int cols) {
    int gid = get_global_id(0);
    int n = rows * cols;
    if (gid >= n) return;
    int row = gid / cols;
    int col = gid - row * cols;

    float dot = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float xv = x[row * cols + c];
        dot += grad_out[row * cols + c] * weight[c] * xv;
    }
    float inv = row_inv[row];
    float xval = x[gid];
    float norm_grad = grad_out[gid] * weight[col] * inv - xval * inv * inv * inv * dot / (float)cols;
    grad_x[gid] = residual_grad[gid] + norm_grad;
}

__kernel void rmsnorm_backward_x_residual_cached_wg_f32(__global const float* x,
                                                        __global const float* weight,
                                                        __global const float* row_inv,
                                                        __global const float* grad_out,
                                                        __global const float* residual_grad,
                                                        __global float* grad_x,
                                                        int rows,
                                                        int cols) {
    int row = get_group_id(1);
    int lid = get_local_id(0);
    if (row >= rows) return;
    __local float scratch_dot[256];
    float dot = 0.0f;
    for (int c = lid; c < cols; c += 256) {
        float xv = x[row * cols + c];
        dot += grad_out[row * cols + c] * weight[c] * xv;
    }
    dot = wg_reduce_sum_256(dot, scratch_dot);
    float inv = row_inv[row];
    float inv3 = inv * inv * inv;
    for (int c = lid; c < cols; c += 256) {
        int idx = row * cols + c;
        float xval = x[idx];
        float norm_grad = grad_out[idx] * weight[c] * inv - xval * inv3 * dot / (float)cols;
        grad_x[idx] = residual_grad[idx] + norm_grad;
    }
}

__kernel void rmsnorm_backward_weight_f32(__global const float* x,
                                          __global const float* grad_out,
                                          __global float* grad_weight,
                                          int rows,
                                          int cols,
                                          float eps) {
    int col = get_global_id(0);
    if (col >= cols) return;

    float acc = 0.0f;
    for (int row = 0; row < rows; ++row) {
        float ss = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float xv = x[row * cols + c];
            ss += xv * xv;
        }
        float inv = rsqrt(ss / (float)cols + eps);
        acc += grad_out[row * cols + col] * x[row * cols + col] * inv;
    }
    grad_weight[col] = acc;
}

__kernel void rmsnorm_row_inv_wg_f32(__global const float* x,
                                     __global float* row_inv,
                                     int rows,
                                     int cols,
                                     float eps) {
    int row = get_group_id(1);
    int lid = get_local_id(0);
    if (row >= rows) return;
    __local float scratch[256];
    float ss = 0.0f;
    for (int c = lid; c < cols; c += 256) {
        float xv = x[row * cols + c];
        ss += xv * xv;
    }
    ss = wg_reduce_sum_256(ss, scratch);
    if (lid == 0) row_inv[row] = rsqrt(ss / (float)cols + eps);
}

__kernel void rmsnorm_backward_weight_cached_f32(__global const float* x,
                                                 __global const float* grad_out,
                                                 __global const float* row_inv,
                                                 __global float* grad_weight,
                                                 int rows,
                                                 int cols) {
    int col = get_global_id(0);
    if (col >= cols) return;

    float acc = 0.0f;
    for (int row = 0; row < rows; ++row) {
        int idx = row * cols + col;
        acc += grad_out[idx] * x[idx] * row_inv[row];
    }
    grad_weight[col] = acc;
}

__kernel void layernorm_rowwise_f32(__global const float* x,
                                    __global const float* weight,
                                    __global const float* bias,
                                    __global float* out,
                                    int rows,
                                    int cols,
                                    float eps) {
    int row = get_global_id(0);
    if (row >= rows) return;

    float mean = 0.0f;
    for (int c = 0; c < cols; ++c) {
        mean += x[row * cols + c];
    }
    mean /= (float)cols;

    float var = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float centered = x[row * cols + c] - mean;
        var += centered * centered;
    }
    var /= (float)cols;

    float inv_std = rsqrt(var + eps);
    for (int c = 0; c < cols; ++c) {
        float normalized = (x[row * cols + c] - mean) * inv_std;
        out[row * cols + c] = normalized * weight[c] + bias[c];
    }
}

__kernel void layernorm_rowwise_wg_f32(__global const float* x,
                                       __global const float* weight,
                                       __global const float* bias,
                                       __global float* out,
                                       int rows,
                                       int cols,
                                       float eps) {
    int row = get_group_id(1);
    int lid = get_local_id(0);
    if (row >= rows) return;
    __local float scratch_sum[256];
    __local float scratch_var[256];

    float sum = 0.0f;
    for (int c = lid; c < cols; c += 256) sum += x[row * cols + c];
    sum = wg_reduce_sum_256(sum, scratch_sum);
    float mean = sum / (float)cols;

    float var_sum = 0.0f;
    for (int c = lid; c < cols; c += 256) {
        float centered = x[row * cols + c] - mean;
        var_sum += centered * centered;
    }
    var_sum = wg_reduce_sum_256(var_sum, scratch_var);
    float inv_std = rsqrt(var_sum / (float)cols + eps);

    for (int c = lid; c < cols; c += 256) {
        float normalized = (x[row * cols + c] - mean) * inv_std;
        out[row * cols + c] = normalized * weight[c] + bias[c];
    }
}
