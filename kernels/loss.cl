__kernel void mse_loss_partial_f32(__global const float* pred,
                                   __global const float* target,
                                   __global float* partial,
                                   int n) {
    int gid = get_global_id(0);
    int start = gid * 256;
    if (start >= n) return;
    float acc = 0.0f;
    for (int i = 0; i < 256; ++i) {
        int idx = start + i;
        if (idx < n) {
            float d = pred[idx] - target[idx];
            acc += d * d;
        }
    }
    partial[gid] = acc;
}

__kernel void mse_backward_f32(__global const float* pred,
                               __global const float* target,
                               __global const float* grad_out,
                               __global float* grad_pred,
                               int n) {
    int gid = get_global_id(0);
    if (gid < n) grad_pred[gid] = grad_out[0] * 2.0f * (pred[gid] - target[gid]) / (float)n;
}

__kernel void mean_reduce_f32(__global const float* partial,
                              __global float* out,
                              int n) {
    int lid = get_local_id(0);
    __local float scratch[256];
    float acc = 0.0f;
    for (int i = lid; i < n; i += 256) acc += partial[i];
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) out[0] = n > 0 ? scratch[0] / (float)n : 0.0f;
}

__kernel void softmax_cross_entropy_partial_f32_i32(__global const float* logits,
                                                     __global const int* targets,
                                                     __global float* partial,
                                                     int rows,
                                                     int cols) {
    int row = get_global_id(0);
    if (row >= rows) return;
    int base = row * cols;
    int target = targets[row];
    float maxv = logits[base];
    for (int c = 1; c < cols; ++c) maxv = fmax(maxv, logits[base + c]);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) sum += exp(logits[base + c] - maxv);
    float logsum = log(sum) + maxv;
    if (target < 0 || target >= cols) partial[row] = 0.0f;
    else partial[row] = logsum - logits[base + target];
}

__kernel void softmax_cross_entropy_backward_f32_i32(__global const float* logits,
                                                      __global const int* targets,
                                                      __global const float* grad_out,
                                                      __global float* grad_logits,
                                                      int rows,
                                                      int cols) {
    int row = get_global_id(0);
    if (row >= rows) return;
    int base = row * cols;
    int target = targets[row];
    float maxv = logits[base];
    for (int c = 1; c < cols; ++c) maxv = fmax(maxv, logits[base + c]);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) sum += exp(logits[base + c] - maxv);
    float scale = grad_out[0] / (float)rows;
    for (int c = 0; c < cols; ++c) {
        float p = exp(logits[base + c] - maxv) / sum;
        float y = (c == target) ? 1.0f : 0.0f;
        grad_logits[base + c] = scale * (p - y);
    }
}
