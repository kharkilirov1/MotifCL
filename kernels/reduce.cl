inline float reduce_sum_256(float value, __local float* scratch) {
    int lid = get_local_id(0);
    scratch[lid] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return scratch[0];
}

inline float reduce_max_256(float value, __local float* scratch) {
    int lid = get_local_id(0);
    scratch[lid] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return scratch[0];
}

__kernel void sum_rows_f32(__global const float* x,
                           __global float* out,
                           int rows,
                           int cols) {
    int col = get_global_id(0);
    if (col >= cols) return;
    float acc = 0.0f;
    for (int r = 0; r < rows; ++r) acc += x[r * cols + col];
    out[col] = acc;
}

__kernel void rowwise_sum_f32(__global const float* x,
                              __global float* out,
                              int rows,
                              int cols) {
    int row = get_global_id(0);
    if (row >= rows) return;
    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) acc += x[row * cols + c];
    out[row] = acc;
}

__kernel void rowwise_sum_wg_f32(__global const float* x,
                                 __global float* out,
                                 int rows,
                                 int cols) {
    int row = get_group_id(1);
    int lid = get_local_id(0);
    if (row >= rows) return;
    __local float scratch[256];
    float acc = 0.0f;
    for (int c = lid; c < cols; c += 256) acc += x[row * cols + c];
    acc = reduce_sum_256(acc, scratch);
    if (lid == 0) out[row] = acc;
}

__kernel void rowwise_max_f32(__global const float* x,
                              __global float* out,
                              int rows,
                              int cols) {
    int row = get_global_id(0);
    if (row >= rows) return;
    float acc = x[row * cols];
    for (int c = 1; c < cols; ++c) acc = fmax(acc, x[row * cols + c]);
    out[row] = acc;
}

__kernel void rowwise_max_wg_f32(__global const float* x,
                                 __global float* out,
                                 int rows,
                                 int cols) {
    int row = get_group_id(1);
    int lid = get_local_id(0);
    if (row >= rows) return;
    __local float scratch[256];
    float acc = -3.402823466e+38F;
    for (int c = lid; c < cols; c += 256) acc = fmax(acc, x[row * cols + c]);
    acc = reduce_max_256(acc, scratch);
    if (lid == 0) out[row] = acc;
}

__kernel void rowwise_argmax_f32_i32(__global const float* x,
                                     __global int* out,
                                     int rows,
                                     int cols) {
    int row = get_global_id(0);
    if (row >= rows) return;
    int base = row * cols;
    int best_idx = 0;
    float best = x[base];
    for (int c = 1; c < cols; ++c) {
        float value = x[base + c];
        if (value > best) {
            best = value;
            best_idx = c;
        }
    }
    out[row] = best_idx;
}

inline uint mcl_sample_hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline float mcl_sample_uniform01(uint x) {
    return (float)(mcl_sample_hash_u32(x) & 0x00ffffffu) * (1.0f / 16777216.0f);
}

__kernel void rowwise_sample_reduce_f32_i32(__global const float* x,
                                            __global int* out,
                                            int rows,
                                            int cols,
                                            float temperature,
                                            int top_k,
                                            float top_p,
                                            uint seed) {
    int row = get_global_id(0);
    if (row >= rows) return;
    int base = row * cols;
    int best_idx = 0;
    float best = x[base];
    for (int c = 1; c < cols; ++c) {
        float value = x[base + c];
        if (value > best) {
            best = value;
            best_idx = c;
        }
    }
    if (temperature <= 0.0f) {
        out[row] = best_idx;
        return;
    }

    float inv_temp = 1.0f / fmax(temperature, 1.0e-6f);
    float u = mcl_sample_uniform01(seed ^ ((uint)row * 0x9e3779b9u) ^ 0x85ebca6bu);

    if (top_p >= 0.999999f && (top_k <= 0 || top_k >= cols)) {
        float max_logit = best;
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) sum += exp((x[base + c] - max_logit) * inv_temp);
        if (sum <= 0.0f) {
            out[row] = best_idx;
            return;
        }
        float target = u * sum;
        float acc = 0.0f;
        for (int c = 0; c < cols; ++c) {
            acc += exp((x[base + c] - max_logit) * inv_temp);
            if (acc >= target) {
                out[row] = c;
                return;
            }
        }
        out[row] = cols - 1;
        return;
    }

    int k = top_k;
    if (k <= 0 || k > cols) k = cols;
    if (k > 128) k = 128;
    int selected[128];
    float selected_values[128];
    for (int i = 0; i < 128; ++i) {
        selected[i] = -1;
        selected_values[i] = -3.402823466e+38F;
    }
    for (int slot = 0; slot < k; ++slot) {
        int slot_idx = -1;
        float slot_value = -3.402823466e+38F;
        for (int c = 0; c < cols; ++c) {
            int already = 0;
            for (int p = 0; p < slot; ++p) {
                if (selected[p] == c) {
                    already = 1;
                    break;
                }
            }
            if (already) continue;
            float value = x[base + c];
            if (slot_idx < 0 || value > slot_value) {
                slot_idx = c;
                slot_value = value;
            }
        }
        selected[slot] = slot_idx;
        selected_values[slot] = slot_value;
    }

    float max_logit = selected_values[0];
    float total = 0.0f;
    for (int i = 0; i < k; ++i) total += exp((selected_values[i] - max_logit) * inv_temp);
    if (total <= 0.0f) {
        out[row] = selected[0];
        return;
    }

    int nucleus_k = k;
    if (top_p < 0.999999f) {
        float threshold = fmin(fmax(top_p, 1.0e-6f), 1.0f) * total;
        float cumulative = 0.0f;
        nucleus_k = 0;
        for (int i = 0; i < k; ++i) {
            cumulative += exp((selected_values[i] - max_logit) * inv_temp);
            ++nucleus_k;
            if (cumulative >= threshold) break;
        }
        if (nucleus_k <= 0) nucleus_k = 1;
    }

    float sum = 0.0f;
    for (int i = 0; i < nucleus_k; ++i) sum += exp((selected_values[i] - max_logit) * inv_temp);
    if (sum <= 0.0f) {
        out[row] = selected[0];
        return;
    }
    float target = u * sum;
    float acc = 0.0f;
    for (int i = 0; i < nucleus_k; ++i) {
        acc += exp((selected_values[i] - max_logit) * inv_temp);
        if (acc >= target) {
            out[row] = selected[i];
            return;
        }
    }
    out[row] = selected[nucleus_k - 1];
}
