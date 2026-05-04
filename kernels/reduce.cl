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
