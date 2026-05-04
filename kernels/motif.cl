__kernel void sarc_apply_f32(__global const float* x,
                             __global const float* fx,
                             __global const float* gamma,
                             __global const float* rms,
                             __global float* out,
                             int rows,
                             int cols) {
    int gid = get_global_id(0);
    int n = rows * cols;
    if (gid >= n) return;
    int row = gid / cols;
    int col = gid - row * cols;
    out[gid] = x[gid] + gamma[col] * fx[gid] / rms[row];
}

__kernel void motif_linear_forward_f32(__global const float* x,
                                       __global const float* w,
                                       __global float* out,
                                       int n) {
    int gid = get_global_id(0);
    if (gid < n) out[gid] = x[gid] + w[gid];
}

__kernel void router_soft_forward_f32(__global const float* logits,
                                      __global float* probs,
                                      int rows,
                                      int cols) {
    int row = get_global_id(0);
    if (row >= rows) return;
    float maxv = logits[row * cols];
    for (int c = 1; c < cols; ++c) maxv = fmax(maxv, logits[row * cols + c]);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float e = exp(logits[row * cols + c] - maxv);
        probs[row * cols + c] = e;
        sum += e;
    }
    for (int c = 0; c < cols; ++c) probs[row * cols + c] /= sum;
}
