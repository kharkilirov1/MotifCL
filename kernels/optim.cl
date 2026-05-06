__kernel void sgd_update_f32(__global float* param,
                             __global const float* grad,
                             float lr,
                             int n) {
    int gid = get_global_id(0);
    if (gid < n) param[gid] -= lr * grad[gid];
}

__kernel void adam_update_f32(__global float* param,
                              __global const float* grad,
                              __global float* m,
                              __global float* v,
                              float lr,
                              float beta1,
                              float beta2,
                              float eps,
                              int step,
                              int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float g = grad[gid];
    float mt = beta1 * m[gid] + (1.0f - beta1) * g;
    float vt = beta2 * v[gid] + (1.0f - beta2) * g * g;
    m[gid] = mt;
    v[gid] = vt;
    float m_hat = mt / (1.0f - pow(beta1, (float)step));
    float v_hat = vt / (1.0f - pow(beta2, (float)step));
    param[gid] -= lr * m_hat / (sqrt(v_hat) + eps);
}

__kernel void adam_update_f32_fast(__global float* param,
                                   __global const float* grad,
                                   __global float* m,
                                   __global float* v,
                                   float lr,
                                   float beta1,
                                   float beta2,
                                   float eps,
                                   float inv_bias1,
                                   float inv_bias2,
                                   int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float g = grad[gid];
    float mt = beta1 * m[gid] + (1.0f - beta1) * g;
    float vt = beta2 * v[gid] + (1.0f - beta2) * g * g;
    m[gid] = mt;
    v[gid] = vt;
    param[gid] -= lr * (mt * inv_bias1) / (sqrt(vt * inv_bias2) + eps);
}
