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
                                   float weight_decay,
                                   int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float g = grad[gid];
    float mt = beta1 * m[gid] + (1.0f - beta1) * g;
    float vt = beta2 * v[gid] + (1.0f - beta2) * g * g;
    m[gid] = mt;
    v[gid] = vt;
    float p = param[gid];
    float update = (mt * inv_bias1) / (sqrt(vt * inv_bias2) + eps);
    param[gid] = p - lr * update - lr * weight_decay * p;
}

__kernel void adam_update_f32_fast4(__global float* p0,
                                    __global const float* g0,
                                    __global float* m0,
                                    __global float* v0,
                                    int n0,
                                    __global float* p1,
                                    __global const float* g1,
                                    __global float* m1,
                                    __global float* v1,
                                    int n1,
                                    __global float* p2,
                                    __global const float* g2,
                                    __global float* m2,
                                    __global float* v2,
                                    int n2,
                                    __global float* p3,
                                    __global const float* g3,
                                    __global float* m3,
                                    __global float* v3,
                                    int n3,
                                    float lr,
                                    float beta1,
                                    float beta2,
                                    float eps,
                                    float inv_bias1,
                                    float inv_bias2,
                                    float weight_decay) {
    int gid = get_global_id(0);
#define ADAM_FAST_ONE(P, G, M, V, N)                                      \
    if (gid < (N)) {                                                       \
        float grad = (G)[gid];                                             \
        float mt = beta1 * (M)[gid] + (1.0f - beta1) * grad;               \
        float vt = beta2 * (V)[gid] + (1.0f - beta2) * grad * grad;        \
        (M)[gid] = mt;                                                     \
        (V)[gid] = vt;                                                     \
        float p = (P)[gid];                                                \
        float update = (mt * inv_bias1) / (sqrt(vt * inv_bias2) + eps);    \
        (P)[gid] = p - lr * update - lr * weight_decay * p;                \
    }
    ADAM_FAST_ONE(p0, g0, m0, v0, n0)
    ADAM_FAST_ONE(p1, g1, m1, v1, n1)
    ADAM_FAST_ONE(p2, g2, m2, v2, n2)
    ADAM_FAST_ONE(p3, g3, m3, v3, n3)
#undef ADAM_FAST_ONE
}

__kernel void adam_update_state_f32(__global float* state,
                                    float beta1,
                                    float beta2) {
    if (get_global_id(0) != 0) return;
    float step = state[0] + 1.0f;
    state[0] = step;
    state[1] = 1.0f / (1.0f - pow(beta1, step));
    state[2] = 1.0f / (1.0f - pow(beta2, step));
}

__kernel void adam_update_f32_fast8_state(__global float* p0,
                                          __global const float* g0,
                                          __global float* m0,
                                          __global float* v0,
                                          int n0,
                                          __global float* p1,
                                          __global const float* g1,
                                          __global float* m1,
                                          __global float* v1,
                                          int n1,
                                          __global float* p2,
                                          __global const float* g2,
                                          __global float* m2,
                                          __global float* v2,
                                          int n2,
                                          __global float* p3,
                                          __global const float* g3,
                                          __global float* m3,
                                          __global float* v3,
                                          int n3,
                                          __global float* p4,
                                          __global const float* g4,
                                          __global float* m4,
                                          __global float* v4,
                                          int n4,
                                          __global float* p5,
                                          __global const float* g5,
                                          __global float* m5,
                                          __global float* v5,
                                          int n5,
                                          __global float* p6,
                                          __global const float* g6,
                                          __global float* m6,
                                          __global float* v6,
                                          int n6,
                                          __global float* p7,
                                          __global const float* g7,
                                          __global float* m7,
                                          __global float* v7,
                                          int n7,
                                          __global const float* state,
                                          float lr,
                                          float beta1,
                                          float beta2,
                                          float eps,
                                          float weight_decay) {
    int gid = get_global_id(0);
    float inv_bias1 = state[1];
    float inv_bias2 = state[2];
#define ADAM_FAST_ONE_STATE(P, G, M, V, N)                                \
    if (gid < (N)) {                                                       \
        float grad = (G)[gid];                                             \
        float mt = beta1 * (M)[gid] + (1.0f - beta1) * grad;               \
        float vt = beta2 * (V)[gid] + (1.0f - beta2) * grad * grad;        \
        (M)[gid] = mt;                                                     \
        (V)[gid] = vt;                                                     \
        float p = (P)[gid];                                                \
        float update = (mt * inv_bias1) / (sqrt(vt * inv_bias2) + eps);    \
        (P)[gid] = p - lr * update - lr * weight_decay * p;                \
    }
    ADAM_FAST_ONE_STATE(p0, g0, m0, v0, n0)
    ADAM_FAST_ONE_STATE(p1, g1, m1, v1, n1)
    ADAM_FAST_ONE_STATE(p2, g2, m2, v2, n2)
    ADAM_FAST_ONE_STATE(p3, g3, m3, v3, n3)
    ADAM_FAST_ONE_STATE(p4, g4, m4, v4, n4)
    ADAM_FAST_ONE_STATE(p5, g5, m5, v5, n5)
    ADAM_FAST_ONE_STATE(p6, g6, m6, v6, n6)
    ADAM_FAST_ONE_STATE(p7, g7, m7, v7, n7)
#undef ADAM_FAST_ONE_STATE
}
