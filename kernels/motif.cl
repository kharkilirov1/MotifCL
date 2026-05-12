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

__kernel void sigmoid_gate_mul_f32(__global const float* x,
                                   __global const float* gate,
                                   __global float* out,
                                   int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float g = gate[gid];
    out[gid] = x[gid] / (1.0f + exp(-g));
}

__kernel void moe_swiglu_forward_f32(__global const float* x,
                                     __global const float* router,
                                     __global const float* gate_w,
                                     __global const float* up_w,
                                     __global const float* down_w,
                                     __global float* out,
                                     int rows,
                                     int in_features,
                                     int hidden,
                                     int experts,
                                     int top_k) {
    int out_col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= rows || out_col >= in_features) return;

    int selected[8];
    float selected_logits[8];
    for (int i = 0; i < 8; ++i) {
        selected[i] = -1;
        selected_logits[i] = -3.402823466e+38F;
    }

    for (int pick = 0; pick < top_k; ++pick) {
        int best_e = -1;
        float best = -3.402823466e+38F;
        for (int e = 0; e < experts; ++e) {
            int already = 0;
            for (int j = 0; j < pick; ++j) {
                if (selected[j] == e) already = 1;
            }
            if (already) continue;
            float logit = 0.0f;
            for (int i = 0; i < in_features; ++i) {
                logit += x[row * in_features + i] * router[i * experts + e];
            }
            if (best_e < 0 || logit > best) {
                best = logit;
                best_e = e;
            }
        }
        selected[pick] = best_e;
        selected_logits[pick] = best;
    }

    float max_logit = selected_logits[0];
    for (int i = 1; i < top_k; ++i) max_logit = fmax(max_logit, selected_logits[i]);
    float denom = 0.0f;
    for (int i = 0; i < top_k; ++i) denom += exp(selected_logits[i] - max_logit);
    denom = fmax(denom, 1e-20f);

    float acc_out = 0.0f;
    for (int pick = 0; pick < top_k; ++pick) {
        int e = selected[pick];
        float route = exp(selected_logits[pick] - max_logit) / denom;
        float expert_acc = 0.0f;
        for (int h = 0; h < hidden; ++h) {
            float gate = 0.0f;
            float up = 0.0f;
            int base_in_h = (e * in_features * hidden) + h;
            for (int i = 0; i < in_features; ++i) {
                float xv = x[row * in_features + i];
                gate += xv * gate_w[base_in_h + i * hidden];
                up += xv * up_w[base_in_h + i * hidden];
            }
            float sig = 1.0f / (1.0f + exp(-gate));
            float hidden_val = gate * sig * up;
            expert_acc += hidden_val * down_w[e * hidden * in_features + h * in_features + out_col];
        }
        acc_out += route * expert_acc;
    }
    out[row * in_features + out_col] = acc_out;
}

__kernel void gated_delta_recurrent_f32(__global const float* q,
                                        __global const float* k,
                                        __global const float* v,
                                        __global const float* gate,
                                        __global float* state,
                                        __global float* out,
                                        int batch,
                                        int seq_len,
                                        int heads,
                                        int head_dim,
                                        float decay) {
    int channel = get_global_id(0);
    int b = get_global_id(1);
    int channels = heads * head_dim;
    if (b >= batch || channel >= channels) return;

    int h = channel / head_dim;
    int d = channel - h * head_dim;
    int state_base = ((b * heads + h) * head_dim + d) * head_dim;
    for (int t = 0; t < seq_len; ++t) {
        int token_base = (b * seq_len + t) * channels + h * head_dim;
        float acc = 0.0f;
        for (int s = 0; s < head_dim; ++s) {
            acc += state[state_base + s] * q[token_base + s];
        }
        float g = gate[token_base + d];
        out[token_base + d] = acc / (1.0f + exp(-g));
        float vd = v[token_base + d];
        for (int s = 0; s < head_dim; ++s) {
            state[state_base + s] = decay * state[state_base + s] + k[token_base + s] * vd;
        }
    }
}
