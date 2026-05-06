__kernel void softmax_rows_f32(__global const float* x,
                               __global float* out,
                               int rows,
                               int cols) {
    int row = get_global_id(0);
    if (row >= rows) return;
    float maxv = x[row * cols];
    for (int c = 1; c < cols; ++c) maxv = fmax(maxv, x[row * cols + c]);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float e = exp(x[row * cols + c] - maxv);
        out[row * cols + c] = e;
        sum += e;
    }
    for (int c = 0; c < cols; ++c) out[row * cols + c] /= sum;
}

__kernel void causal_mask_f32(__global const float* scores,
                              __global float* out,
                              int rows,
                              int cols,
                              float mask_value) {
    int gid = get_global_id(0);
    int n = rows * cols;
    if (gid >= n) return;
    int row = gid / cols;
    int col = gid - row * cols;
    out[gid] = (col > row) ? mask_value : scores[gid];
}

__kernel void rope_batch_f32(__global float* x, int rows, int cols) {
    int gid = get_global_id(0);
    int n = rows * cols;
    if (gid < n) x[gid] = x[gid];
}

#define FA_WG 128
#define FA_TILE 16
#define FA_MAX_HEAD_DIM 128
#define FA_STAGED_MAX_TOKENS 128

inline float fa_tile_max(__local float* scores, int valid) {
    float m = -3.402823466e+38F;
    for (int i = 0; i < valid; ++i) m = fmax(m, scores[i]);
    return m;
}

inline float fa_tile_sum_exp(__local float* scores, int valid, float m) {
    float s = 0.0f;
    for (int i = 0; i < valid; ++i) s += exp(scores[i] - m);
    return s;
}

inline void fa_load_kv_tile(__global const float* k,
                            __global const float* v,
                            __local float* k_tile,
                            __local float* v_tile,
                            int b,
                            int kt_start,
                            int valid,
                            int tokens,
                            int channels,
                            int head_offset,
                            int head_dim) {
    int lid = get_local_id(0);
    for (int idx = lid; idx < valid * head_dim; idx += FA_WG) {
        int t = idx / head_dim;
        int d = idx - t * head_dim;
        int src = (b * tokens + kt_start + t) * channels + head_offset + d;
        k_tile[idx] = k[src];
        v_tile[idx] = v[src];
    }
}

inline void fa_compute_scores(__local const float* q_vec,
                              __local const float* k_tile,
                              __local float* scores,
                              int valid,
                              int head_dim,
                              float scale) {
    int lid = get_local_id(0);
    if (lid < valid) {
        float score = 0.0f;
        int base = lid * head_dim;
        for (int d = 0; d < head_dim; ++d) score += q_vec[d] * k_tile[base + d];
        scores[lid] = score * scale;
    }
}

__kernel void multihead_attention_flash_f32(__global const float* q,
                                            __global const float* k,
                                            __global const float* v,
                                            __global float* out,
                                            int batch,
                                            int tokens,
                                            int channels,
                                            int n_head,
                                            int head_dim,
                                            int causal,
                                            float scale) {
    int lid = get_local_id(0);
    int bh = get_group_id(0);
    int token = get_group_id(1);
    int b = bh / n_head;
    int head = bh - b * n_head;
    if (b >= batch || token >= tokens || head >= n_head || head_dim > FA_MAX_HEAD_DIM) return;

    int head_offset = head * head_dim;
    int base_q = (b * tokens + token) * channels + head_offset;
    __local float q_vec[FA_MAX_HEAD_DIM];
    __local float k_tile[FA_TILE * FA_MAX_HEAD_DIM];
    __local float v_tile[FA_TILE * FA_MAX_HEAD_DIM];
    __local float scores[FA_TILE];
    __local float shared[2];

    if (lid < head_dim) q_vec[lid] = q[base_q + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc = 0.0f;
    float m = -3.402823466e+38F;
    float denom = 0.0f;
    int d_out = lid;

    for (int kt = 0; kt < tokens; kt += FA_TILE) {
        int valid = min(FA_TILE, tokens - kt);
        if (causal && kt > token) valid = 0;
        if (causal && kt + valid - 1 > token) valid = max(0, token - kt + 1);
        fa_load_kv_tile(k, v, k_tile, v_tile, b, kt, valid, tokens, channels, head_offset, head_dim);
        barrier(CLK_LOCAL_MEM_FENCE);
        fa_compute_scores(q_vec, k_tile, scores, valid, head_dim, scale);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0) {
            float tile_m = fa_tile_max(scores, valid);
            float new_m = fmax(m, tile_m);
            float alpha = (m == -3.402823466e+38F) ? 0.0f : exp(m - new_m);
            float tile_sum = fa_tile_sum_exp(scores, valid, new_m);
            shared[0] = new_m;
            shared[1] = denom * alpha + tile_sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float new_m = shared[0];
        float new_denom = shared[1];
        float alpha_thread = (m == -3.402823466e+38F) ? 0.0f : exp(m - new_m);
        if (d_out < head_dim) {
            acc *= alpha_thread;
            for (int t = 0; t < valid; ++t) {
                float p = exp(scores[t] - new_m);
                acc += p * v_tile[t * head_dim + d_out];
            }
        }
        m = new_m;
        denom = new_denom;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (d_out < head_dim) {
        out[(b * tokens + token) * channels + head_offset + d_out] = acc / denom;
    }
}

__kernel void multihead_attention_f32(__global const float* q,
                                      __global const float* k,
                                      __global const float* v,
                                      __global float* out,
                                      int batch,
                                      int tokens,
                                      int channels,
                                      int n_head,
                                      int head_dim,
                                      int causal,
                                      float scale) {
    int gid = get_global_id(0);
    int total = batch * tokens * channels;
    if (gid >= total) return;

    int c = gid % channels;
    int token = (gid / channels) % tokens;
    int b = gid / (tokens * channels);
    int head = c / head_dim;
    if (head >= n_head) return;

    int base_q = (b * tokens + token) * channels;
    int head_offset = head * head_dim;

    float max_score = -3.402823466e+38F;
    for (int key_token = 0; key_token < tokens; ++key_token) {
        if (causal && key_token > token) continue;
        int base_k = (b * tokens + key_token) * channels;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
        }
        score *= scale;
        max_score = fmax(max_score, score);
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (int key_token = 0; key_token < tokens; ++key_token) {
        if (causal && key_token > token) continue;
        int base_k = (b * tokens + key_token) * channels;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
        }
        float prob = exp(score * scale - max_score);
        denom += prob;
        acc += prob * v[(b * tokens + key_token) * channels + c];
    }

    out[gid] = acc / denom;
}

__kernel void multihead_attention_backward_v_f32(__global const float* q,
                                                 __global const float* k,
                                                 __global const float* grad_out,
                                                 __global float* grad_v,
                                                 int batch,
                                                 int tokens,
                                                 int channels,
                                                 int n_head,
                                                 int head_dim,
                                                 int causal,
                                                 float scale) {
    int gid = get_global_id(0);
    int total = batch * tokens * channels;
    if (gid >= total) return;

    int c = gid % channels;
    int key_token = (gid / channels) % tokens;
    int b = gid / (tokens * channels);
    int head = c / head_dim;
    if (head >= n_head) return;
    int head_offset = head * head_dim;

    float acc = 0.0f;
    for (int query_token = 0; query_token < tokens; ++query_token) {
        if (causal && key_token > query_token) continue;
        int base_q = (b * tokens + query_token) * channels;

        float max_score = -3.402823466e+38F;
        for (int kt = 0; kt < tokens; ++kt) {
            if (causal && kt > query_token) continue;
            int base_k = (b * tokens + kt) * channels;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
            }
            max_score = fmax(max_score, score * scale);
        }

        float denom = 0.0f;
        float key_score = 0.0f;
        int base_key = (b * tokens + key_token) * channels;
        for (int kt = 0; kt < tokens; ++kt) {
            if (causal && kt > query_token) continue;
            int base_k = (b * tokens + kt) * channels;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
            }
            float scaled = score * scale;
            if (kt == key_token) key_score = scaled;
            denom += exp(scaled - max_score);
        }
        float prob = exp(key_score - max_score) / denom;
        acc += prob * grad_out[(b * tokens + query_token) * channels + c];
    }

    grad_v[gid] = acc;
}

__kernel void multihead_attention_backward_q_f32(__global const float* q,
                                                 __global const float* k,
                                                 __global const float* v,
                                                 __global const float* grad_out,
                                                 __global float* grad_q,
                                                 int batch,
                                                 int tokens,
                                                 int channels,
                                                 int n_head,
                                                 int head_dim,
                                                 int causal,
                                                 float scale) {
    int gid = get_global_id(0);
    int total = batch * tokens * channels;
    if (gid >= total) return;

    int c = gid % channels;
    int query_token = (gid / channels) % tokens;
    int b = gid / (tokens * channels);
    int head = c / head_dim;
    if (head >= n_head) return;
    int head_offset = head * head_dim;
    int base_q = (b * tokens + query_token) * channels;

    float max_score = -3.402823466e+38F;
    for (int kt = 0; kt < tokens; ++kt) {
        if (causal && kt > query_token) continue;
        int base_k = (b * tokens + kt) * channels;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
        }
        max_score = fmax(max_score, score * scale);
    }

    float denom = 0.0f;
    for (int kt = 0; kt < tokens; ++kt) {
        if (causal && kt > query_token) continue;
        int base_k = (b * tokens + kt) * channels;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
        }
        denom += exp(score * scale - max_score);
    }

    float sum_p_grad_p = 0.0f;
    for (int kt = 0; kt < tokens; ++kt) {
        if (causal && kt > query_token) continue;
        int base_k = (b * tokens + kt) * channels;
        float score = 0.0f;
        float grad_p = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
            grad_p += grad_out[base_q + head_offset + d] * v[base_k + head_offset + d];
        }
        float prob = exp(score * scale - max_score) / denom;
        sum_p_grad_p += prob * grad_p;
    }

    float acc = 0.0f;
    for (int kt = 0; kt < tokens; ++kt) {
        if (causal && kt > query_token) continue;
        int base_k = (b * tokens + kt) * channels;
        float score = 0.0f;
        float grad_p = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
            grad_p += grad_out[base_q + head_offset + d] * v[base_k + head_offset + d];
        }
        float prob = exp(score * scale - max_score) / denom;
        float grad_score = prob * (grad_p - sum_p_grad_p) * scale;
        acc += grad_score * k[base_k + c];
    }

    grad_q[gid] = acc;
}

__kernel void multihead_attention_backward_k_f32(__global const float* q,
                                                 __global const float* k,
                                                 __global const float* v,
                                                 __global const float* grad_out,
                                                 __global float* grad_k,
                                                 int batch,
                                                 int tokens,
                                                 int channels,
                                                 int n_head,
                                                 int head_dim,
                                                 int causal,
                                                 float scale) {
    int gid = get_global_id(0);
    int total = batch * tokens * channels;
    if (gid >= total) return;

    int c = gid % channels;
    int key_token = (gid / channels) % tokens;
    int b = gid / (tokens * channels);
    int head = c / head_dim;
    if (head >= n_head) return;
    int head_offset = head * head_dim;

    float acc = 0.0f;
    for (int query_token = 0; query_token < tokens; ++query_token) {
        if (causal && key_token > query_token) continue;
        int base_q = (b * tokens + query_token) * channels;

        float max_score = -3.402823466e+38F;
        for (int kt = 0; kt < tokens; ++kt) {
            if (causal && kt > query_token) continue;
            int base_k = (b * tokens + kt) * channels;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
            }
            max_score = fmax(max_score, score * scale);
        }

        float denom = 0.0f;
        for (int kt = 0; kt < tokens; ++kt) {
            if (causal && kt > query_token) continue;
            int base_k = (b * tokens + kt) * channels;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
            }
            denom += exp(score * scale - max_score);
        }

        float sum_p_grad_p = 0.0f;
        float key_prob = 0.0f;
        float key_grad_p = 0.0f;
        for (int kt = 0; kt < tokens; ++kt) {
            if (causal && kt > query_token) continue;
            int base_k = (b * tokens + kt) * channels;
            float score = 0.0f;
            float grad_p = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
                grad_p += grad_out[base_q + head_offset + d] * v[base_k + head_offset + d];
            }
            float prob = exp(score * scale - max_score) / denom;
            sum_p_grad_p += prob * grad_p;
            if (kt == key_token) {
                key_prob = prob;
                key_grad_p = grad_p;
            }
        }

        float grad_score = key_prob * (key_grad_p - sum_p_grad_p) * scale;
        acc += grad_score * q[base_q + c];
    }

    grad_k[gid] = acc;
}

__kernel void multihead_attention_backward_fused_f32(__global const float* q,
                                                     __global const float* k,
                                                     __global const float* v,
                                                     __global const float* grad_out,
                                                     __global float* grad_q,
                                                     __global float* grad_k,
                                                     __global float* grad_v,
                                                     int batch,
                                                     int tokens,
                                                     int channels,
                                                     int n_head,
                                                     int head_dim,
                                                     int causal,
                                                     float scale) {
    int gid = get_global_id(0);
    int total = batch * tokens * channels;
    if (gid >= total * 3) return;

    int part = gid / total;
    int elem = gid - part * total;
    int c = elem % channels;
    int token = (elem / channels) % tokens;
    int b = elem / (tokens * channels);
    int head = c / head_dim;
    if (head >= n_head) return;
    int head_offset = head * head_dim;

    if (part == 2) {
        int key_token = token;
        float acc = 0.0f;
        for (int query_token = 0; query_token < tokens; ++query_token) {
            if (causal && key_token > query_token) continue;
            int base_q = (b * tokens + query_token) * channels;

            float max_score = -3.402823466e+38F;
            for (int kt = 0; kt < tokens; ++kt) {
                if (causal && kt > query_token) continue;
                int base_k = (b * tokens + kt) * channels;
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
                }
                max_score = fmax(max_score, score * scale);
            }

            float denom = 0.0f;
            float key_score = 0.0f;
            for (int kt = 0; kt < tokens; ++kt) {
                if (causal && kt > query_token) continue;
                int base_k = (b * tokens + kt) * channels;
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
                }
                float scaled = score * scale;
                if (kt == key_token) key_score = scaled;
                denom += exp(scaled - max_score);
            }
            float prob = exp(key_score - max_score) / denom;
            acc += prob * grad_out[(b * tokens + query_token) * channels + c];
        }
        grad_v[elem] = acc;
        return;
    }

    if (part == 0) {
        int query_token = token;
        int base_q = (b * tokens + query_token) * channels;

        float max_score = -3.402823466e+38F;
        for (int kt = 0; kt < tokens; ++kt) {
            if (causal && kt > query_token) continue;
            int base_k = (b * tokens + kt) * channels;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
            }
            max_score = fmax(max_score, score * scale);
        }

        float denom = 0.0f;
        for (int kt = 0; kt < tokens; ++kt) {
            if (causal && kt > query_token) continue;
            int base_k = (b * tokens + kt) * channels;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
            }
            denom += exp(score * scale - max_score);
        }

        float sum_p_grad_p = 0.0f;
        for (int kt = 0; kt < tokens; ++kt) {
            if (causal && kt > query_token) continue;
            int base_k = (b * tokens + kt) * channels;
            float score = 0.0f;
            float grad_p = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
                grad_p += grad_out[base_q + head_offset + d] * v[base_k + head_offset + d];
            }
            float prob = exp(score * scale - max_score) / denom;
            sum_p_grad_p += prob * grad_p;
        }

        float acc = 0.0f;
        for (int kt = 0; kt < tokens; ++kt) {
            if (causal && kt > query_token) continue;
            int base_k = (b * tokens + kt) * channels;
            float score = 0.0f;
            float grad_p = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
                grad_p += grad_out[base_q + head_offset + d] * v[base_k + head_offset + d];
            }
            float prob = exp(score * scale - max_score) / denom;
            float grad_score = prob * (grad_p - sum_p_grad_p) * scale;
            acc += grad_score * k[base_k + c];
        }

        grad_q[elem] = acc;
        return;
    }

    int key_token = token;
    float acc = 0.0f;
    for (int query_token = 0; query_token < tokens; ++query_token) {
        if (causal && key_token > query_token) continue;
        int base_q = (b * tokens + query_token) * channels;

        float max_score = -3.402823466e+38F;
        for (int kt = 0; kt < tokens; ++kt) {
            if (causal && kt > query_token) continue;
            int base_k = (b * tokens + kt) * channels;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
            }
            max_score = fmax(max_score, score * scale);
        }

        float denom = 0.0f;
        for (int kt = 0; kt < tokens; ++kt) {
            if (causal && kt > query_token) continue;
            int base_k = (b * tokens + kt) * channels;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
            }
            denom += exp(score * scale - max_score);
        }

        float sum_p_grad_p = 0.0f;
        float key_prob = 0.0f;
        float key_grad_p = 0.0f;
        for (int kt = 0; kt < tokens; ++kt) {
            if (causal && kt > query_token) continue;
            int base_k = (b * tokens + kt) * channels;
            float score = 0.0f;
            float grad_p = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + head_offset + d] * k[base_k + head_offset + d];
                grad_p += grad_out[base_q + head_offset + d] * v[base_k + head_offset + d];
            }
            float prob = exp(score * scale - max_score) / denom;
            sum_p_grad_p += prob * grad_p;
            if (kt == key_token) {
                key_prob = prob;
                key_grad_p = grad_p;
            }
        }

        float grad_score = key_prob * (key_grad_p - sum_p_grad_p) * scale;
        acc += grad_score * q[base_q + c];
    }

    grad_k[elem] = acc;
}

inline void fa_load_head_pair(__global const float* a,
                              __global const float* b,
                              __local float* a_vec,
                              __local float* b_vec,
                              int base,
                              int head_dim) {
    int lid = get_local_id(0);
    if (lid < head_dim) {
        a_vec[lid] = a[base + lid];
        b_vec[lid] = b[base + lid];
    }
}

inline void fa_compute_scores_and_gradp(__local const float* q_vec,
                                        __local const float* go_vec,
                                        __local const float* k_tile,
                                        __local const float* v_tile,
                                        __local float* scores,
                                        __local float* gradp,
                                        int valid,
                                        int head_dim,
                                        float scale) {
    int lid = get_local_id(0);
    if (lid < valid) {
        int base = lid * head_dim;
        float score = 0.0f;
        float gp = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_vec[d] * k_tile[base + d];
            gp += go_vec[d] * v_tile[base + d];
        }
        scores[lid] = score * scale;
        gradp[lid] = gp;
    }
}

inline void fa_softmax_stats_for_query(__global const float* k,
                                       __global const float* v,
                                       __local const float* q_vec,
                                       __local const float* go_vec,
                                       __local float* k_tile,
                                       __local float* v_tile,
                                       __local float* scores,
                                       __local float* gradp,
                                       __local float* shared,
                                       int b,
                                       int query_token,
                                       int tokens,
                                       int channels,
                                       int head_offset,
                                       int head_dim,
                                       int causal,
                                       float scale) {
    int lid = get_local_id(0);
    float m = -3.402823466e+38F;
    float denom = 0.0f;
    float pg_num = 0.0f;
    for (int kt = 0; kt < tokens; kt += FA_TILE) {
        int valid = min(FA_TILE, tokens - kt);
        if (causal && kt > query_token) valid = 0;
        if (causal && kt + valid - 1 > query_token) valid = max(0, query_token - kt + 1);
        fa_load_kv_tile(k, v, k_tile, v_tile, b, kt, valid, tokens, channels, head_offset, head_dim);
        barrier(CLK_LOCAL_MEM_FENCE);
        fa_compute_scores_and_gradp(q_vec, go_vec, k_tile, v_tile, scores, gradp, valid, head_dim, scale);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 0) {
            float tile_m = fa_tile_max(scores, valid);
            float new_m = fmax(m, tile_m);
            float alpha = (m == -3.402823466e+38F) ? 0.0f : exp(m - new_m);
            float tile_sum = 0.0f;
            float tile_pg = 0.0f;
            for (int i = 0; i < valid; ++i) {
                float e = exp(scores[i] - new_m);
                tile_sum += e;
                tile_pg += e * gradp[i];
            }
            m = new_m;
            denom = denom * alpha + tile_sum;
            pg_num = pg_num * alpha + tile_pg;
            shared[0] = m;
            shared[1] = denom;
            shared[2] = pg_num;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        m = shared[0];
        denom = shared[1];
        pg_num = shared[2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        shared[0] = m;
        shared[1] = denom;
        shared[2] = pg_num / denom;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void multihead_attention_backward_flash_f32(__global const float* q,
                                                     __global const float* k,
                                                     __global const float* v,
                                                     __global const float* grad_out,
                                                     __global float* grad_q,
                                                     __global float* grad_k,
                                                     __global float* grad_v,
                                                     int batch,
                                                     int tokens,
                                                     int channels,
                                                     int n_head,
                                                     int head_dim,
                                                     int causal,
                                                     float scale) {
    int lid = get_local_id(0);
    int bh = get_group_id(0);
    int packed_y = get_group_id(1);
    int part = packed_y / tokens;
    int token = packed_y - part * tokens;
    int b = bh / n_head;
    int head = bh - b * n_head;
    if (b >= batch || token >= tokens || head >= n_head || head_dim > FA_MAX_HEAD_DIM) return;

    int head_offset = head * head_dim;
    __local float q_vec[FA_MAX_HEAD_DIM];
    __local float go_vec[FA_MAX_HEAD_DIM];
    __local float k_tile[FA_TILE * FA_MAX_HEAD_DIM];
    __local float v_tile[FA_TILE * FA_MAX_HEAD_DIM];
    __local float scores[FA_TILE];
    __local float gradp[FA_TILE];
    __local float shared[4];

    if (part == 0) {
        int query_token = token;
        int base_q = (b * tokens + query_token) * channels + head_offset;
        fa_load_head_pair(q, grad_out, q_vec, go_vec, base_q, head_dim);
        barrier(CLK_LOCAL_MEM_FENCE);

        fa_softmax_stats_for_query(k, v, q_vec, go_vec, k_tile, v_tile, scores, gradp, shared,
                                   b, query_token, tokens, channels, head_offset, head_dim, causal, scale);
        float m = shared[0];
        float denom = shared[1];
        float sum_pg = shared[2];
        float acc = 0.0f;
        int d_out = lid;
        for (int kt = 0; kt < tokens; kt += FA_TILE) {
            int valid = min(FA_TILE, tokens - kt);
            if (causal && kt > query_token) valid = 0;
            if (causal && kt + valid - 1 > query_token) valid = max(0, query_token - kt + 1);
            fa_load_kv_tile(k, v, k_tile, v_tile, b, kt, valid, tokens, channels, head_offset, head_dim);
            barrier(CLK_LOCAL_MEM_FENCE);
            fa_compute_scores_and_gradp(q_vec, go_vec, k_tile, v_tile, scores, gradp, valid, head_dim, scale);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (d_out < head_dim) {
                for (int t = 0; t < valid; ++t) {
                    float prob = exp(scores[t] - m) / denom;
                    float grad_score = prob * (gradp[t] - sum_pg) * scale;
                    acc += grad_score * k_tile[t * head_dim + d_out];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (d_out < head_dim) grad_q[(b * tokens + query_token) * channels + head_offset + d_out] = acc;
        return;
    }

    if (part == 1) {
        int key_token = token;
        float acc = 0.0f;
        int d_out = lid;
        for (int query_token = 0; query_token < tokens; ++query_token) {
            if (causal && key_token > query_token) continue;
            int base_q = (b * tokens + query_token) * channels + head_offset;
            fa_load_head_pair(q, grad_out, q_vec, go_vec, base_q, head_dim);
            barrier(CLK_LOCAL_MEM_FENCE);
            fa_softmax_stats_for_query(k, v, q_vec, go_vec, k_tile, v_tile, scores, gradp, shared,
                                       b, query_token, tokens, channels, head_offset, head_dim, causal, scale);
            float m = shared[0];
            float denom = shared[1];
            float sum_pg = shared[2];
            if (lid == 0) {
                float key_score = 0.0f;
                float key_gradp = 0.0f;
                int base_k = (b * tokens + key_token) * channels + head_offset;
                for (int d = 0; d < head_dim; ++d) {
                    key_score += q_vec[d] * k[base_k + d];
                    key_gradp += go_vec[d] * v[base_k + d];
                }
                float prob = exp(key_score * scale - m) / denom;
                shared[3] = prob * (key_gradp - sum_pg) * scale;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (d_out < head_dim) acc += shared[3] * q_vec[d_out];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (d_out < head_dim) grad_k[(b * tokens + key_token) * channels + head_offset + d_out] = acc;
        return;
    }

    int key_token = token;
    float acc = 0.0f;
    int d_out = lid;
    for (int query_token = 0; query_token < tokens; ++query_token) {
        if (causal && key_token > query_token) continue;
        int base_q = (b * tokens + query_token) * channels + head_offset;
        fa_load_head_pair(q, grad_out, q_vec, go_vec, base_q, head_dim);
        barrier(CLK_LOCAL_MEM_FENCE);
        fa_softmax_stats_for_query(k, v, q_vec, go_vec, k_tile, v_tile, scores, gradp, shared,
                                   b, query_token, tokens, channels, head_offset, head_dim, causal, scale);
        float m = shared[0];
        float denom = shared[1];
        if (lid == 0) {
            float key_score = 0.0f;
            int base_k = (b * tokens + key_token) * channels + head_offset;
            for (int d = 0; d < head_dim; ++d) key_score += q_vec[d] * k[base_k + d];
            shared[3] = exp(key_score * scale - m) / denom;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (d_out < head_dim) acc += shared[3] * go_vec[d_out];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (d_out < head_dim) grad_v[(b * tokens + key_token) * channels + head_offset + d_out] = acc;
}

__kernel void attention_backward_probs_f32(__global const float* q,
                                           __global const float* k,
                                           __global float* probs,
                                           int batch,
                                           int tokens,
                                           int channels,
                                           int n_head,
                                           int head_dim,
                                           int causal,
                                           float scale) {
    int lid = get_local_id(0);
    int row = get_group_id(1);
    int rows = batch * n_head * tokens;
    if (row >= rows || tokens > FA_STAGED_MAX_TOKENS || head_dim > FA_MAX_HEAD_DIM) return;

    int query_token = row % tokens;
    int bh = row / tokens;
    int head = bh % n_head;
    int b = bh / n_head;
    int head_offset = head * head_dim;
    int base_q = (b * tokens + query_token) * channels + head_offset;
    int row_base = row * tokens;

    __local float scores[FA_STAGED_MAX_TOKENS];

    if (lid < tokens) {
        int key_token = lid;
        if (causal && key_token > query_token) {
            scores[lid] = -3.402823466e+38F;
        } else {
            int base_k = (b * tokens + key_token) * channels + head_offset;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q[base_q + d] * k[base_k + d];
            }
            scores[lid] = score * scale;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float max_score = -3.402823466e+38F;
        for (int kt = 0; kt < tokens; ++kt) max_score = fmax(max_score, scores[kt]);
        float denom = 0.0f;
        for (int kt = 0; kt < tokens; ++kt) {
            float value = (scores[kt] == -3.402823466e+38F) ? 0.0f : exp(scores[kt] - max_score);
            probs[row_base + kt] = value;
            denom += value;
        }
        float inv = 1.0f / denom;
        for (int kt = 0; kt < tokens; ++kt) probs[row_base + kt] *= inv;
    }
}

__kernel void attention_backward_dp_f32(__global const float* grad_out,
                                        __global const float* v,
                                        __global float* dp,
                                        int batch,
                                        int tokens,
                                        int channels,
                                        int n_head,
                                        int head_dim,
                                        int causal) {
    int gid = get_global_id(0);
    int pairs = batch * n_head * tokens * tokens;
    if (gid >= pairs) return;

    int key_token = gid % tokens;
    int row = gid / tokens;
    int query_token = row % tokens;
    int bh = row / tokens;
    int head = bh % n_head;
    int b = bh / n_head;
    if (causal && key_token > query_token) {
        dp[gid] = 0.0f;
        return;
    }

    int head_offset = head * head_dim;
    int base_go = (b * tokens + query_token) * channels + head_offset;
    int base_v = (b * tokens + key_token) * channels + head_offset;
    float acc = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        acc += grad_out[base_go + d] * v[base_v + d];
    }
    dp[gid] = acc;
}

__kernel void attention_backward_ds_f32(__global const float* probs,
                                        __global const float* dp,
                                        __global float* ds,
                                        int batch,
                                        int tokens,
                                        int n_head,
                                        int causal,
                                        float scale) {
    int lid = get_local_id(0);
    int row = get_group_id(1);
    int rows = batch * n_head * tokens;
    if (row >= rows || tokens > FA_STAGED_MAX_TOKENS) return;

    int query_token = row % tokens;
    int row_base = row * tokens;
    __local float shared[1];

    if (lid == 0) {
        float sum_p_dp = 0.0f;
        for (int kt = 0; kt < tokens; ++kt) {
            sum_p_dp += probs[row_base + kt] * dp[row_base + kt];
        }
        shared[0] = sum_p_dp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < tokens) {
        int key_token = lid;
        if (causal && key_token > query_token) {
            ds[row_base + key_token] = 0.0f;
        } else {
            float p = probs[row_base + key_token];
            ds[row_base + key_token] = p * (dp[row_base + key_token] - shared[0]) * scale;
        }
    }
}

__kernel void attention_backward_apply_q_f32(__global const float* ds,
                                             __global const float* k,
                                             __global float* grad_q,
                                             int batch,
                                             int tokens,
                                             int channels,
                                             int n_head,
                                             int head_dim,
                                             int causal) {
    int gid = get_global_id(0);
    int total = batch * tokens * channels;
    if (gid >= total) return;

    int d = gid % head_dim;
    int c = gid % channels;
    int query_token = (gid / channels) % tokens;
    int b = gid / (tokens * channels);
    int head = c / head_dim;
    if (head >= n_head) return;

    int row = ((b * n_head + head) * tokens + query_token);
    int row_base = row * tokens;
    int head_offset = head * head_dim;
    float acc = 0.0f;
    for (int kt = 0; kt < tokens; ++kt) {
        if (causal && kt > query_token) break;
        int base_k = (b * tokens + kt) * channels + head_offset;
        acc += ds[row_base + kt] * k[base_k + d];
    }
    grad_q[gid] = acc;
}

__kernel void attention_backward_apply_k_f32(__global const float* ds,
                                             __global const float* q,
                                             __global float* grad_k,
                                             int batch,
                                             int tokens,
                                             int channels,
                                             int n_head,
                                             int head_dim,
                                             int causal) {
    int gid = get_global_id(0);
    int total = batch * tokens * channels;
    if (gid >= total) return;

    int d = gid % head_dim;
    int c = gid % channels;
    int key_token = (gid / channels) % tokens;
    int b = gid / (tokens * channels);
    int head = c / head_dim;
    if (head >= n_head) return;

    int head_offset = head * head_dim;
    float acc = 0.0f;
    for (int query_token = 0; query_token < tokens; ++query_token) {
        if (causal && key_token > query_token) continue;
        int row = ((b * n_head + head) * tokens + query_token);
        int base_q = (b * tokens + query_token) * channels + head_offset;
        acc += ds[row * tokens + key_token] * q[base_q + d];
    }
    grad_k[gid] = acc;
}

__kernel void attention_backward_apply_v_f32(__global const float* probs,
                                             __global const float* grad_out,
                                             __global float* grad_v,
                                             int batch,
                                             int tokens,
                                             int channels,
                                             int n_head,
                                             int head_dim,
                                             int causal) {
    int gid = get_global_id(0);
    int total = batch * tokens * channels;
    if (gid >= total) return;

    int d = gid % head_dim;
    int c = gid % channels;
    int key_token = (gid / channels) % tokens;
    int b = gid / (tokens * channels);
    int head = c / head_dim;
    if (head >= n_head) return;

    int head_offset = head * head_dim;
    float acc = 0.0f;
    for (int query_token = 0; query_token < tokens; ++query_token) {
        if (causal && key_token > query_token) continue;
        int row = ((b * n_head + head) * tokens + query_token);
        int base_go = (b * tokens + query_token) * channels + head_offset;
        acc += probs[row * tokens + key_token] * grad_out[base_go + d];
    }
    grad_v[gid] = acc;
}
