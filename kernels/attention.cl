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

#ifndef FA_WG
#define FA_WG 128
#endif
#ifndef FA_TILE
#define FA_TILE 16
#endif
#ifndef FA_MAX_HEAD_DIM
#define FA_MAX_HEAD_DIM 128
#endif
#ifndef FA_STAGED_MAX_TOKENS
#define FA_STAGED_MAX_TOKENS 128
#endif

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

__kernel void gqa_backward_probs_f32(__global const float* q,
                                     __global const float* k,
                                     __global float* probs,
                                     int batch,
                                     int query_tokens,
                                     int key_tokens,
                                     int n_head,
                                     int n_kv_head,
                                     int head_dim,
                                     int causal,
                                     int query_offset,
                                     float scale) {
    int lid = get_local_id(0);
    int row = get_group_id(1);
    int rows = batch * n_head * query_tokens;
    if (row >= rows || key_tokens > FA_STAGED_MAX_TOKENS || head_dim > FA_MAX_HEAD_DIM) return;

    int query_token = row % query_tokens;
    int bh = row / query_tokens;
    int head = bh % n_head;
    int b = bh / n_head;
    int group = n_head / n_kv_head;
    int kv_head = head / group;
    int q_channels = n_head * head_dim;
    int kv_channels = n_kv_head * head_dim;
    int base_q = (b * query_tokens + query_token) * q_channels + head * head_dim;
    int row_base = row * key_tokens;
    __local float scores[FA_STAGED_MAX_TOKENS];

    if (lid < key_tokens) {
        int key_token = lid;
        if (causal && key_token > query_offset + query_token) {
            scores[lid] = -3.402823466e+38F;
        } else {
            int base_k = (b * key_tokens + key_token) * kv_channels + kv_head * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) score += q[base_q + d] * k[base_k + d];
            scores[lid] = score * scale;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float max_score = -3.402823466e+38F;
        for (int kt = 0; kt < key_tokens; ++kt) max_score = fmax(max_score, scores[kt]);
        float denom = 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) {
            float value = (scores[kt] == -3.402823466e+38F) ? 0.0f : exp(scores[kt] - max_score);
            probs[row_base + kt] = value;
            denom += value;
        }
        float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) probs[row_base + kt] *= inv;
    }
}

__kernel void gqa_backward_dp_f32(__global const float* grad_out,
                                  __global const float* v,
                                  __global float* dp,
                                  int batch,
                                  int query_tokens,
                                  int key_tokens,
                                  int n_head,
                                  int n_kv_head,
                                  int head_dim,
                                  int causal,
                                  int query_offset) {
    int gid = get_global_id(0);
    int pairs = batch * n_head * query_tokens * key_tokens;
    if (gid >= pairs) return;
    int key_token = gid % key_tokens;
    int row = gid / key_tokens;
    int query_token = row % query_tokens;
    int bh = row / query_tokens;
    int head = bh % n_head;
    int b = bh / n_head;
    if (causal && key_token > query_offset + query_token) {
        dp[gid] = 0.0f;
        return;
    }
    int group = n_head / n_kv_head;
    int kv_head = head / group;
    int q_channels = n_head * head_dim;
    int kv_channels = n_kv_head * head_dim;
    int base_go = (b * query_tokens + query_token) * q_channels + head * head_dim;
    int base_v = (b * key_tokens + key_token) * kv_channels + kv_head * head_dim;
    float acc = 0.0f;
    for (int d = 0; d < head_dim; ++d) acc += grad_out[base_go + d] * v[base_v + d];
    dp[gid] = acc;
}

// Tuned GQA backward kernels for the common GPT head_dim=64 path.
__kernel void gqa_backward_probs_hd64_f32(__global const float* q,
                                     __global const float* k,
                                     __global float* probs,
                                     int batch,
                                     int query_tokens,
                                     int key_tokens,
                                     int n_head,
                                     int n_kv_head,
                                     int head_dim,
                                     int causal,
                                     int query_offset,
                                     float scale) {
    int lid = get_local_id(0);
    int row = get_group_id(1);
    int rows = batch * n_head * query_tokens;
    if (row >= rows || key_tokens > FA_STAGED_MAX_TOKENS || head_dim != 64) return;

    int query_token = row % query_tokens;
    int bh = row / query_tokens;
    int head = bh % n_head;
    int b = bh / n_head;
    int group = n_head / n_kv_head;
    int kv_head = head / group;
    int q_channels = n_head * 64;
    int kv_channels = n_kv_head * 64;
    int base_q = (b * query_tokens + query_token) * q_channels + head * 64;
    int row_base = row * key_tokens;
    __local float scores[FA_STAGED_MAX_TOKENS];

    if (lid < key_tokens) {
        int key_token = lid;
        if (causal && key_token > query_offset + query_token) {
            scores[lid] = -3.402823466e+38F;
        } else {
            int base_k = (b * key_tokens + key_token) * kv_channels + kv_head * 64;
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < 64; ++d) score += q[base_q + d] * k[base_k + d];
            scores[lid] = score * scale;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float max_score = -3.402823466e+38F;
        for (int kt = 0; kt < key_tokens; ++kt) max_score = fmax(max_score, scores[kt]);
        float denom = 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) {
            float value = (scores[kt] == -3.402823466e+38F) ? 0.0f : exp(scores[kt] - max_score);
            probs[row_base + kt] = value;
            denom += value;
        }
        float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) probs[row_base + kt] *= inv;
    }
}

__kernel void gqa_backward_dp_hd64_f32(__global const float* grad_out,
                                  __global const float* v,
                                  __global float* dp,
                                  int batch,
                                  int query_tokens,
                                  int key_tokens,
                                  int n_head,
                                  int n_kv_head,
                                  int head_dim,
                                  int causal,
                                  int query_offset) {
    int gid = get_global_id(0);
    int pairs = batch * n_head * query_tokens * key_tokens;
    if (gid >= pairs) return;
    int key_token = gid % key_tokens;
    int row = gid / key_tokens;
    int query_token = row % query_tokens;
    int bh = row / query_tokens;
    int head = bh % n_head;
    int b = bh / n_head;
    if (causal && key_token > query_offset + query_token) {
        dp[gid] = 0.0f;
        return;
    }
    int group = n_head / n_kv_head;
    int kv_head = head / group;
    int q_channels = n_head * 64;
    int kv_channels = n_kv_head * 64;
    int base_go = (b * query_tokens + query_token) * q_channels + head * 64;
    int base_v = (b * key_tokens + key_token) * kv_channels + kv_head * 64;
    float acc = 0.0f;
    #pragma unroll
    for (int d = 0; d < 64; ++d) acc += grad_out[base_go + d] * v[base_v + d];
    dp[gid] = acc;
}

// HD64 row-fused staged backward:
//   - reads global softmax probs
//   - computes dp = dO dot V in LDS/private registers
//   - computes ds in LDS and writes it once for the K/V stage
//   - computes dQ immediately from LDS ds, avoiding a separate apply_q kernel
__kernel void gqa_backward_dq_ds_from_probs_hd64_f32(__global const float* probs,
                                                    __global const float* grad_out,
                                                    __global const float* k,
                                                    __global const float* v,
                                                    __global float* ds,
                                                    __global float* grad_q,
                                                    int batch,
                                                    int query_tokens,
                                                    int key_tokens,
                                                    int n_head,
                                                    int n_kv_head,
                                                    int head_dim,
                                                    int causal,
                                                    int query_offset,
                                                    float scale) {
    int lid = get_local_id(0);
    int row = get_group_id(1);
    int rows = batch * n_head * query_tokens;
    if (row >= rows || key_tokens > FA_STAGED_MAX_TOKENS || head_dim != 64) return;

    int query_token = row % query_tokens;
    int bh = row / query_tokens;
    int head = bh % n_head;
    int b = bh / n_head;
    int group = n_head / n_kv_head;
    int kv_head = head / group;
    int q_channels = n_head * 64;
    int kv_channels = n_kv_head * 64;
    int q_head_offset = head * 64;
    int base_go = (b * query_tokens + query_token) * q_channels + q_head_offset;
    int row_base = row * key_tokens;

    __local float dp_local[FA_STAGED_MAX_TOKENS];
    __local float ds_local[FA_STAGED_MAX_TOKENS];
    __local float shared[1];

    if (lid < key_tokens) {
        int key_token = lid;
        if (causal && key_token > query_offset + query_token) {
            dp_local[lid] = 0.0f;
        } else {
            int base_v = (b * key_tokens + key_token) * kv_channels + kv_head * 64;
            float acc = 0.0f;
            #pragma unroll
            for (int d = 0; d < 64; ++d) acc += grad_out[base_go + d] * v[base_v + d];
            dp_local[lid] = acc;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float sum_p_dp = 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) sum_p_dp += probs[row_base + kt] * dp_local[kt];
        shared[0] = sum_p_dp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < key_tokens) {
        int key_token = lid;
        float value = 0.0f;
        if (!(causal && key_token > query_offset + query_token)) {
            const float p = probs[row_base + key_token];
            value = p * (dp_local[lid] - shared[0]) * scale;
        }
        ds_local[lid] = value;
        ds[row_base + key_token] = value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 64) {
        float acc = 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) {
            if (causal && kt > query_offset + query_token) break;
            int base_k = (b * key_tokens + kt) * kv_channels + kv_head * 64;
            acc += ds_local[kt] * k[base_k + lid];
        }
        grad_q[base_go + lid] = acc;
    }
}

// HD64 dQ from materialized probs+dp without materializing global ds.
// Used by the partial dK/dV path below.
__kernel void gqa_backward_dq_from_probs_dp_hd64_f32(__global const float* probs,
                                                    __global const float* dp,
                                                    __global const float* k,
                                                    __global float* grad_q,
                                                    int batch,
                                                    int query_tokens,
                                                    int key_tokens,
                                                    int n_head,
                                                    int n_kv_head,
                                                    int head_dim,
                                                    int causal,
                                                    int query_offset,
                                                    float scale) {
    int lid = get_local_id(0);
    int row = get_group_id(1);
    int rows = batch * n_head * query_tokens;
    if (row >= rows || key_tokens > FA_STAGED_MAX_TOKENS || head_dim != 64) return;

    int query_token = row % query_tokens;
    int bh = row / query_tokens;
    int head = bh % n_head;
    int b = bh / n_head;
    int group = n_head / n_kv_head;
    int kv_head = head / group;
    int q_channels = n_head * 64;
    int kv_channels = n_kv_head * 64;
    int base_q = (b * query_tokens + query_token) * q_channels + head * 64;
    int row_base = row * key_tokens;
    __local float ds_local[FA_STAGED_MAX_TOKENS];
    __local float shared[1];

    if (lid == 0) {
        float sum_p_dp = 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) sum_p_dp += probs[row_base + kt] * dp[row_base + kt];
        shared[0] = sum_p_dp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < key_tokens) {
        int key_token = lid;
        float value = 0.0f;
        if (!(causal && key_token > query_offset + query_token)) {
            const float p = probs[row_base + key_token];
            value = p * (dp[row_base + key_token] - shared[0]) * scale;
        }
        ds_local[lid] = value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 64) {
        float acc = 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) {
            if (causal && kt > query_offset + query_token) break;
            int base_k = (b * key_tokens + kt) * kv_channels + kv_head * 64;
            acc += ds_local[kt] * k[base_k + lid];
        }
        grad_q[base_q + lid] = acc;
    }
}

// Emits per-query-tile partial dK/dV reductions:
// partial layout: [batch, n_kv_head, ceil(query_tokens/tile_q), key_tokens, 64].
// This replaces global ds with compact partial dK/dV buffers and a final reduce.
__kernel void gqa_backward_kv_partial_from_probs_dp_hd64_f32(__global const float* probs,
                                                            __global const float* dp,
                                                            __global const float* q,
                                                            __global const float* grad_out,
                                                            __global float* partial_dk,
                                                            __global float* partial_dv,
                                                            int batch,
                                                            int query_tokens,
                                                            int key_tokens,
                                                            int n_head,
                                                            int n_kv_head,
                                                            int head_dim,
                                                            int causal,
                                                            int query_offset,
                                                            float scale,
                                                            int tile_q,
                                                            int num_tiles) {
    int lid = get_local_id(0);
    int gid1 = get_group_id(1);
    int groups_total = batch * n_kv_head * num_tiles * key_tokens;
    if (gid1 >= groups_total || lid >= 64 || head_dim != 64 || tile_q <= 0) return;

    int key_token = gid1 % key_tokens;
    int tmp = gid1 / key_tokens;
    int tile = tmp % num_tiles;
    tmp /= num_tiles;
    int kv_head = tmp % n_kv_head;
    int b = tmp / n_kv_head;
    int group = n_head / n_kv_head;
    int q_channels = n_head * 64;
    int tile_begin = tile * tile_q;
    int tile_end = min(query_tokens, tile_begin + tile_q);

    __local float shared[1];
    float acc_k = 0.0f;
    float acc_v = 0.0f;

    for (int q_head = kv_head * group; q_head < (kv_head + 1) * group; ++q_head) {
        int q_head_offset = q_head * 64;
        for (int query_token = tile_begin; query_token < tile_end; ++query_token) {
            if (causal && key_token > query_offset + query_token) continue;
            int row = ((b * n_head + q_head) * query_tokens + query_token);
            int row_base = row * key_tokens;
            int base_q = (b * query_tokens + query_token) * q_channels + q_head_offset;

            if (lid == 0) {
                float sum_p_dp = 0.0f;
                for (int kt = 0; kt < key_tokens; ++kt) {
                    sum_p_dp += probs[row_base + kt] * dp[row_base + kt];
                }
                shared[0] = sum_p_dp;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            const float p = probs[row_base + key_token];
            const float ds = p * (dp[row_base + key_token] - shared[0]) * scale;
            acc_k += ds * q[base_q + lid];
            acc_v += p * grad_out[base_q + lid];
        }
    }

    int out = gid1 * 64 + lid;
    partial_dk[out] = acc_k;
    partial_dv[out] = acc_v;
}

__kernel void gqa_backward_kv_reduce_partials_hd64_f32(__global const float* partial_dk,
                                                      __global const float* partial_dv,
                                                      __global float* grad_k,
                                                      __global float* grad_v,
                                                      int batch,
                                                      int key_tokens,
                                                      int n_kv_head,
                                                      int head_dim,
                                                      int num_tiles) {
    int lid = get_local_id(0);
    int gid1 = get_group_id(1);
    int groups_total = batch * n_kv_head * key_tokens;
    if (gid1 >= groups_total || lid >= 64 || head_dim != 64) return;

    int key_token = gid1 % key_tokens;
    int tmp = gid1 / key_tokens;
    int kv_head = tmp % n_kv_head;
    int b = tmp / n_kv_head;
    float acc_k = 0.0f;
    float acc_v = 0.0f;
    for (int tile = 0; tile < num_tiles; ++tile) {
        int partial_group = (((b * n_kv_head + kv_head) * num_tiles + tile) * key_tokens + key_token);
        int partial_idx = partial_group * 64 + lid;
        acc_k += partial_dk[partial_idx];
        acc_v += partial_dv[partial_idx];
    }

    int kv_channels = n_kv_head * 64;
    int out = (b * key_tokens + key_token) * kv_channels + kv_head * 64 + lid;
    grad_k[out] = acc_k;
    grad_v[out] = acc_v;
}

__kernel void gqa_backward_ds_f32(__global const float* probs,
                                  __global const float* dp,
                                  __global float* ds,
                                  int batch,
                                  int query_tokens,
                                  int key_tokens,
                                  int n_head,
                                  int causal,
                                  int query_offset,
                                  float scale) {
    int lid = get_local_id(0);
    int row = get_group_id(1);
    int rows = batch * n_head * query_tokens;
    if (row >= rows || key_tokens > FA_STAGED_MAX_TOKENS) return;
    int query_token = row % query_tokens;
    int row_base = row * key_tokens;
    __local float shared[1];
    if (lid == 0) {
        float sum_p_dp = 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) sum_p_dp += probs[row_base + kt] * dp[row_base + kt];
        shared[0] = sum_p_dp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < key_tokens) {
        int key_token = lid;
        if (causal && key_token > query_offset + query_token) {
            ds[row_base + key_token] = 0.0f;
        } else {
            float p = probs[row_base + key_token];
            ds[row_base + key_token] = p * (dp[row_base + key_token] - shared[0]) * scale;
        }
    }
}

__kernel void gqa_backward_apply_q_f32(__global const float* ds,
                                       __global const float* k,
                                       __global float* grad_q,
                                       int batch,
                                       int query_tokens,
                                       int key_tokens,
                                       int n_head,
                                       int n_kv_head,
                                       int head_dim,
                                       int causal,
                                       int query_offset) {
    int gid = get_global_id(0);
    int q_channels = n_head * head_dim;
    int kv_channels = n_kv_head * head_dim;
    int total = batch * query_tokens * q_channels;
    if (gid >= total) return;
    int d = gid % head_dim;
    int c = gid % q_channels;
    int query_token = (gid / q_channels) % query_tokens;
    int b = gid / (query_tokens * q_channels);
    int head = c / head_dim;
    int group = n_head / n_kv_head;
    int kv_head = head / group;
    int row = ((b * n_head + head) * query_tokens + query_token);
    int row_base = row * key_tokens;
    float acc = 0.0f;
    for (int kt = 0; kt < key_tokens; ++kt) {
        if (causal && kt > query_offset + query_token) break;
        int base_k = (b * key_tokens + kt) * kv_channels + kv_head * head_dim;
        acc += ds[row_base + kt] * k[base_k + d];
    }
    grad_q[gid] = acc;
}

__kernel void gqa_backward_apply_k_f32(__global const float* ds,
                                       __global const float* q,
                                       __global float* grad_k,
                                       int batch,
                                       int query_tokens,
                                       int key_tokens,
                                       int n_head,
                                       int n_kv_head,
                                       int head_dim,
                                       int causal,
                                       int query_offset) {
    int gid = get_global_id(0);
    int kv_channels = n_kv_head * head_dim;
    int q_channels = n_head * head_dim;
    int total = batch * key_tokens * kv_channels;
    if (gid >= total) return;
    int d = gid % head_dim;
    int c = gid % kv_channels;
    int key_token = (gid / kv_channels) % key_tokens;
    int b = gid / (key_tokens * kv_channels);
    int kv_head = c / head_dim;
    int group = n_head / n_kv_head;
    float acc = 0.0f;
    for (int q_head = kv_head * group; q_head < (kv_head + 1) * group; ++q_head) {
        int q_head_offset = q_head * head_dim;
        for (int query_token = 0; query_token < query_tokens; ++query_token) {
            if (causal && key_token > query_offset + query_token) continue;
            int row = ((b * n_head + q_head) * query_tokens + query_token);
            int base_q = (b * query_tokens + query_token) * q_channels + q_head_offset;
            acc += ds[row * key_tokens + key_token] * q[base_q + d];
        }
    }
    grad_k[gid] = acc;
}

__kernel void gqa_backward_apply_v_f32(__global const float* probs,
                                       __global const float* grad_out,
                                       __global float* grad_v,
                                       int batch,
                                       int query_tokens,
                                       int key_tokens,
                                       int n_head,
                                       int n_kv_head,
                                       int head_dim,
                                       int causal,
                                       int query_offset) {
    int gid = get_global_id(0);
    int kv_channels = n_kv_head * head_dim;
    int q_channels = n_head * head_dim;
    int total = batch * key_tokens * kv_channels;
    if (gid >= total) return;
    int d = gid % head_dim;
    int c = gid % kv_channels;
    int key_token = (gid / kv_channels) % key_tokens;
    int b = gid / (key_tokens * kv_channels);
    int kv_head = c / head_dim;
    int group = n_head / n_kv_head;
    float acc = 0.0f;
    for (int q_head = kv_head * group; q_head < (kv_head + 1) * group; ++q_head) {
        int q_head_offset = q_head * head_dim;
        for (int query_token = 0; query_token < query_tokens; ++query_token) {
            if (causal && key_token > query_offset + query_token) continue;
            int row = ((b * n_head + q_head) * query_tokens + query_token);
            int base_go = (b * query_tokens + query_token) * q_channels + q_head_offset;
            acc += probs[row * key_tokens + key_token] * grad_out[base_go + d];
        }
    }
    grad_v[gid] = acc;
}

__kernel void gqa_backward_apply_kv_hd64_f32(__global const float* ds,
                                             __global const float* probs,
                                             __global const float* q,
                                             __global const float* grad_out,
                                             __global float* grad_k,
                                             __global float* grad_v,
                                             int batch,
                                             int query_tokens,
                                             int key_tokens,
                                             int n_head,
                                             int n_kv_head,
                                             int head_dim,
                                             int causal,
                                             int query_offset) {
    int gid = get_global_id(0);
    int kv_channels = n_kv_head * 64;
    int q_channels = n_head * 64;
    int total = batch * key_tokens * kv_channels;
    if (gid >= total || head_dim != 64) return;
    int d = gid & 63;
    int c = gid % kv_channels;
    int key_token = (gid / kv_channels) % key_tokens;
    int b = gid / (key_tokens * kv_channels);
    int kv_head = c / 64;
    int group = n_head / n_kv_head;
    float acc_k = 0.0f;
    float acc_v = 0.0f;
    for (int q_head = kv_head * group; q_head < (kv_head + 1) * group; ++q_head) {
        int q_head_offset = q_head * 64;
        for (int query_token = 0; query_token < query_tokens; ++query_token) {
            if (causal && key_token > query_offset + query_token) continue;
            int row = ((b * n_head + q_head) * query_tokens + query_token);
            int base_q = (b * query_tokens + query_token) * q_channels + q_head_offset;
            acc_k += ds[row * key_tokens + key_token] * q[base_q + d];
            acc_v += probs[row * key_tokens + key_token] * grad_out[base_q + d];
        }
    }
    grad_k[gid] = acc_k;
    grad_v[gid] = acc_v;
}

// HD64 GQA backward path that does not materialize global probs/dp/ds tensors.
// It recomputes row softmax inside the apply kernels: more ALU, less VRAM
// traffic and no temporary attention-backward tensors in captured graphs.
__kernel void gqa_backward_dq_no_tmp_hd64_f32(__global const float* q,
                                             __global const float* k,
                                             __global const float* v,
                                             __global const float* grad_out,
                                             __global float* grad_q,
                                             int batch,
                                             int query_tokens,
                                             int key_tokens,
                                             int n_head,
                                             int n_kv_head,
                                             int head_dim,
                                             int causal,
                                             int query_offset,
                                             float scale) {
    int lid = get_local_id(0);
    int row = get_group_id(1);
    int rows = batch * n_head * query_tokens;
    if (row >= rows || key_tokens > FA_STAGED_MAX_TOKENS || head_dim != 64) return;

    int query_token = row % query_tokens;
    int bh = row / query_tokens;
    int head = bh % n_head;
    int b = bh / n_head;
    int group = n_head / n_kv_head;
    int kv_head = head / group;
    int q_channels = n_head * 64;
    int kv_channels = n_kv_head * 64;
    int base_q = (b * query_tokens + query_token) * q_channels + head * 64;
    int base_go = base_q;

    __local float scores[FA_STAGED_MAX_TOKENS];
    __local float probs[FA_STAGED_MAX_TOKENS];
    __local float dp[FA_STAGED_MAX_TOKENS];
    __local float shared[1];

    if (lid < key_tokens) {
        int key_token = lid;
        if (causal && key_token > query_offset + query_token) {
            scores[lid] = -3.402823466e+38F;
            dp[lid] = 0.0f;
        } else {
            int base_k = (b * key_tokens + key_token) * kv_channels + kv_head * 64;
            int base_v = base_k;
            float score = 0.0f;
            float dp_acc = 0.0f;
            #pragma unroll
            for (int d = 0; d < 64; ++d) {
                score += q[base_q + d] * k[base_k + d];
                dp_acc += grad_out[base_go + d] * v[base_v + d];
            }
            scores[lid] = score * scale;
            dp[lid] = dp_acc;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        float max_score = -3.402823466e+38F;
        for (int kt = 0; kt < key_tokens; ++kt) max_score = fmax(max_score, scores[kt]);
        float denom = 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) {
            float value = (scores[kt] == -3.402823466e+38F) ? 0.0f : exp(scores[kt] - max_score);
            probs[kt] = value;
            denom += value;
        }
        float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        float sum_p_dp = 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) {
            probs[kt] *= inv;
            sum_p_dp += probs[kt] * dp[kt];
        }
        shared[0] = sum_p_dp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 64) {
        float acc = 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) {
            if (causal && kt > query_offset + query_token) break;
            int base_k = (b * key_tokens + kt) * kv_channels + kv_head * 64;
            float ds = probs[kt] * (dp[kt] - shared[0]) * scale;
            acc += ds * k[base_k + lid];
        }
        grad_q[base_q + lid] = acc;
    }
}

__kernel void gqa_backward_kv_no_tmp_hd64_f32(__global const float* q,
                                             __global const float* k,
                                             __global const float* v,
                                             __global const float* grad_out,
                                             __global float* grad_k,
                                             __global float* grad_v,
                                             int batch,
                                             int query_tokens,
                                             int key_tokens,
                                             int n_head,
                                             int n_kv_head,
                                             int head_dim,
                                             int causal,
                                             int query_offset,
                                             float scale) {
    int lid = get_local_id(0);
    int group_id = get_group_id(1);
    int groups_total = batch * n_kv_head * key_tokens;
    if (group_id >= groups_total || key_tokens > FA_STAGED_MAX_TOKENS || head_dim != 64 || lid >= 64) return;

    int key_token = group_id % key_tokens;
    int bkv = group_id / key_tokens;
    int kv_head = bkv % n_kv_head;
    int b = bkv / n_kv_head;
    int q_channels = n_head * 64;
    int kv_channels = n_kv_head * 64;
    int base_out = (b * key_tokens + key_token) * kv_channels + kv_head * 64;
    int group = n_head / n_kv_head;

    __local float scores[FA_STAGED_MAX_TOKENS];
    __local float dpvals[FA_STAGED_MAX_TOKENS];
    __local float reduce[64];
    __local float shared[2];

    float acc_k = 0.0f;
    float acc_v = 0.0f;
    for (int q_head = kv_head * group; q_head < (kv_head + 1) * group; ++q_head) {
        int q_head_offset = q_head * 64;
        for (int query_token = 0; query_token < query_tokens; ++query_token) {
            if (causal && key_token > query_offset + query_token) continue;
            int base_q = (b * query_tokens + query_token) * q_channels + q_head_offset;
            int base_go = base_q;

            for (int kt = 0; kt < key_tokens; ++kt) {
                if (causal && kt > query_offset + query_token) {
                    if (lid == 0) {
                        scores[kt] = -3.402823466e+38F;
                        dpvals[kt] = 0.0f;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    continue;
                }

                int base_k = (b * key_tokens + kt) * kv_channels + kv_head * 64;
                int base_v = base_k;
                reduce[lid] = q[base_q + lid] * k[base_k + lid];
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int stride = 32; stride > 0; stride >>= 1) {
                    if (lid < stride) reduce[lid] += reduce[lid + stride];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (lid == 0) scores[kt] = reduce[0] * scale;
                barrier(CLK_LOCAL_MEM_FENCE);

                reduce[lid] = grad_out[base_go + lid] * v[base_v + lid];
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int stride = 32; stride > 0; stride >>= 1) {
                    if (lid < stride) reduce[lid] += reduce[lid + stride];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (lid == 0) dpvals[kt] = reduce[0];
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                float max_score = -3.402823466e+38F;
                for (int kt = 0; kt < key_tokens; ++kt) max_score = fmax(max_score, scores[kt]);
                float denom = 0.0f;
                float numer_target = 0.0f;
                float dp_target = 0.0f;
                float sum_numer_dp = 0.0f;
                for (int kt = 0; kt < key_tokens; ++kt) {
                    float numer = (scores[kt] == -3.402823466e+38F) ? 0.0f : exp(scores[kt] - max_score);
                    denom += numer;
                    sum_numer_dp += numer * dpvals[kt];
                    if (kt == key_token) {
                        numer_target = numer;
                        dp_target = dpvals[kt];
                    }
                }
                float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
                float p_target = numer_target * inv;
                float sum_p_dp = sum_numer_dp * inv;
                shared[0] = p_target;
                shared[1] = p_target * (dp_target - sum_p_dp) * scale;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            acc_k += shared[1] * q[base_q + lid];
            acc_v += shared[0] * grad_out[base_go + lid];
        }
    }

    grad_k[base_out + lid] = acc_k;
    grad_v[base_out + lid] = acc_v;
}

__kernel void rope_f32(__global const float* x,
                       __global float* out,
                       int batch,
                       int tokens,
                       int channels,
                       int n_head,
                       int head_dim,
                       int rotary_dim,
                       int token_offset,
                       float theta,
                       int inverse) {
    int gid = get_global_id(0);
    int total = batch * tokens * channels;
    if (gid >= total) return;
    int d = gid % head_dim;
    int c = gid % channels;
    int token = (gid / channels) % tokens;
    int b = gid / (tokens * channels);
    int head = c / head_dim;
    if (head >= n_head) return;
    int rd = rotary_dim <= 0 ? head_dim : rotary_dim;
    rd = min(rd, head_dim);
    rd = rd - (rd & 1);
    if (d >= rd) {
        out[gid] = x[gid];
        return;
    }
    int pair_d = d & ~1;
    int pair_index = pair_d >> 1;
    float exponent = (float)pair_d / (float)head_dim;
    float angle = (float)(token_offset + token) / pow(theta, exponent);
    if (inverse) angle = -angle;
    float cs = cos(angle);
    float sn = sin(angle);
    int base = (b * tokens + token) * channels + head * head_dim;
    float even = x[base + pair_d];
    float odd = x[base + pair_d + 1];
    out[gid] = (d == pair_d) ? (even * cs - odd * sn) : (even * sn + odd * cs);
}

__kernel void rope_positions_f32(__global const float* x,
                                 __global const int* positions,
                                 __global float* out,
                                 int batch,
                                 int tokens,
                                 int channels,
                                 int n_head,
                                 int head_dim,
                                 int rotary_dim,
                                 float theta) {
    int gid = get_global_id(0);
    int total = batch * tokens * channels;
    if (gid >= total) return;
    int d = gid % head_dim;
    int c = gid % channels;
    int token = (gid / channels) % tokens;
    int b = gid / (tokens * channels);
    int head = c / head_dim;
    if (head >= n_head) return;
    int rd = rotary_dim <= 0 ? head_dim : rotary_dim;
    rd = min(rd, head_dim);
    rd = rd - (rd & 1);
    if (d >= rd) {
        out[gid] = x[gid];
        return;
    }
    int pos = positions[b * tokens + token];
    if (pos < 0) pos = 0;
    int pair_d = d & ~1;
    float exponent = (float)pair_d / (float)head_dim;
    float angle = (float)pos / pow(theta, exponent);
    float cs = cos(angle);
    float sn = sin(angle);
    int base = (b * tokens + token) * channels + head * head_dim;
    float even = x[base + pair_d];
    float odd = x[base + pair_d + 1];
    out[gid] = (d == pair_d) ? (even * cs - odd * sn) : (even * sn + odd * cs);
}

// Decode-only fused Q/K RMSNorm + RoPE for one generated token.
// One workgroup processes one (batch, q-head or kv-head). This replaces:
//   rmsnorm(q heads) + rmsnorm(k heads) + rope(q) + rope(k)
// with a single launch and keeps the same [B, heads*head_dim] layouts.
__kernel void qk_norm_rope_decode_f32(__global const float* q,
                                      __global const float* k,
                                      __global const float* q_weight,
                                      __global const float* k_weight,
                                      __global float* q_out,
                                      __global float* k_out,
                                      int batch,
                                      int n_head,
                                      int n_kv_head,
                                      int head_dim,
                                      int rotary_dim,
                                      int token_offset,
                                      float theta,
                                      float q_eps,
                                      float k_eps,
                                      __local float* scratch) {
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    const int group = get_group_id(0);
    const int heads_total = n_head + n_kv_head;
    const int b = group / heads_total;
    const int local_head = group - b * heads_total;
    if (b >= batch || local_head >= heads_total) return;

    const int is_q = local_head < n_head;
    const int head = is_q ? local_head : (local_head - n_head);
    const int channels = (is_q ? n_head : n_kv_head) * head_dim;
    __global const float* src = is_q ? q : k;
    __global const float* weight = is_q ? q_weight : k_weight;
    __global float* dst = is_q ? q_out : k_out;
    const float eps = is_q ? q_eps : k_eps;
    const int base = b * channels + head * head_dim;

    float ss = 0.0f;
    for (int d = lid; d < head_dim; d += local_size) {
        const float v = src[base + d];
        ss += v * v;
    }
    scratch[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = rsqrt(scratch[0] / (float)head_dim + eps);

    int rd = rotary_dim <= 0 ? head_dim : rotary_dim;
    rd = min(rd, head_dim);
    rd = rd - (rd & 1);
    for (int d = lid; d < head_dim; d += local_size) {
        if (d >= rd) {
            dst[base + d] = src[base + d] * inv * weight[d];
            continue;
        }
        const int pair_d = d & ~1;
        const float exponent = (float)pair_d / (float)head_dim;
        const float angle = (float)token_offset / pow(theta, exponent);
        const float cs = cos(angle);
        const float sn = sin(angle);
        const float even = src[base + pair_d] * inv * weight[pair_d];
        const float odd = src[base + pair_d + 1] * inv * weight[pair_d + 1];
        dst[base + d] = (d == pair_d) ? (even * cs - odd * sn) : (even * sn + odd * cs);
    }
}


__kernel void qk_norm_rope_cache_append_decode_f32(__global const float* q,
                                                   __global const float* k,
                                                   __global const float* v,
                                                   __global const float* q_weight,
                                                   __global const float* k_weight,
                                                   __global float* q_out,
                                                   __global float* cache_k,
                                                   __global float* cache_v,
                                                   int batch,
                                                   int n_head,
                                                   int n_kv_head,
                                                   int head_dim,
                                                   int rotary_dim,
                                                   int token_offset,
                                                   int max_tokens,
                                                   float theta,
                                                   float q_eps,
                                                   float k_eps,
                                                   __local float* scratch) {
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    const int group = get_group_id(0);
    const int kv_channels = n_kv_head * head_dim;
    const int heads_total = n_head + 2 * n_kv_head;
    const int b = group / heads_total;
    const int local_head = group - b * heads_total;
    if (b >= batch || local_head >= heads_total) return;

    if (local_head >= n_head + n_kv_head) {
        const int head = local_head - n_head - n_kv_head;
        const int src_base = b * kv_channels + head * head_dim;
        const int dst_base = (b * max_tokens + token_offset) * kv_channels + head * head_dim;
        for (int d = lid; d < head_dim; d += local_size) {
            cache_v[dst_base + d] = v[src_base + d];
        }
        return;
    }

    const int is_q = local_head < n_head;
    const int head = is_q ? local_head : (local_head - n_head);
    const int channels = (is_q ? n_head : n_kv_head) * head_dim;
    __global const float* src = is_q ? q : k;
    __global const float* weight = is_q ? q_weight : k_weight;
    const float eps = is_q ? q_eps : k_eps;
    const int src_base = b * channels + head * head_dim;

    float ss = 0.0f;
    for (int d = lid; d < head_dim; d += local_size) {
        const float sv = src[src_base + d];
        ss += sv * sv;
    }
    scratch[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv = rsqrt(scratch[0] / (float)head_dim + eps);

    int rd = rotary_dim <= 0 ? head_dim : rotary_dim;
    rd = min(rd, head_dim);
    rd = rd - (rd & 1);
    const int dst_base = is_q
        ? src_base
        : ((b * max_tokens + token_offset) * kv_channels + head * head_dim);
    for (int d = lid; d < head_dim; d += local_size) {
        float value;
        if (d >= rd) {
            value = src[src_base + d] * inv * weight[d];
        } else {
            const int pair_d = d & ~1;
            const float exponent = (float)pair_d / (float)head_dim;
            const float angle = (float)token_offset / pow(theta, exponent);
            const float cs = cos(angle);
            const float sn = sin(angle);
            const float even = src[src_base + pair_d] * inv * weight[pair_d];
            const float odd = src[src_base + pair_d + 1] * inv * weight[pair_d + 1];
            value = (d == pair_d) ? (even * cs - odd * sn) : (even * sn + odd * cs);
        }
        if (is_q) q_out[dst_base + d] = value;
        else cache_k[dst_base + d] = value;
    }
}

// Decode-only fused RoPE(q) + RoPE(k) + KV append for one generated token when
// the model has no Q/K RMSNorm. This replaces two rope launches plus
// kv_cache_append with one launch and keeps q in [B, n_head*head_dim].
__kernel void rope_cache_append_decode_f32(__global const float* q,
                                           __global const float* k,
                                           __global const float* v,
                                           __global float* q_out,
                                           __global float* cache_k,
                                           __global float* cache_v,
                                           int batch,
                                           int n_head,
                                           int n_kv_head,
                                           int head_dim,
                                           int rotary_dim,
                                           int token_offset,
                                           int max_tokens,
                                           float theta,
                                           int total) {
    const int gid = get_global_id(0);
    if (gid >= total) return;
    const int q_channels = n_head * head_dim;
    const int kv_channels = n_kv_head * head_dim;
    const int span = q_channels + 2 * kv_channels;
    const int b = gid / span;
    const int r = gid - b * span;
    if (b >= batch) return;

    int rd = rotary_dim <= 0 ? head_dim : rotary_dim;
    rd = min(rd, head_dim);
    rd = rd - (rd & 1);

    if (r < q_channels) {
        const int c = r;
        const int d = c % head_dim;
        const int head_base = b * q_channels + (c / head_dim) * head_dim;
        if (d >= rd) {
            q_out[head_base + d] = q[head_base + d];
            return;
        }
        const int pair_d = d & ~1;
        const float exponent = (float)pair_d / (float)head_dim;
        const float angle = (float)token_offset / pow(theta, exponent);
        const float cs = cos(angle);
        const float sn = sin(angle);
        const float even = q[head_base + pair_d];
        const float odd = q[head_base + pair_d + 1];
        q_out[head_base + d] = (d == pair_d) ? (even * cs - odd * sn) : (even * sn + odd * cs);
        return;
    }

    if (r < q_channels + kv_channels) {
        const int c = r - q_channels;
        const int d = c % head_dim;
        const int src_base = b * kv_channels + (c / head_dim) * head_dim;
        const int dst_base = (b * max_tokens + token_offset) * kv_channels + (c / head_dim) * head_dim;
        float value;
        if (d >= rd) {
            value = k[src_base + d];
        } else {
            const int pair_d = d & ~1;
            const float exponent = (float)pair_d / (float)head_dim;
            const float angle = (float)token_offset / pow(theta, exponent);
            const float cs = cos(angle);
            const float sn = sin(angle);
            const float even = k[src_base + pair_d];
            const float odd = k[src_base + pair_d + 1];
            value = (d == pair_d) ? (even * cs - odd * sn) : (even * sn + odd * cs);
        }
        cache_k[dst_base + d] = value;
        return;
    }

    const int c = r - q_channels - kv_channels;
    const int dst = (b * max_tokens + token_offset) * kv_channels + c;
    cache_v[dst] = v[b * kv_channels + c];
}

__kernel void qkv_split_f32(__global const float* packed,
                            __global float* q,
                            __global float* k,
                            __global float* v,
                            int rows,
                            int q_dim,
                            int kv_dim) {
    int gid = get_global_id(0);
    int max_dim = max(q_dim, kv_dim);
    int n = rows * max_dim;
    if (gid >= n) return;
    int row = gid / max_dim;
    int d = gid - row * max_dim;
    int packed_base = row * (q_dim + 2 * kv_dim);
    if (d < q_dim) q[row * q_dim + d] = packed[packed_base + d];
    if (d < kv_dim) {
        k[row * kv_dim + d] = packed[packed_base + q_dim + d];
        v[row * kv_dim + d] = packed[packed_base + q_dim + kv_dim + d];
    }
}

__kernel void qkv_split_backward_f32(__global const float* grad_part,
                                     __global float* grad_packed,
                                     int rows,
                                     int q_dim,
                                     int kv_dim,
                                     int part) {
    int gid = get_global_id(0);
    int total_dim = q_dim + 2 * kv_dim;
    int n = rows * total_dim;
    if (gid >= n) return;
    int row = gid / total_dim;
    int d = gid - row * total_dim;
    float value = 0.0f;
    if (part == 0 && d < q_dim) value = grad_part[row * q_dim + d];
    if (part == 1 && d >= q_dim && d < q_dim + kv_dim) value = grad_part[row * kv_dim + (d - q_dim)];
    if (part == 2 && d >= q_dim + kv_dim) value = grad_part[row * kv_dim + (d - q_dim - kv_dim)];
    grad_packed[gid] = value;
}

__kernel void kv_cache_append_f32(__global const float* new_k,
                                  __global const float* new_v,
                                  __global float* cache_k,
                                  __global float* cache_v,
                                  int batch,
                                  int new_tokens,
                                  int max_tokens,
                                  int kv_channels,
                                  int start_pos,
                                  int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    int c = gid % kv_channels;
    int t = (gid / kv_channels) % new_tokens;
    int b = gid / (new_tokens * kv_channels);
    int dst = (b * max_tokens + start_pos + t) * kv_channels + c;
    cache_k[dst] = new_k[gid];
    cache_v[dst] = new_v[gid];
}

__kernel void kv_cache_append_positions_f32(__global const float* new_k,
                                            __global const float* new_v,
                                            __global const int* positions,
                                            __global float* cache_k,
                                            __global float* cache_v,
                                            int batch,
                                            int new_tokens,
                                            int max_tokens,
                                            int kv_channels,
                                            int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    int c = gid % kv_channels;
    int t = (gid / kv_channels) % new_tokens;
    int b = gid / (new_tokens * kv_channels);
    int pos = positions[b * new_tokens + t];
    if (pos < 0 || pos >= max_tokens) return;
    int dst = (b * max_tokens + pos) * kv_channels + c;
    cache_k[dst] = new_k[gid];
    cache_v[dst] = new_v[gid];
}

__kernel void paged_kv_cache_append_f32(__global const float* new_k,
                                        __global const float* new_v,
                                        __global const int* page_table,
                                        __global float* cache_k_pages,
                                        __global float* cache_v_pages,
                                        int batch,
                                        int new_tokens,
                                        int page_size,
                                        int page_count,
                                        int kv_channels,
                                        int start_pos,
                                        int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    int c = gid % kv_channels;
    int t = (gid / kv_channels) % new_tokens;
    int b = gid / (new_tokens * kv_channels);
    int abs_pos = start_pos + t;
    int logical_page = (abs_pos / page_size) % page_count;
    int slot = abs_pos - (abs_pos / page_size) * page_size;
    int phys_page = page_table[b * page_count + logical_page];
    if (phys_page < 0) return;
    int dst = (phys_page * page_size + slot) * kv_channels + c;
    cache_k_pages[dst] = new_k[gid];
    cache_v_pages[dst] = new_v[gid];
}

__kernel void paged_grouped_query_attention_f32(__global const float* q,
                                                __global const float* k_pages,
                                                __global const float* v_pages,
                                                __global const int* page_table,
                                                __global float* out,
                                                int batch,
                                                int query_tokens,
                                                int key_tokens,
                                                int n_head,
                                                int n_kv_head,
                                                int head_dim,
                                                int causal,
                                                int query_abs_start,
                                                int key_abs_start,
                                                int page_size,
                                                int page_count,
                                                int sliding_window,
                                                float scale) {
    int gid = get_global_id(0);
    int q_channels = n_head * head_dim;
    int kv_channels = n_kv_head * head_dim;
    int total = batch * query_tokens * q_channels;
    if (gid >= total) return;
    int d = gid % head_dim;
    int c = gid % q_channels;
    int query_token = (gid / q_channels) % query_tokens;
    int b = gid / (query_tokens * q_channels);
    int q_head = c / head_dim;
    int group = n_head / n_kv_head;
    int kv_head = q_head / group;
    int q_base = (b * query_tokens + query_token) * q_channels + q_head * head_dim;
    int kv_head_offset = kv_head * head_dim;
    int abs_query = query_abs_start + query_token;
    int min_key = key_abs_start;
    if (sliding_window > 0) {
        int window_min = abs_query - sliding_window + 1;
        if (window_min > min_key) min_key = window_min;
    }

    float max_score = -3.402823466e+38F;
    int valid = 0;
    for (int kt = 0; kt < key_tokens; ++kt) {
        int abs_key = key_abs_start + kt;
        if (abs_key < min_key) continue;
        if (causal && abs_key > abs_query) continue;
        int logical_page = (abs_key / page_size) % page_count;
        int slot = abs_key - (abs_key / page_size) * page_size;
        int phys_page = page_table[b * page_count + logical_page];
        int k_base = (phys_page * page_size + slot) * kv_channels + kv_head_offset;
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) score += q[q_base + i] * k_pages[k_base + i];
        max_score = fmax(max_score, score * scale);
        ++valid;
    }
    if (valid == 0) {
        out[gid] = 0.0f;
        return;
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (int kt = 0; kt < key_tokens; ++kt) {
        int abs_key = key_abs_start + kt;
        if (abs_key < min_key) continue;
        if (causal && abs_key > abs_query) continue;
        int logical_page = (abs_key / page_size) % page_count;
        int slot = abs_key - (abs_key / page_size) * page_size;
        int phys_page = page_table[b * page_count + logical_page];
        int k_base = (phys_page * page_size + slot) * kv_channels + kv_head_offset;
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) score += q[q_base + i] * k_pages[k_base + i];
        float prob = exp(score * scale - max_score);
        denom += prob;
        acc += prob * v_pages[(phys_page * page_size + slot) * kv_channels + kv_head_offset + d];
    }
    out[gid] = denom > 0.0f ? acc / denom : 0.0f;
}

__kernel void grouped_query_attention_f32(__global const float* q,
                                          __global const float* k,
                                          __global const float* v,
                                          __global float* out,
                                          int batch,
                                          int query_tokens,
                                          int key_tokens,
                                          int n_head,
                                          int n_kv_head,
                                          int head_dim,
                                          int causal,
                                          int query_offset,
                                          int key_stride,
                                          float scale) {
    int gid = get_global_id(0);
    int q_channels = n_head * head_dim;
    int kv_channels = n_kv_head * head_dim;
    int total = batch * query_tokens * q_channels;
    if (gid >= total) return;
    int d = gid % head_dim;
    int c = gid % q_channels;
    int query_token = (gid / q_channels) % query_tokens;
    int b = gid / (query_tokens * q_channels);
    int q_head = c / head_dim;
    int group = n_head / n_kv_head;
    int kv_head = q_head / group;
    int q_base = (b * query_tokens + query_token) * q_channels + q_head * head_dim;
    int kv_head_offset = kv_head * head_dim;

    float max_score = -3.402823466e+38F;
    for (int kt = 0; kt < key_tokens; ++kt) {
        if (causal && kt > query_offset + query_token) continue;
        int k_base = (b * key_stride + kt) * kv_channels + kv_head_offset;
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) score += q[q_base + i] * k[k_base + i];
        max_score = fmax(max_score, score * scale);
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (int kt = 0; kt < key_tokens; ++kt) {
        if (causal && kt > query_offset + query_token) continue;
        int k_base = (b * key_stride + kt) * kv_channels + kv_head_offset;
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) score += q[q_base + i] * k[k_base + i];
        float prob = exp(score * scale - max_score);
        denom += prob;
        acc += prob * v[(b * key_stride + kt) * kv_channels + kv_head_offset + d];
    }
    out[gid] = acc / denom;
}

__kernel void grouped_query_attention_windowed_f32(__global const float* q,
                                                   __global const float* k,
                                                   __global const float* v,
                                                   __global float* out,
                                                   int batch,
                                                   int query_tokens,
                                                   int key_tokens,
                                                   int n_head,
                                                   int n_kv_head,
                                                   int head_dim,
                                                   int causal,
                                                   int query_offset,
                                                   int sliding_window,
                                                   int key_stride,
                                                   float scale) {
    int gid = get_global_id(0);
    int q_channels = n_head * head_dim;
    int kv_channels = n_kv_head * head_dim;
    int total = batch * query_tokens * q_channels;
    if (gid >= total) return;
    int d = gid % head_dim;
    int c = gid % q_channels;
    int query_token = (gid / q_channels) % query_tokens;
    int b = gid / (query_tokens * q_channels);
    int q_head = c / head_dim;
    int group = n_head / n_kv_head;
    int kv_head = q_head / group;
    int q_base = (b * query_tokens + query_token) * q_channels + q_head * head_dim;
    int kv_head_offset = kv_head * head_dim;
    int abs_query = query_offset + query_token;
    int min_key = sliding_window > 0 ? abs_query - sliding_window + 1 : 0;
    if (min_key < 0) min_key = 0;

    float max_score = -3.402823466e+38F;
    int valid = 0;
    for (int kt = 0; kt < key_tokens; ++kt) {
        if (causal && kt > abs_query) continue;
        if (kt < min_key) continue;
        int k_base = (b * key_stride + kt) * kv_channels + kv_head_offset;
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) score += q[q_base + i] * k[k_base + i];
        max_score = fmax(max_score, score * scale);
        ++valid;
    }
    if (valid == 0) {
        out[gid] = 0.0f;
        return;
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (int kt = 0; kt < key_tokens; ++kt) {
        if (causal && kt > abs_query) continue;
        if (kt < min_key) continue;
        int k_base = (b * key_stride + kt) * kv_channels + kv_head_offset;
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) score += q[q_base + i] * k[k_base + i];
        float prob = exp(score * scale - max_score);
        denom += prob;
        acc += prob * v[(b * key_stride + kt) * kv_channels + kv_head_offset + d];
    }
    out[gid] = denom > 0.0f ? acc / denom : 0.0f;
}

__kernel void grouped_query_attention_prefill_wg_f32(__global const float* q,
                                                     __global const float* k,
                                                     __global const float* v,
                                                     __global float* out,
                                                     int batch,
                                                     int query_tokens,
                                                     int key_tokens,
                                                     int n_head,
                                                     int n_kv_head,
                                                     int head_dim,
                                                     int causal,
                                                     int query_offset,
                                                     int sliding_window,
                                                     int key_stride,
                                                     float scale,
                                                     __local float* scratch) {
    int group_id = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    int query_token = group_id % query_tokens;
    int head_batch = group_id / query_tokens;
    int q_head = head_batch % n_head;
    int b = head_batch / n_head;
    if (b >= batch) return;

    int q_channels = n_head * head_dim;
    int kv_channels = n_kv_head * head_dim;
    int group = n_head / n_kv_head;
    int kv_head = q_head / group;
    int q_base = (b * query_tokens + query_token) * q_channels + q_head * head_dim;
    int kv_head_offset = kv_head * head_dim;
    int abs_query = query_offset + query_token;

    int min_key = 0;
    if (sliding_window > 0) {
        min_key = abs_query - sliding_window + 1;
        if (min_key < 0) min_key = 0;
    }
    int max_key = key_tokens - 1;
    if (causal && abs_query < max_key) max_key = abs_query;
    int valid = max_key >= min_key ? (max_key - min_key + 1) : 0;

    if (valid <= 0) {
        int out_base_zero = (b * query_tokens + query_token) * q_channels + q_head * head_dim;
        for (int d = lid; d < head_dim; d += local_size) out[out_base_zero + d] = 0.0f;
        return;
    }

    __local float* scores = scratch;
    __local float* red = scratch + valid;

    for (int idx = lid; idx < valid; idx += local_size) {
        int kt = min_key + idx;
        int k_base = (b * key_stride + kt) * kv_channels + kv_head_offset;
        float acc_dot = 0.0f;
        int i = 0;
        for (; i + 3 < head_dim; i += 4) {
            float4 qv = vload4(0, q + q_base + i);
            float4 kv = vload4(0, k + k_base + i);
            acc_dot += dot(qv, kv);
        }
        for (; i < head_dim; ++i) acc_dot += q[q_base + i] * k[k_base + i];
        scores[idx] = acc_dot * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float local_max = -3.402823466e+38F;
    for (int idx = lid; idx < valid; idx += local_size) {
        local_max = fmax(local_max, scores[idx]);
    }
    red[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) red[lid] = fmax(red[lid], red[lid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_score = red[0];

    float local_denom = 0.0f;
    for (int idx = lid; idx < valid; idx += local_size) {
        float p = exp(scores[idx] - max_score);
        scores[idx] = p;
        local_denom += p;
    }
    red[lid] = local_denom;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) red[lid] += red[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float denom = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    int out_base = (b * query_tokens + query_token) * q_channels + q_head * head_dim;
    for (int d = lid; d < head_dim; d += local_size) {
        float acc = 0.0f;
        if (denom > 0.0f) {
            for (int idx = 0; idx < valid; ++idx) {
                int kt = min_key + idx;
                acc += scores[idx] * v[(b * key_stride + kt) * kv_channels + kv_head_offset + d];
            }
            acc /= denom;
        }
        out[out_base + d] = acc;
    }
}

__kernel void grouped_query_attention_decode_f32(__global const float* q,
                                                 __global const float* k,
                                                 __global const float* v,
                                                 __global float* out,
                                                 int batch,
                                                 int key_tokens,
                                                 int key_stride,
                                                 int n_head,
                                                 int n_kv_head,
                                                 int head_dim,
                                                 int query_offset,
                                                 int sliding_window,
                                                 float scale,
                                                 __local float* scratch) {
    int group_id = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    int q_head = group_id % n_head;
    int b = group_id / n_head;
    if (b >= batch) return;

    int q_channels = n_head * head_dim;
    int kv_channels = n_kv_head * head_dim;
    int group = n_head / n_kv_head;
    int kv_head = q_head / group;
    int q_base = b * q_channels + q_head * head_dim;
    int kv_head_offset = kv_head * head_dim;
    int abs_query = query_offset;
    int min_key = sliding_window > 0 ? abs_query - sliding_window + 1 : 0;
    if (min_key < 0) min_key = 0;
    int max_key = abs_query < key_tokens ? abs_query : (key_tokens - 1);

    __local float* scores = scratch;
    __local float* red = scratch + key_tokens;

    for (int kt = lid; kt < key_tokens; kt += local_size) {
        float score = -3.402823466e+38F;
        if (kt >= min_key && kt <= max_key) {
            int k_base = (b * key_stride + kt) * kv_channels + kv_head_offset;
            float acc_dot = 0.0f;
            int i = 0;
            for (; i + 3 < head_dim; i += 4) {
                float4 qv = vload4(0, q + q_base + i);
                float4 kv = vload4(0, k + k_base + i);
                acc_dot += dot(qv, kv);
            }
            for (; i < head_dim; ++i) acc_dot += q[q_base + i] * k[k_base + i];
            score = acc_dot * scale;
        }
        scores[kt] = score;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float local_max = -3.402823466e+38F;
    for (int kt = lid; kt < key_tokens; kt += local_size) {
        local_max = fmax(local_max, scores[kt]);
    }
    red[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) red[lid] = fmax(red[lid], red[lid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float max_score = red[0];

    float local_denom = 0.0f;
    for (int kt = lid; kt < key_tokens; kt += local_size) {
        float p = 0.0f;
        if (scores[kt] > -3.0e+38F) p = exp(scores[kt] - max_score);
        scores[kt] = p;
        local_denom += p;
    }
    red[lid] = local_denom;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) red[lid] += red[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float denom = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    int out_base = b * q_channels + q_head * head_dim;
    for (int d = lid; d < head_dim; d += local_size) {
        float acc = 0.0f;
        if (denom > 0.0f) {
            for (int kt = min_key; kt <= max_key; ++kt) {
                const float p = scores[kt];
                if (p != 0.0f) {
                    acc += p * v[(b * key_stride + kt) * kv_channels + kv_head_offset + d];
                }
            }
            acc /= denom;
        }
        out[out_base + d] = acc;
    }
}

__kernel void grouped_query_attention_backward_f32(__global const float* q,
                                                   __global const float* k,
                                                   __global const float* v,
                                                   __global const float* grad_out,
                                                   __global float* grad_q,
                                                   __global float* grad_k,
                                                   __global float* grad_v,
                                                   int batch,
                                                   int query_tokens,
                                                   int key_tokens,
                                                   int n_head,
                                                   int n_kv_head,
                                                   int head_dim,
                                                   int causal,
                                                   int query_offset,
                                                   float scale) {
    int gid = get_global_id(0);
    int q_channels = n_head * head_dim;
    int kv_channels = n_kv_head * head_dim;
    int q_total = batch * query_tokens * q_channels;
    int kv_total = batch * key_tokens * kv_channels;
    if (gid >= q_total + 2 * kv_total) return;
    int group = n_head / n_kv_head;

    if (gid < q_total) {
        int d = gid % head_dim;
        int c = gid % q_channels;
        int query_token = (gid / q_channels) % query_tokens;
        int b = gid / (query_tokens * q_channels);
        int q_head = c / head_dim;
        int kv_head = q_head / group;
        int q_base = (b * query_tokens + query_token) * q_channels + q_head * head_dim;
        int kv_head_offset = kv_head * head_dim;

        float max_score = -3.402823466e+38F;
        for (int kt = 0; kt < key_tokens; ++kt) {
            if (causal && kt > query_offset + query_token) continue;
            int k_base = (b * key_tokens + kt) * kv_channels + kv_head_offset;
            float score = 0.0f;
            for (int i = 0; i < head_dim; ++i) score += q[q_base + i] * k[k_base + i];
            max_score = fmax(max_score, score * scale);
        }
        float denom = 0.0f;
        float sum_p_dp = 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) {
            if (causal && kt > query_offset + query_token) continue;
            int k_base = (b * key_tokens + kt) * kv_channels + kv_head_offset;
            float score = 0.0f;
            float dp = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                score += q[q_base + i] * k[k_base + i];
                dp += grad_out[q_base + i] * v[k_base + i];
            }
            float e = exp(score * scale - max_score);
            denom += e;
            sum_p_dp += e * dp;
        }
        sum_p_dp /= denom;
        float acc = 0.0f;
        for (int kt = 0; kt < key_tokens; ++kt) {
            if (causal && kt > query_offset + query_token) continue;
            int k_base = (b * key_tokens + kt) * kv_channels + kv_head_offset;
            float score = 0.0f;
            float dp = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                score += q[q_base + i] * k[k_base + i];
                dp += grad_out[q_base + i] * v[k_base + i];
            }
            float prob = exp(score * scale - max_score) / denom;
            float ds = prob * (dp - sum_p_dp) * scale;
            acc += ds * k[k_base + d];
        }
        grad_q[gid] = acc;
        return;
    }

    int local_idx = gid - q_total;
    int part = local_idx / kv_total;
    int elem = local_idx - part * kv_total;
    int d = elem % head_dim;
    int c = elem % kv_channels;
    int key_token = (elem / kv_channels) % key_tokens;
    int b = elem / (key_tokens * kv_channels);
    int kv_head = c / head_dim;
    float acc = 0.0f;
    for (int q_head = kv_head * group; q_head < (kv_head + 1) * group; ++q_head) {
        int q_head_offset = q_head * head_dim;
        for (int query_token = 0; query_token < query_tokens; ++query_token) {
            if (causal && key_token > query_offset + query_token) continue;
            int q_base = (b * query_tokens + query_token) * q_channels + q_head_offset;
            float max_score = -3.402823466e+38F;
            for (int kt = 0; kt < key_tokens; ++kt) {
                if (causal && kt > query_offset + query_token) continue;
                int k_base = (b * key_tokens + kt) * kv_channels + kv_head * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; ++i) score += q[q_base + i] * k[k_base + i];
                max_score = fmax(max_score, score * scale);
            }
            float denom = 0.0f;
            float sum_p_dp = 0.0f;
            float key_score = 0.0f;
            float key_dp = 0.0f;
            for (int kt = 0; kt < key_tokens; ++kt) {
                if (causal && kt > query_offset + query_token) continue;
                int k_base = (b * key_tokens + kt) * kv_channels + kv_head * head_dim;
                float score = 0.0f;
                float dp = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    score += q[q_base + i] * k[k_base + i];
                    dp += grad_out[q_base + i] * v[k_base + i];
                }
                float scaled = score * scale;
                float e = exp(scaled - max_score);
                denom += e;
                sum_p_dp += e * dp;
                if (kt == key_token) {
                    key_score = scaled;
                    key_dp = dp;
                }
            }
            float prob = exp(key_score - max_score) / denom;
            if (part == 0) {
                float ds = prob * (key_dp - sum_p_dp / denom) * scale;
                acc += ds * q[q_base + d];
            } else {
                acc += prob * grad_out[q_base + d];
            }
        }
    }
    if (part == 0) grad_k[elem] = acc;
    else grad_v[elem] = acc;
}

inline int mcl_attention_mask_index(int layout,
                                    int b,
                                    int query_token,
                                    int key_token,
                                    int query_tokens,
                                    int key_tokens) {
    if (layout == 1) return query_token * key_tokens + key_token;                  // [Q,K]
    if (layout == 2) return b * key_tokens + key_token;                            // [B,K]
    if (layout == 3) return (b * query_tokens + query_token) * key_tokens + key_token; // [B,Q,K] or [B*Q,K]
    if (layout == 4) return b * key_tokens + key_token;                            // [B,1,K]
    if (layout == 5) return query_token * key_tokens + key_token;                  // [1,Q,K]
    return query_token * key_tokens + key_token;
}

#define MCL_MASK_F32(mask, idx) ((mask)[idx] != 0.0f)
#define MCL_MASK_I32(mask, idx) ((mask)[idx] != 0)
#define MCL_MASK_U8(mask, idx) ((mask)[idx] != (uchar)0)
#define MCL_MASK_BIAS_F32(mask, idx) ((mask)[idx])
#define MCL_MASK_BIAS_ZERO(mask, idx) (0.0f)

#define DEFINE_GQA_BACKWARD_PROBS_MASKED(NAME, MASK_T, MASKED, MASK_BIAS, SUPPORTS_ADDITIVE) \
__kernel void NAME(__global const float* q, \
                   __global const float* k, \
                   __global const MASK_T* mask, \
                   __global float* probs, \
                   int batch, \
                   int query_tokens, \
                   int key_tokens, \
                   int n_head, \
                   int n_kv_head, \
                   int head_dim, \
                   int causal, \
                   int query_offset, \
                   int mask_layout, \
                   int mask_mode, \
                   float scale) { \
    int lid = get_local_id(0); \
    int row = get_group_id(1); \
    int rows = batch * n_head * query_tokens; \
    if (row >= rows || key_tokens > FA_STAGED_MAX_TOKENS || head_dim > FA_MAX_HEAD_DIM) return; \
    int query_token = row % query_tokens; \
    int bh = row / query_tokens; \
    int head = bh % n_head; \
    int b = bh / n_head; \
    int group = n_head / n_kv_head; \
    int kv_head = head / group; \
    int q_channels = n_head * head_dim; \
    int kv_channels = n_kv_head * head_dim; \
    int base_q = (b * query_tokens + query_token) * q_channels + head * head_dim; \
    int row_base = row * key_tokens; \
    __local float scores[FA_STAGED_MAX_TOKENS]; \
    if (lid < key_tokens) { \
        int key_token = lid; \
        int midx = mcl_attention_mask_index(mask_layout, b, query_token, key_token, query_tokens, key_tokens); \
        if ((causal && key_token > query_offset + query_token) || (mask_mode == 0 && MASKED(mask, midx))) { \
            scores[lid] = -3.402823466e+38F; \
        } else { \
            int base_k = (b * key_tokens + key_token) * kv_channels + kv_head * head_dim; \
            float score = 0.0f; \
            for (int d = 0; d < head_dim; ++d) score += q[base_q + d] * k[base_k + d]; \
            float bias = (mask_mode != 0 && SUPPORTS_ADDITIVE) ? MASK_BIAS(mask, midx) : 0.0f; \
            scores[lid] = score * scale + bias; \
        } \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    if (lid == 0) { \
        float max_score = -3.402823466e+38F; \
        for (int kt = 0; kt < key_tokens; ++kt) max_score = fmax(max_score, scores[kt]); \
        float denom = 0.0f; \
        for (int kt = 0; kt < key_tokens; ++kt) { \
            float value = (scores[kt] == -3.402823466e+38F) ? 0.0f : exp(scores[kt] - max_score); \
            probs[row_base + kt] = value; \
            denom += value; \
        } \
        float inv = denom > 0.0f ? 1.0f / denom : 0.0f; \
        for (int kt = 0; kt < key_tokens; ++kt) probs[row_base + kt] *= inv; \
    } \
}

#define DEFINE_GROUPED_QUERY_ATTENTION_MASKED_FWD(NAME, MASK_T, MASKED, MASK_BIAS, SUPPORTS_ADDITIVE) \
__kernel void NAME(__global const float* q, \
                   __global const float* k, \
                   __global const float* v, \
                   __global const MASK_T* mask, \
                   __global float* out, \
                   int batch, \
                   int query_tokens, \
                   int key_tokens, \
                   int n_head, \
                   int n_kv_head, \
                   int head_dim, \
                   int causal, \
                   int query_offset, \
                   int mask_layout, \
                   int mask_mode, \
                   int key_stride, \
                   float scale) { \
    int gid = get_global_id(0); \
    int q_channels = n_head * head_dim; \
    int kv_channels = n_kv_head * head_dim; \
    int total = batch * query_tokens * q_channels; \
    if (gid >= total) return; \
    int d = gid % head_dim; \
    int c = gid % q_channels; \
    int query_token = (gid / q_channels) % query_tokens; \
    int b = gid / (query_tokens * q_channels); \
    int q_head = c / head_dim; \
    int group = n_head / n_kv_head; \
    int kv_head = q_head / group; \
    int q_base = (b * query_tokens + query_token) * q_channels + q_head * head_dim; \
    int kv_head_offset = kv_head * head_dim; \
    float max_score = -3.402823466e+38F; \
    int valid = 0; \
    for (int kt = 0; kt < key_tokens; ++kt) { \
        if (causal && kt > query_offset + query_token) continue; \
        int midx = mcl_attention_mask_index(mask_layout, b, query_token, kt, query_tokens, key_tokens); \
        if (mask_mode == 0 && MASKED(mask, midx)) continue; \
        int k_base = (b * key_stride + kt) * kv_channels + kv_head_offset; \
        float score = 0.0f; \
        for (int i = 0; i < head_dim; ++i) score += q[q_base + i] * k[k_base + i]; \
        float bias = (mask_mode != 0 && SUPPORTS_ADDITIVE) ? MASK_BIAS(mask, midx) : 0.0f; \
        max_score = fmax(max_score, score * scale + bias); \
        ++valid; \
    } \
    if (valid == 0) { out[gid] = 0.0f; return; } \
    float denom = 0.0f; \
    float acc = 0.0f; \
    for (int kt = 0; kt < key_tokens; ++kt) { \
        if (causal && kt > query_offset + query_token) continue; \
        int midx = mcl_attention_mask_index(mask_layout, b, query_token, kt, query_tokens, key_tokens); \
        if (mask_mode == 0 && MASKED(mask, midx)) continue; \
        int k_base = (b * key_stride + kt) * kv_channels + kv_head_offset; \
        float score = 0.0f; \
        for (int i = 0; i < head_dim; ++i) score += q[q_base + i] * k[k_base + i]; \
        float bias = (mask_mode != 0 && SUPPORTS_ADDITIVE) ? MASK_BIAS(mask, midx) : 0.0f; \
        float prob = exp(score * scale + bias - max_score); \
        denom += prob; \
        acc += prob * v[(b * key_stride + kt) * kv_channels + kv_head_offset + d]; \
    } \
    out[gid] = denom > 0.0f ? acc / denom : 0.0f; \
}

#define DEFINE_GROUPED_QUERY_ATTENTION_MASKED_BWD(NAME, MASK_T, MASKED, MASK_BIAS, SUPPORTS_ADDITIVE) \
__kernel void NAME(__global const float* q, \
                   __global const float* k, \
                   __global const float* v, \
                   __global const float* grad_out, \
                   __global const MASK_T* mask, \
                   __global float* grad_q, \
                   __global float* grad_k, \
                   __global float* grad_v, \
                   int batch, \
                   int query_tokens, \
                   int key_tokens, \
                   int n_head, \
                   int n_kv_head, \
                   int head_dim, \
                   int causal, \
                   int query_offset, \
                   int mask_layout, \
                   int mask_mode, \
                   float scale) { \
    int gid = get_global_id(0); \
    int q_channels = n_head * head_dim; \
    int kv_channels = n_kv_head * head_dim; \
    int q_total = batch * query_tokens * q_channels; \
    int kv_total = batch * key_tokens * kv_channels; \
    if (gid >= q_total + 2 * kv_total) return; \
    int group = n_head / n_kv_head; \
    if (gid < q_total) { \
        int d = gid % head_dim; \
        int c = gid % q_channels; \
        int query_token = (gid / q_channels) % query_tokens; \
        int b = gid / (query_tokens * q_channels); \
        int q_head = c / head_dim; \
        int kv_head = q_head / group; \
        int q_base = (b * query_tokens + query_token) * q_channels + q_head * head_dim; \
        int kv_head_offset = kv_head * head_dim; \
        float max_score = -3.402823466e+38F; \
        int valid = 0; \
        for (int kt = 0; kt < key_tokens; ++kt) { \
            if (causal && kt > query_offset + query_token) continue; \
            int midx = mcl_attention_mask_index(mask_layout, b, query_token, kt, query_tokens, key_tokens); \
            if (mask_mode == 0 && MASKED(mask, midx)) continue; \
            int k_base = (b * key_tokens + kt) * kv_channels + kv_head_offset; \
            float score = 0.0f; \
            for (int i = 0; i < head_dim; ++i) score += q[q_base + i] * k[k_base + i]; \
            float bias = (mask_mode != 0 && SUPPORTS_ADDITIVE) ? MASK_BIAS(mask, midx) : 0.0f; \
            max_score = fmax(max_score, score * scale + bias); \
            ++valid; \
        } \
        if (valid == 0) { grad_q[gid] = 0.0f; return; } \
        float denom = 0.0f; \
        float sum_p_dp = 0.0f; \
        for (int kt = 0; kt < key_tokens; ++kt) { \
            if (causal && kt > query_offset + query_token) continue; \
            int midx = mcl_attention_mask_index(mask_layout, b, query_token, kt, query_tokens, key_tokens); \
            if (mask_mode == 0 && MASKED(mask, midx)) continue; \
            int k_base = (b * key_tokens + kt) * kv_channels + kv_head_offset; \
            float score = 0.0f; \
            float dp = 0.0f; \
            for (int i = 0; i < head_dim; ++i) { \
                score += q[q_base + i] * k[k_base + i]; \
                dp += grad_out[q_base + i] * v[k_base + i]; \
            } \
            float bias = (mask_mode != 0 && SUPPORTS_ADDITIVE) ? MASK_BIAS(mask, midx) : 0.0f; \
            float e = exp(score * scale + bias - max_score); \
            denom += e; \
            sum_p_dp += e * dp; \
        } \
        if (denom <= 0.0f) { grad_q[gid] = 0.0f; return; } \
        sum_p_dp /= denom; \
        float acc = 0.0f; \
        for (int kt = 0; kt < key_tokens; ++kt) { \
            if (causal && kt > query_offset + query_token) continue; \
            int midx = mcl_attention_mask_index(mask_layout, b, query_token, kt, query_tokens, key_tokens); \
            if (mask_mode == 0 && MASKED(mask, midx)) continue; \
            int k_base = (b * key_tokens + kt) * kv_channels + kv_head_offset; \
            float score = 0.0f; \
            float dp = 0.0f; \
            for (int i = 0; i < head_dim; ++i) { \
                score += q[q_base + i] * k[k_base + i]; \
                dp += grad_out[q_base + i] * v[k_base + i]; \
            } \
            float bias = (mask_mode != 0 && SUPPORTS_ADDITIVE) ? MASK_BIAS(mask, midx) : 0.0f; \
            float prob = exp(score * scale + bias - max_score) / denom; \
            float ds = prob * (dp - sum_p_dp) * scale; \
            acc += ds * k[k_base + d]; \
        } \
        grad_q[gid] = acc; \
        return; \
    } \
    int local_idx = gid - q_total; \
    int part = local_idx / kv_total; \
    int elem = local_idx - part * kv_total; \
    int d = elem % head_dim; \
    int c = elem % kv_channels; \
    int key_token = (elem / kv_channels) % key_tokens; \
    int b = elem / (key_tokens * kv_channels); \
    int kv_head = c / head_dim; \
    float acc = 0.0f; \
    for (int q_head = kv_head * group; q_head < (kv_head + 1) * group; ++q_head) { \
        int q_head_offset = q_head * head_dim; \
        for (int query_token = 0; query_token < query_tokens; ++query_token) { \
            if (causal && key_token > query_offset + query_token) continue; \
            int key_midx = mcl_attention_mask_index(mask_layout, b, query_token, key_token, query_tokens, key_tokens); \
            if (mask_mode == 0 && MASKED(mask, key_midx)) continue; \
            int q_base = (b * query_tokens + query_token) * q_channels + q_head_offset; \
            float max_score = -3.402823466e+38F; \
            int valid = 0; \
            for (int kt = 0; kt < key_tokens; ++kt) { \
                if (causal && kt > query_offset + query_token) continue; \
                int midx = mcl_attention_mask_index(mask_layout, b, query_token, kt, query_tokens, key_tokens); \
                if (mask_mode == 0 && MASKED(mask, midx)) continue; \
                int k_base = (b * key_tokens + kt) * kv_channels + kv_head * head_dim; \
                float score = 0.0f; \
                for (int i = 0; i < head_dim; ++i) score += q[q_base + i] * k[k_base + i]; \
                float bias = (mask_mode != 0 && SUPPORTS_ADDITIVE) ? MASK_BIAS(mask, midx) : 0.0f; \
                max_score = fmax(max_score, score * scale + bias); \
                ++valid; \
            } \
            if (valid == 0) continue; \
            float denom = 0.0f; \
            float sum_p_dp = 0.0f; \
            float key_score = 0.0f; \
            float key_dp = 0.0f; \
            float key_bias = 0.0f; \
            for (int kt = 0; kt < key_tokens; ++kt) { \
                if (causal && kt > query_offset + query_token) continue; \
                int midx = mcl_attention_mask_index(mask_layout, b, query_token, kt, query_tokens, key_tokens); \
                if (mask_mode == 0 && MASKED(mask, midx)) continue; \
                int k_base = (b * key_tokens + kt) * kv_channels + kv_head * head_dim; \
                float score = 0.0f; \
                float dp = 0.0f; \
                for (int i = 0; i < head_dim; ++i) { \
                    score += q[q_base + i] * k[k_base + i]; \
                    dp += grad_out[q_base + i] * v[k_base + i]; \
                } \
                float bias = (mask_mode != 0 && SUPPORTS_ADDITIVE) ? MASK_BIAS(mask, midx) : 0.0f; \
                float scaled = score * scale + bias; \
                float e = exp(scaled - max_score); \
                denom += e; \
                sum_p_dp += e * dp; \
                if (kt == key_token) { \
                    key_score = scaled; \
                    key_dp = dp; \
                    key_bias = bias; \
                } \
            } \
            if (denom <= 0.0f) continue; \
            float prob = exp(key_score - max_score) / denom; \
            if (part == 0) { \
                (void)key_bias; \
                float ds = prob * (key_dp - sum_p_dp / denom) * scale; \
                acc += ds * q[q_base + d]; \
            } else { \
                acc += prob * grad_out[q_base + d]; \
            } \
        } \
    } \
    if (part == 0) grad_k[elem] = acc; \
    else grad_v[elem] = acc; \
}

DEFINE_GROUPED_QUERY_ATTENTION_MASKED_FWD(grouped_query_attention_mask_f32, float, MCL_MASK_F32, MCL_MASK_BIAS_F32, 1)
DEFINE_GROUPED_QUERY_ATTENTION_MASKED_FWD(grouped_query_attention_mask_i32, int, MCL_MASK_I32, MCL_MASK_BIAS_ZERO, 0)
DEFINE_GROUPED_QUERY_ATTENTION_MASKED_FWD(grouped_query_attention_mask_u8, uchar, MCL_MASK_U8, MCL_MASK_BIAS_ZERO, 0)

DEFINE_GQA_BACKWARD_PROBS_MASKED(gqa_backward_probs_mask_f32, float, MCL_MASK_F32, MCL_MASK_BIAS_F32, 1)
DEFINE_GQA_BACKWARD_PROBS_MASKED(gqa_backward_probs_mask_i32, int, MCL_MASK_I32, MCL_MASK_BIAS_ZERO, 0)
DEFINE_GQA_BACKWARD_PROBS_MASKED(gqa_backward_probs_mask_u8, uchar, MCL_MASK_U8, MCL_MASK_BIAS_ZERO, 0)

DEFINE_GROUPED_QUERY_ATTENTION_MASKED_BWD(grouped_query_attention_backward_mask_f32, float, MCL_MASK_F32, MCL_MASK_BIAS_F32, 1)
DEFINE_GROUPED_QUERY_ATTENTION_MASKED_BWD(grouped_query_attention_backward_mask_i32, int, MCL_MASK_I32, MCL_MASK_BIAS_ZERO, 0)
DEFINE_GROUPED_QUERY_ATTENTION_MASKED_BWD(grouped_query_attention_backward_mask_u8, uchar, MCL_MASK_U8, MCL_MASK_BIAS_ZERO, 0)
