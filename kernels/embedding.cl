__kernel void embedding_gather_f32_i32(__global const float* weight,
                                       __global const int* indices,
                                       __global float* out,
                                       int vocab_size,
                                       int embed_dim,
                                       int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    int token = gid / embed_dim;
    int d = gid - token * embed_dim;
    int idx = indices[token];
    if (idx < 0 || idx >= vocab_size) {
        out[gid] = 0.0f;
    } else {
        out[gid] = weight[idx * embed_dim + d];
    }
}

__kernel void token_position_embedding_f32_i32(__global const float* token_weight,
                                               __global const float* pos_weight,
                                               __global const int* token_ids,
                                               __global float* out,
                                               int vocab_size,
                                               int seq_len,
                                               int embed_dim,
                                               int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    int d = gid % embed_dim;
    int token_linear = gid / embed_dim;
    int pos = token_linear % seq_len;
    int token_id = token_ids[token_linear];
    float tok = 0.0f;
    if (token_id >= 0 && token_id < vocab_size) tok = token_weight[token_id * embed_dim + d];
    out[gid] = tok + pos_weight[pos * embed_dim + d];
}

__kernel void embedding_weight_backward_f32_i32(__global const int* indices,
                                                __global const float* grad_out,
                                                __global float* grad_weight,
                                                int vocab_size,
                                                int embed_dim,
                                                int token_count,
                                                int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    int vocab = gid / embed_dim;
    int d = gid - vocab * embed_dim;
    if (vocab >= vocab_size) return;

    float acc = 0.0f;
    for (int t = 0; t < token_count; ++t) {
        if (indices[t] == vocab) {
            acc += grad_out[t * embed_dim + d];
        }
    }
    grad_weight[gid] = acc;
}

__kernel void position_embedding_backward_f32_i32(__global const float* grad_out,
                                                  __global float* grad_position,
                                                  int batch,
                                                  int seq_len,
                                                  int embed_dim,
                                                  int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    int pos = gid / embed_dim;
    int d = gid - pos * embed_dim;
    if (pos >= seq_len) return;

    float acc = 0.0f;
    for (int b = 0; b < batch; ++b) {
        acc += grad_out[(b * seq_len + pos) * embed_dim + d];
    }
    grad_position[gid] = acc;
}
