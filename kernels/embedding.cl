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

inline ushort embedding_load_u16_le(__global const uchar* p, int offset) {
    return (ushort)(((ushort)p[offset]) | ((ushort)p[offset + 1] << 8));
}

inline float embedding_f16_to_f32(ushort h) {
    uint sign = ((uint)h & 0x8000u) << 16;
    uint exp = ((uint)h >> 10) & 31u;
    uint mant = (uint)h & 1023u;
    if (exp == 0u) {
        if (mant == 0u) return as_float(sign);
        while ((mant & 1024u) == 0u) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 1023u;
    } else if (exp == 31u) {
        return as_float(sign | 0x7f800000u | (mant << 13));
    }
    uint bits = sign | ((exp + 112u) << 23) | (mant << 13);
    return as_float(bits);
}

inline void embedding_k_scale_min(int j, __global const uchar* q, int* d, int* m) {
    if (j < 4) {
        *d = ((int)q[j]) & 63;
        *m = ((int)q[j + 4]) & 63;
    } else {
        *d = (((int)q[j + 4]) & 15) | ((((int)q[j - 4]) >> 6) << 4);
        *m = (((int)q[j + 4]) >> 4) | ((((int)q[j]) >> 6) << 4);
    }
}

inline float embedding_q4_k_value(__global const uchar* weight, int idx) {
    int block = idx >> 8;
    int loc = idx & 255;
    int sub = loc >> 5;
    int l = loc & 31;
    int group = sub >> 1;
    int base = block * 144;
    __global const uchar* scales = weight + base + 4;
    __global const uchar* qs = weight + base + 16 + group * 32;
    int sc = 0;
    int mn = 0;
    embedding_k_scale_min(sub, scales, &sc, &mn);
    int code = (sub & 1) ? (((int)qs[l]) >> 4) : (((int)qs[l]) & 15);
    float d = embedding_f16_to_f32(embedding_load_u16_le(weight, base));
    float dmin = embedding_f16_to_f32(embedding_load_u16_le(weight, base + 2));
    return d * ((float)sc) * ((float)code) - dmin * ((float)mn);
}

inline float embedding_q5_k_value(__global const uchar* weight, int idx) {
    int block = idx >> 8;
    int loc = idx & 255;
    int sub = loc >> 5;
    int l = loc & 31;
    int group = sub >> 1;
    int base = block * 176;
    __global const uchar* scales = weight + base + 4;
    __global const uchar* qh = weight + base + 16;
    __global const uchar* ql = weight + base + 48 + group * 32;
    int sc = 0;
    int mn = 0;
    embedding_k_scale_min(sub, scales, &sc, &mn);
    int high_bit = 1 << sub;
    int code = (sub & 1) ? (((int)ql[l]) >> 4) : (((int)ql[l]) & 15);
    if ((((int)qh[l]) & high_bit) != 0) code += 16;
    float d = embedding_f16_to_f32(embedding_load_u16_le(weight, base));
    float dmin = embedding_f16_to_f32(embedding_load_u16_le(weight, base + 2));
    return d * ((float)sc) * ((float)code) - dmin * ((float)mn);
}

__kernel void embedding_gather_transposed_q4_k_i32(__global const uchar* weight,
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
    // GGUF tensors store dimensions in ggml order: dim[0] is the contiguous
    // axis.  token_embd.weight has dims [embed_dim, vocab], so one token row is
    // laid out contiguously as token * embed_dim + d.
    out[gid] = (idx >= 0 && idx < vocab_size) ? embedding_q4_k_value(weight, idx * embed_dim + d) : 0.0f;
}

__kernel void embedding_gather_transposed_q5_k_i32(__global const uchar* weight,
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
    // GGUF [embed_dim, vocab] memory order is token-major for embedding
    // gathers, not d-major.
    out[gid] = (idx >= 0 && idx < vocab_size) ? embedding_q5_k_value(weight, idx * embed_dim + d) : 0.0f;
}

__kernel void embedding_per_layer_slice_f32(__global const float* packed,
                                            __global float* out,
                                            int token_count,
                                            int layer_count,
                                            int ple_dim,
                                            int layer,
                                            int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    int token = gid / ple_dim;
    int d = gid - token * ple_dim;
    out[gid] = packed[(token * layer_count + layer) * ple_dim + d];
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
