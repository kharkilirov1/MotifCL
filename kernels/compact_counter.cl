// Finite-state counter synapse kernels (research: memory-native training).
// Each weight is a finite-state code in [0,3*(2C-1)); 4 codes pack into 24 bits = 3 bytes.
// state is packed [out, in/4, 3] bytes (requires in % 4 == 0); grad_w/w are dense [out,in];
// scale/v are per-output-row [out]. The update is two kernels so the heavy per-element
// work parallelises across out*(in/4) groups instead of one work-item per row.

inline void cc_decode(uchar code, int C, int* t, int* c) {
    int lv = 2 * C - 1;
    int z = (int)code;
    *t = z / lv - 1;
    *c = z % lv - (C - 1);
}
inline uchar cc_encode(int t, int c, int C) {
    int lv = 2 * C - 1;
    return (uchar)((t + 1) * lv + (c + (C - 1)));
}
inline uint cc_hash_u32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16; return x;
}
inline float cc_uniform01(uint x) {
    return (float)(cc_hash_u32(x) & 0x00ffffffu) * (1.0f / 16777216.0f);
}
inline void cc_unpack4(__global const uchar* p, int* c0, int* c1, int* c2, int* c3) {
    int b0 = p[0], b1 = p[1], b2 = p[2];
    *c0 = b0 & 0x3F;
    *c1 = ((b0 >> 6) | (b1 << 2)) & 0x3F;
    *c2 = ((b1 >> 4) | (b2 << 4)) & 0x3F;
    *c3 = (b2 >> 2) & 0x3F;
}
inline void cc_pack4(__global uchar* p, int c0, int c1, int c2, int c3) {
    p[0] = (uchar)((c0 & 0x3F) | ((c1 & 0x03) << 6));
    p[1] = (uchar)(((c1 >> 2) & 0x0F) | ((c2 & 0x0F) << 4));
    p[2] = (uchar)(((c2 >> 4) & 0x03) | ((c3 & 0x3F) << 2));
}

// Decode packed state -> dense ternary weight w[out,in] = scale[row]*t. One wi per group.
__kernel void decode_counter_state_weight_f32(__global const uchar* state,
                                              __global const float* scale,
                                              __global float* w,
                                              int C, int in_features, int n_groups) {
    int gid = get_global_id(0);
    if (gid >= n_groups) return;
    int gpr = in_features / 4;
    int row = gid / gpr;
    int g = gid - row * gpr;
    int cc[4];
    cc_unpack4(state + gid * 3, &cc[0], &cc[1], &cc[2], &cc[3]);
    float s = scale[row];
    for (int k = 0; k < 4; ++k) {
        int t, c; cc_decode((uchar)cc[k], C, &t, &c);
        w[row * in_features + g * 4 + k] = s * (float)t;
    }
}

// Pass 1 (one work-GROUP per output row, parallel reduction): row-RMS + new scale + denom.
#define CC_RS_LSIZE 64
__kernel void counter_row_stats_f32(__global const uchar* state,
                                    __global const float* scale,
                                    __global float* v,
                                    __global const float* grad_w,
                                    __global float* scale_new,
                                    __global float* denom,
                                    int C, int in_features, int out_features,
                                    float lr_scale, float rms_beta, float rms_eps) {
    int row = get_group_id(0);
    if (row >= out_features) return;
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int gpr = in_features / 4;
    __local float lg[CC_RS_LSIZE];
    __local float lgs[CC_RS_LSIZE];
    float g_sq = 0.0f, grad_s = 0.0f;
    for (int g = lid; g < gpr; g += lsz) {
        int cc[4];
        cc_unpack4(state + (row * gpr + g) * 3, &cc[0], &cc[1], &cc[2], &cc[3]);
        for (int k = 0; k < 4; ++k) {
            float gw = grad_w[row * in_features + g * 4 + k];
            g_sq += gw * gw;
            int t, c; cc_decode((uchar)cc[k], C, &t, &c);
            grad_s += gw * (float)t;
        }
    }
    lg[lid] = g_sq;
    lgs[lid] = grad_s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) { lg[lid] += lg[lid + s]; lgs[lid] += lgs[lid + s]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        float gsq = lg[0] / (float)in_features;
        float gs = lgs[0] / sqrt((float)in_features);
        float vv = rms_beta * v[row] + (1.0f - rms_beta) * gsq;
        v[row] = vv;
        denom[row] = fmax(sqrt(vv), rms_eps);
        scale_new[row] = clamp(scale[row] - lr_scale * gs, 1e-5f, 10.0f);
    }
}

// Pass 2 (one wi per packed group of 4 weights): SR counter tick; read-modify-write the
// 3-byte group so the 4 codes never race across work-items.
__kernel void counter_apply_update_f32(__global uchar* state,
                                       __global const float* scale,
                                       __global const float* scale_new,
                                       __global const float* denom,
                                       __global const float* grad_w,
                                       int C, int in_features, int n_groups,
                                       float lr, uint seed) {
    int gid = get_global_id(0);
    if (gid >= n_groups) return;
    int gpr = in_features / 4;
    int row = gid / gpr;
    int g = gid - row * gpr;
    int cc[4];
    cc_unpack4(state + gid * 3, &cc[0], &cc[1], &cc[2], &cc[3]);
    float s_old = scale[row], s_new = scale_new[row], dn = denom[row], Cf = (float)C;
    int nc[4];
    for (int k = 0; k < 4; ++k) {
        int elem = row * in_features + g * 4 + k;
        int t, c; cc_decode((uchar)cc[k], C, &t, &c);
        float tick = -lr * (grad_w[elem] / dn) * (Cf / s_new);
        float c_reb = (float)c * (s_old / s_new);
        float val = c_reb + tick;
        float f = floor(val);
        float rnd = cc_uniform01(seed ^ cc_hash_u32((uint)elem));
        float ccv = f + (rnd < (val - f) ? 1.0f : 0.0f);
        float carry = trunc(ccv / Cf);
        float rem = ccv - carry * Cf;
        int nt = t + (int)carry;
        int ct = nt < -1 ? -1 : (nt > 1 ? 1 : nt);
        if (ct != nt) rem = (ccv > 0.0f ? 1.0f : (ccv < 0.0f ? -1.0f : 0.0f)) * (Cf - 1.0f);
        int ri = (int)rem;
        ri = ri < -(C - 1) ? -(C - 1) : (ri > C - 1 ? C - 1 : ri);
        nc[k] = (int)cc_encode(ct, ri, C);
    }
    cc_pack4(state + gid * 3, nc[0], nc[1], nc[2], nc[3]);
}

// ---------------------------------------------------------------------------
// Fused (memory-native) backward: the weight gradient grad_w[o,i] is NEVER
// materialised as a dense [out,in] buffer. Instead each work-item recomputes
// grad_w[o,i] = sum_r grad_out[r,o] * x[r,i] on the fly from the activation
// input x [N,in] and the upstream gradient grad_out [N,out]. Trades ~2x compute
// (the sum is recomputed in pass 1 and pass 2) for zero gradient memory.
// ---------------------------------------------------------------------------

// Pass 1 fused (one work-GROUP per output row): row-RMS + new scale + denom,
// computing grad_w on the fly. Mirrors counter_row_stats_f32 exactly except the
// gradient is recomputed instead of read from a grad_w buffer.
__kernel void counter_row_stats_fused_f32(__global const uchar* state,
                                          __global const float* scale,
                                          __global float* v,
                                          __global const float* grad_out,
                                          __global const float* x,
                                          __global float* scale_new,
                                          __global float* denom,
                                          int C, int in_features, int out_features, int N,
                                          float lr_scale, float rms_beta, float rms_eps) {
    int row = get_group_id(0);
    if (row >= out_features) return;
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int gpr = in_features / 4;
    __local float lg[CC_RS_LSIZE];
    __local float lgs[CC_RS_LSIZE];
    float g_sq = 0.0f, grad_s = 0.0f;
    for (int g = lid; g < gpr; g += lsz) {
        int cc[4];
        cc_unpack4(state + (row * gpr + g) * 3, &cc[0], &cc[1], &cc[2], &cc[3]);
        for (int k = 0; k < 4; ++k) {
            int i = g * 4 + k;
            float gw = 0.0f;                                   // grad_w[row,i] recomputed on the fly
            for (int r = 0; r < N; ++r) gw += grad_out[r * out_features + row] * x[r * in_features + i];
            g_sq += gw * gw;
            int t, c; cc_decode((uchar)cc[k], C, &t, &c);
            grad_s += gw * (float)t;
        }
    }
    lg[lid] = g_sq;
    lgs[lid] = grad_s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) { lg[lid] += lg[lid + s]; lgs[lid] += lgs[lid + s]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        float gsq = lg[0] / (float)in_features;
        float gs = lgs[0] / sqrt((float)in_features);
        float vv = rms_beta * v[row] + (1.0f - rms_beta) * gsq;
        v[row] = vv;
        denom[row] = fmax(sqrt(vv), rms_eps);
        scale_new[row] = clamp(scale[row] - lr_scale * gs, 1e-5f, 10.0f);
    }
}

// Pass 2 fused (one wi per packed group of 4 weights): SR counter tick with grad_w
// recomputed on the fly. Identical SR math/seed to counter_apply_update_f32.
__kernel void counter_apply_update_fused_f32(__global uchar* state,
                                             __global const float* scale,
                                             __global const float* scale_new,
                                             __global const float* denom,
                                             __global const float* grad_out,
                                             __global const float* x,
                                             int C, int in_features, int out_features, int N,
                                             int n_groups, float lr, uint seed) {
    int gid = get_global_id(0);
    if (gid >= n_groups) return;
    int gpr = in_features / 4;
    int row = gid / gpr;
    int g = gid - row * gpr;
    int cc[4];
    cc_unpack4(state + gid * 3, &cc[0], &cc[1], &cc[2], &cc[3]);
    float s_old = scale[row], s_new = scale_new[row], dn = denom[row], Cf = (float)C;
    int nc[4];
    for (int k = 0; k < 4; ++k) {
        int i = g * 4 + k;
        int elem = row * in_features + i;
        float gw = 0.0f;                                       // grad_w[row,i] recomputed on the fly
        for (int r = 0; r < N; ++r) gw += grad_out[r * out_features + row] * x[r * in_features + i];
        int t, c; cc_decode((uchar)cc[k], C, &t, &c);
        float tick = -lr * (gw / dn) * (Cf / s_new);
        float c_reb = (float)c * (s_old / s_new);
        float val = c_reb + tick;
        float f = floor(val);
        float rnd = cc_uniform01(seed ^ cc_hash_u32((uint)elem));
        float ccv = f + (rnd < (val - f) ? 1.0f : 0.0f);
        float carry = trunc(ccv / Cf);
        float rem = ccv - carry * Cf;
        int nt = t + (int)carry;
        int ct = nt < -1 ? -1 : (nt > 1 ? 1 : nt);
        if (ct != nt) rem = (ccv > 0.0f ? 1.0f : (ccv < 0.0f ? -1.0f : 0.0f)) * (Cf - 1.0f);
        int ri = (int)rem;
        ri = ri < -(C - 1) ? -(C - 1) : (ri > C - 1 ? C - 1 : ri);
        nc[k] = (int)cc_encode(ct, ri, C);
    }
    cc_pack4(state + gid * 3, nc[0], nc[1], nc[2], nc[3]);
}

// ---------------------------------------------------------------------------
// In-kernel grad_x (memory-native backprop-to-input):
//   grad_x[r,i] = sum_o grad_out[r,o] * scale[o] * t[o,i]
// decoding the ternary weight t[o,i] from packed state on the fly. Avoids
// decoding a dense [out,in] weight just to compute grad_x -- only the grad_x
// output [N,in] is allocated, never the weight. One work-item per (r,i).
// ---------------------------------------------------------------------------
__kernel void counter_backward_input_f32(__global const float* grad_out,
                                         __global const uchar* state,
                                         __global const float* scale,
                                         __global float* grad_x,
                                         int C, int in_features, int out_features, int N) {
    int gid = get_global_id(0);
    if (gid >= N * in_features) return;
    int r = gid / in_features;
    int i = gid - r * in_features;
    int gpr = in_features / 4;
    int g = i >> 2;        // i / 4 (packed group)
    int k = i & 3;         // i % 4 (code within group)
    int lv = 2 * C - 1;
    float acc = 0.0f;
    for (int o = 0; o < out_features; ++o) {
        int cc[4];
        cc_unpack4(state + (o * gpr + g) * 3, &cc[0], &cc[1], &cc[2], &cc[3]);
        int t = cc[k] / lv - 1;                       // decode ternary t[o,i] (matches cc_decode)
        acc += grad_out[r * out_features + o] * scale[o] * (float)t;
    }
    grad_x[gid] = acc;
}
