#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable

layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 0) readonly buffer QBuf { float q[]; };
#ifdef KV_VEC4
// head_dim is a multiple of four here, and so is every K/V row start, so the row of
// one key is a run of head_dim/4 aligned vec4. One lane owns one key and reads its
// whole row: the four components are added in ascending d order, which keeps the
// summation order (and the result) identical to the scalar body.
layout(std430, set = 0, binding = 1) readonly buffer KBuf { vec4 k4[]; };
layout(std430, set = 0, binding = 2) readonly buffer VBuf { vec4 v4[]; };
#else
layout(std430, set = 0, binding = 1) readonly buffer KBuf { float k[]; };
layout(std430, set = 0, binding = 2) readonly buffer VBuf { float v[]; };
#endif
layout(std430, set = 0, binding = 3) readonly buffer DoutBuf { float dout[]; };
layout(std430, set = 0, binding = 4) writeonly buffer PBuf { float p_out[]; };
layout(std430, set = 0, binding = 5) writeonly buffer DsBuf { float ds_out[]; };
layout(std430, set = 0, binding = 6) writeonly buffer DqBuf { float dq[]; };

layout(push_constant) uniform Push {
    uint batch;
    uint query_tokens;
    uint key_tokens;
    uint n_head;
    uint n_kv_head;
    uint head_dim;
    float scale;
} p;

#define MAXK 256
#define MAXD 64

shared float qs[ROWS][MAXD];
shared float ds_in[ROWS][MAXD];
shared float probs[ROWS][MAXK];
shared float dps[ROWS][MAXK];

void main() {
    const uint lane = gl_LocalInvocationID.x;
    const uint r0 = gl_WorkGroupID.x * uint(ROWS);
    const uint h = gl_WorkGroupID.y;
    const uint b = gl_WorkGroupID.z;
    const uint T = p.query_tokens;
    if (h >= p.n_head || b >= p.batch || r0 >= T) return;

    const uint group_size = p.n_head / p.n_kv_head;
    const uint kv_head = h / group_size;
    const uint q_channels = p.n_head * p.head_dim;
    const uint kv_channels = p.n_kv_head * p.head_dim;
    const float neg_inf = -3.402823466e+38;
    const uint key_hi = min(p.key_tokens, r0 + uint(ROWS));

    // Stage the query and dout rows.
    for (uint j = 0u; j < uint(ROWS); ++j) {
        const uint tq = r0 + j;
        if (tq < T) {
            const uint qb = (b * T + tq) * q_channels + h * p.head_dim;
            for (uint d = lane; d < p.head_dim; d += 64u) {
                qs[j][d] = q[qb + d];
                ds_in[j][d] = dout[qb + d];
            }
        }
    }
    barrier();

    // Pass 1: causal-masked scores, per-row subgroup max.
    float lane_max[ROWS];
    for (uint j = 0u; j < uint(ROWS); ++j) lane_max[j] = neg_inf;
    for (uint tk = lane; tk < key_hi; tk += 64u) {
        const uint kb = (b * p.key_tokens + tk) * kv_channels + kv_head * p.head_dim;
        float dot[ROWS];
        for (uint j = 0u; j < uint(ROWS); ++j) dot[j] = 0.0;
#ifdef KV_VEC4
        const uint kbase = kb >> 2;
        for (uint i = 0u; i < (p.head_dim >> 2); ++i) {
            const vec4 kv = k4[kbase + i];
            const uint d0 = i * 4u;
            for (uint j = 0u; j < uint(ROWS); ++j) {
                float acc = dot[j];
                acc += qs[j][d0 + 0u] * kv.x;
                acc += qs[j][d0 + 1u] * kv.y;
                acc += qs[j][d0 + 2u] * kv.z;
                acc += qs[j][d0 + 3u] * kv.w;
                dot[j] = acc;
            }
        }
#else
        for (uint d = 0u; d < p.head_dim; ++d) {
            const float kd = k[kb + d];
            for (uint j = 0u; j < uint(ROWS); ++j) dot[j] += qs[j][d] * kd;
        }
#endif
        for (uint j = 0u; j < uint(ROWS); ++j) {
            float s = dot[j] * p.scale;
            if (tk > r0 + j) s = neg_inf;
            probs[j][tk] = s;
            lane_max[j] = max(lane_max[j], s);
        }
    }
    float row_max[ROWS];
    for (uint j = 0u; j < uint(ROWS); ++j) {
        row_max[j] = subgroupMax(lane_max[j]);
        if (row_max[j] == neg_inf) row_max[j] = 0.0;
    }

    // Pass 2: unnormalized softmax numerators, per-row subgroup sum.
    float lane_sum[ROWS];
    for (uint j = 0u; j < uint(ROWS); ++j) lane_sum[j] = 0.0;
    for (uint tk = lane; tk < key_hi; tk += 64u) {
        for (uint j = 0u; j < uint(ROWS); ++j) {
            const float e = probs[j][tk] == neg_inf ? 0.0 : exp(probs[j][tk] - row_max[j]);
            probs[j][tk] = e;
            lane_sum[j] += e;
        }
    }
    float inv[ROWS];
    for (uint j = 0u; j < uint(ROWS); ++j) {
        const float sum = subgroupAdd(lane_sum[j]);
        inv[j] = sum > 0.0 ? 1.0 / sum : 0.0;
    }

    // Pass 3: dp = dout . v per key, plus the sum(dp*p) row dot.
    float lane_dot[ROWS];
    for (uint j = 0u; j < uint(ROWS); ++j) lane_dot[j] = 0.0;
    for (uint tk = lane; tk < key_hi; tk += 64u) {
        const uint vb = (b * p.key_tokens + tk) * kv_channels + kv_head * p.head_dim;
        float dp[ROWS];
        for (uint j = 0u; j < uint(ROWS); ++j) dp[j] = 0.0;
#ifdef KV_VEC4
        const uint vbase = vb >> 2;
        for (uint i = 0u; i < (p.head_dim >> 2); ++i) {
            const vec4 vv = v4[vbase + i];
            const uint d0 = i * 4u;
            for (uint j = 0u; j < uint(ROWS); ++j) {
                float acc = dp[j];
                acc += ds_in[j][d0 + 0u] * vv.x;
                acc += ds_in[j][d0 + 1u] * vv.y;
                acc += ds_in[j][d0 + 2u] * vv.z;
                acc += ds_in[j][d0 + 3u] * vv.w;
                dp[j] = acc;
            }
        }
#else
        for (uint d = 0u; d < p.head_dim; ++d) {
            const float vd = v[vb + d];
            for (uint j = 0u; j < uint(ROWS); ++j) dp[j] += ds_in[j][d] * vd;
        }
#endif
        for (uint j = 0u; j < uint(ROWS); ++j) {
            dps[j][tk] = dp[j];
            lane_dot[j] += dp[j] * probs[j][tk] * inv[j];
        }
    }
    float row_dot[ROWS];
    for (uint j = 0u; j < uint(ROWS); ++j) row_dot[j] = subgroupAdd(lane_dot[j]);
    barrier();

    // Pass 4: emit normalized probabilities and ds; cache ds for the dQ pass.
    const uint row_base = (b * p.n_head + h) * T * p.key_tokens + r0 * p.key_tokens;
    for (uint tk = lane; tk < p.key_tokens; tk += 64u) {
        const uint base = row_base + tk;
        for (uint j = 0u; j < uint(ROWS); ++j) {
            float pv = 0.0;
            float dv = 0.0;
            if (tk < key_hi) {
                pv = probs[j][tk] * inv[j];
                dv = pv * (dps[j][tk] - row_dot[j]);
            }
            if (r0 + j < T) {
                p_out[base + j * p.key_tokens] = pv;
                ds_out[base + j * p.key_tokens] = dv;
            }
            dps[j][tk] = dv;
        }
    }
    barrier();

    // Pass 5: dQ with one lane per output dimension, reusing each K element
    // across all staged rows.
    if (lane < p.head_dim) {
        const uint d = lane;
        const uint kb = kv_head * p.head_dim + d;
        float acc[ROWS];
        for (uint j = 0u; j < uint(ROWS); ++j) acc[j] = 0.0;
#ifdef KV_VEC4
        // This pass is already coalesced across lanes (lane == d), so it keeps one
        // scalar read per key; the vec4 view selects the lane's component.
        const uint comp = d & 3u;
        for (uint tk = 0u; tk < p.key_tokens; ++tk) {
            const uint at = (b * p.key_tokens + tk) * kv_channels + kv_head * p.head_dim + (d & ~3u);
            const vec4 kv = k4[at >> 2];
            const float kd = comp == 0u ? kv.x : (comp == 1u ? kv.y : (comp == 2u ? kv.z : kv.w));
            for (uint j = 0u; j < uint(ROWS); ++j) acc[j] += dps[j][tk] * kd;
        }
#else
        for (uint tk = 0u; tk < p.key_tokens; ++tk) {
            const float kd = k[(b * p.key_tokens + tk) * kv_channels + kb];
            for (uint j = 0u; j < uint(ROWS); ++j) acc[j] += dps[j][tk] * kd;
        }
#endif
        const uint qb = h * p.head_dim + d;
        for (uint j = 0u; j < uint(ROWS); ++j) {
            if (r0 + j < T) dq[(b * T + r0 + j) * q_channels + qb] = acc[j] * p.scale;
        }
    }
}
