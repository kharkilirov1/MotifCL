
layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 0) readonly buffer QBuf { vec4 q4[]; };
layout(std430, set = 0, binding = 1) readonly buffer DoutBuf { vec4 dout4[]; };
layout(std430, set = 0, binding = 2) readonly buffer PBuf { float p[]; };
layout(std430, set = 0, binding = 3) readonly buffer DsBuf { float ds[]; };
layout(std430, set = 0, binding = 4) writeonly buffer DkBuf { vec4 dk4[]; };
layout(std430, set = 0, binding = 5) writeonly buffer DvBuf { vec4 dv4[]; };

layout(push_constant) uniform Push {
    uint batch;
    uint query_tokens;
    uint key_tokens;
    uint n_head;
    uint n_kv_head;
    uint head_dim;
    float scale;
} pc;

#ifndef DMAX
#error "the staged dK/dV body needs DMAX = head_dim / 4"
#endif
#if (CHUNK * DMAX) % 64 != 0
#error "one staged chunk must divide evenly across the 64 lanes"
#endif
#define VECS_PER_CHUNK (CHUNK * DMAX)
#define LOADS_PER_LANE (VECS_PER_CHUNK / 64)

shared vec4 tile[2][CHUNK][DMAX];

// Stage rows [r0, r0+CHUNK) of one head slice into buffer `buf` from the q buffer
// (`which` == 0) or the dout buffer (`which` == 1). Every lane copies four vec4,
// which keeps the loads coalesced across the workgroup; rows past T are zero-filled
// and skipped by the reduction.
void stage_chunk(int buf, uint r0, int which, uint b, uint q_col, uint q_channels, uint T) {
    const uint lane = gl_LocalInvocationID.x;
    for (uint l = 0u; l < uint(LOADS_PER_LANE); ++l) {
        const uint idx = lane + l * 64u;
        const uint j = idx / uint(DMAX);
        const uint i = idx - j * uint(DMAX);
        const uint row = r0 + j;
        tile[buf][j][i] = vec4(0.0);
        if (row < T) {
            const uint at = (((b * T + row) * q_channels + q_col) >> 2) + i;
            tile[buf][j][i] = which == 0 ? q4[at] : dout4[at];
        }
    }
}

void main() {
    const uint lane = gl_LocalInvocationID.x;
    const uint key = gl_WorkGroupID.x * 64u + lane;
    const uint kv_head = gl_WorkGroupID.y;
    const uint b = gl_WorkGroupID.z;
    const uint T = pc.query_tokens;
    if (b >= pc.batch || kv_head >= pc.n_kv_head || key >= pc.key_tokens) return;

    const uint group_size = pc.n_head / pc.n_kv_head;
    const uint q_channels = pc.n_head * pc.head_dim;
    const uint kv_channels = pc.n_kv_head * pc.head_dim;
    const uint head0 = kv_head * group_size;
    const uint out_base = kv_head * pc.head_dim;
    const uint dk_base = ((b * T + key) * kv_channels + out_base) >> 2;
    const uint chunk_of = (T + uint(CHUNK) - 1u) / uint(CHUNK);

    vec4 acc[DMAX];
    for (int i = 0; i < DMAX; ++i) acc[i] = vec4(0.0);

    // ---- dK: ds * q ------------------------------------------------------
    for (uint gi = 0u; gi < group_size; ++gi) {
        const uint hh = head0 + gi;
        const uint head_base = (b * pc.n_head + hh) * T * pc.key_tokens;
        const uint q_col = hh * pc.head_dim;
        stage_chunk(0, 0u, 0, b, q_col, q_channels, T);
        barrier();
        for (uint c = 0u; c < chunk_of; ++c) {
            const uint cur = c & 1u;
            const uint next = 1u - cur;
            if (c + 1u < chunk_of) stage_chunk(int(next), (c + 1u) * uint(CHUNK), 0, b, q_col, q_channels, T);
            const uint r0 = c * uint(CHUNK);
            for (uint j = 0u; j < uint(CHUNK); ++j) {
                const uint row = r0 + j;
                if (row >= T) break;
                const float dsv = ds[head_base + row * pc.key_tokens + key];
                for (int i = 0; i < DMAX; ++i) acc[i] += dsv * tile[cur][j][i];
            }
            barrier();
        }
    }
    for (int i = 0; i < DMAX; ++i) dk4[dk_base + uint(i)] = acc[i] * pc.scale;

    // ---- dV: p * dout ----------------------------------------------------
    for (int i = 0; i < DMAX; ++i) acc[i] = vec4(0.0);
    for (uint gi = 0u; gi < group_size; ++gi) {
        const uint hh = head0 + gi;
        const uint head_base = (b * pc.n_head + hh) * T * pc.key_tokens;
        const uint q_col = hh * pc.head_dim;
        stage_chunk(0, 0u, 1, b, q_col, q_channels, T);
        barrier();
        for (uint c = 0u; c < chunk_of; ++c) {
            const uint cur = c & 1u;
            const uint next = 1u - cur;
            if (c + 1u < chunk_of) stage_chunk(int(next), (c + 1u) * uint(CHUNK), 1, b, q_col, q_channels, T);
            const uint r0 = c * uint(CHUNK);
            for (uint j = 0u; j < uint(CHUNK); ++j) {
                const uint row = r0 + j;
                if (row >= T) break;
                const float pv = p[head_base + row * pc.key_tokens + key];
                for (int i = 0; i < DMAX; ++i) acc[i] += pv * tile[cur][j][i];
            }
            barrier();
        }
    }
    for (int i = 0; i < DMAX; ++i) dv4[dk_base + uint(i)] = acc[i];
}
