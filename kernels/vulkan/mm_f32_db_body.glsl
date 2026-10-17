// Double-buffered register-blocked FP32 GEMM body: C[M,N] = op(A) * op(B).
//
// This is the tunable body shared by the 32x32 (64-thread, one wave) and 64x64
// (256-thread, four waves) specializations. The wrapper .comp files define TILE,
// LANES, LX/LY and pull this file in.
//
// Design notes for Polaris/gfx803 (wave64, 32 KiB LDS, no FP16 math, no float
// atomics):
//   * 4x4 outputs per lane (16 FMA per k step) from two LDS.128 reads, so the
//     LDS:FFMA ratio stays at the FMA-bound boundary instead of the 8-scalar
//     read pattern that caps the older kernels at ~25% of peak.
//   * K is walked in 16-deep tiles held in a two-buffer LDS ring. The next tile
//     is loaded into registers *before* the current tile is consumed, so global
//     latency is hidden behind the FMA loop instead of stalling an entire
//     workgroup between tiles.
//   * A and B are viewed as vec4 arrays: every global access is one 128-bit
//     load, which is the only way to feed 2304 FP32 lanes from this memory
//     system. Callers therefore require M, K and N to be multiples of four;
//     other shapes use the scalar kernels.
//   * Three operand layouts share one body (the trainer issues all three):
//       a_layout 0: A[m][k] stride K      a_layout 1: A[k][m] stride M
//       b_layout 0: B[k][n] stride N      b_layout 1: B[n][k] stride K
//     which covers A*B (0,0), A*B^T (0,1) and A^T*B (1,0).
layout(std430, set = 0, binding = 0) readonly buffer ABuf { vec4 A4[]; };
layout(std430, set = 0, binding = 1) readonly buffer BBuf { vec4 B4[]; };
layout(std430, set = 0, binding = 2) writeonly buffer CBuf { vec4 C4[]; };

layout(push_constant) uniform Push {
    uint M;
    uint K;
    uint N;
    uint a_layout;
    uint b_layout;
} p;

#define KT 16
#define TQ (TILE / 4)
#define LQ (TILE / 4)
#define AQ (TILE * KT / 4 / LANES)
#define BQ (KT * TILE / 4 / LANES)

shared vec4 As[2][KT][LQ];
shared vec4 Bs[2][KT][LQ];

// ---- tile staging ---------------------------------------------------------

void load_a(uint lid, uint m0, uint kt, out vec4 r0, out vec4 r1) {
    if (p.a_layout == 0u) {
        // A[m][k]: one vec4 covers four consecutive k of a single row.
        for (int i = 0; i < AQ; ++i) {
            const uint idx = lid + uint(i) * LANES;
            const uint r = idx >> 2;
            const uint kq = idx & 3u;
            const uint gm = m0 + r;
            const uint gk = kt + kq * 4u;
            vec4 v = vec4(0.0);
            if (gm < p.M && gk < p.K) v = A4[(gm * p.K + gk) >> 2];
            if (i == 0) r0 = v; else r1 = v;
        }
    } else {
        // A[k][m]: one vec4 covers four consecutive m of a single k row.
        for (int i = 0; i < AQ; ++i) {
            const uint idx = lid + uint(i) * LANES;
            const uint kk = idx / TQ;
            const uint mq = idx - kk * TQ;
            const uint gm = m0 + mq * 4u;
            const uint gk = kt + kk;
            vec4 v = vec4(0.0);
            if (gk < p.K && gm < p.M) v = A4[(gk * p.M + gm) >> 2];
            if (i == 0) r0 = v; else r1 = v;
        }
    }
}

void load_b(uint lid, uint n0, uint kt, out vec4 r0, out vec4 r1) {
    if (p.b_layout == 0u) {
        // B[k][n]: one vec4 covers four consecutive n of a single k row.
        for (int i = 0; i < BQ; ++i) {
            const uint idx = lid + uint(i) * LANES;
            const uint kk = idx / TQ;
            const uint nq = idx - kk * TQ;
            const uint gn = n0 + nq * 4u;
            const uint gk = kt + kk;
            vec4 v = vec4(0.0);
            if (gk < p.K && gn < p.N) v = B4[(gk * p.N + gn) >> 2];
            if (i == 0) r0 = v; else r1 = v;
        }
    } else {
        // B[n][k]: one vec4 covers four consecutive k of a single n row.
        for (int i = 0; i < BQ; ++i) {
            const uint idx = lid + uint(i) * LANES;
            const uint n = idx >> 2;
            const uint kq = idx & 3u;
            const uint gn = n0 + n;
            const uint gk = kt + kq * 4u;
            vec4 v = vec4(0.0);
            if (gn < p.N && gk < p.K) v = B4[(gn * p.K + gk) >> 2];
            if (i == 0) r0 = v; else r1 = v;
        }
    }
}

void store_a(int buf, uint lid, vec4 r0, vec4 r1) {
    if (p.a_layout == 0u) {
        for (int i = 0; i < AQ; ++i) {
            const vec4 v = (i == 0) ? r0 : r1;
            const uint idx = lid + uint(i) * LANES;
            const uint r = idx >> 2;
            const uint kq = idx & 3u;
            const uint q = r >> 2;
            const uint c = r & 3u;
            As[buf][kq * 4u + 0u][q][c] = v.x;
            As[buf][kq * 4u + 1u][q][c] = v.y;
            As[buf][kq * 4u + 2u][q][c] = v.z;
            As[buf][kq * 4u + 3u][q][c] = v.w;
        }
    } else {
        for (int i = 0; i < AQ; ++i) {
            const vec4 v = (i == 0) ? r0 : r1;
            const uint idx = lid + uint(i) * LANES;
            const uint kk = idx / TQ;
            const uint mq = idx - kk * TQ;
            As[buf][kk][mq] = v;
        }
    }
}

void store_b(int buf, uint lid, vec4 r0, vec4 r1) {
    if (p.b_layout == 0u) {
        for (int i = 0; i < BQ; ++i) {
            const vec4 v = (i == 0) ? r0 : r1;
            const uint idx = lid + uint(i) * LANES;
            const uint kk = idx / TQ;
            const uint nq = idx - kk * TQ;
            Bs[buf][kk][nq] = v;
        }
    } else {
        for (int i = 0; i < BQ; ++i) {
            const vec4 v = (i == 0) ? r0 : r1;
            const uint idx = lid + uint(i) * LANES;
            const uint n = idx >> 2;
            const uint kq = idx & 3u;
            const uint q = n >> 2;
            const uint c = n & 3u;
            Bs[buf][kq * 4u + 0u][q][c] = v.x;
            Bs[buf][kq * 4u + 1u][q][c] = v.y;
            Bs[buf][kq * 4u + 2u][q][c] = v.z;
            Bs[buf][kq * 4u + 3u][q][c] = v.w;
        }
    }
}

// ---- main -----------------------------------------------------------------

void main() {
    const uint lid = uint(gl_LocalInvocationID.y) * uint(LX) + uint(gl_LocalInvocationID.x);
    const uint lrow = uint(gl_LocalInvocationID.y);
    const uint lcol = uint(gl_LocalInvocationID.x);
    const uint m0 = uint(gl_WorkGroupID.y) * uint(TILE);
    const uint n0 = uint(gl_WorkGroupID.x) * uint(TILE);
    const uint ktiles = (p.K + uint(KT) - 1u) / uint(KT);

    vec4 ra0 = vec4(0.0);
    vec4 ra1 = vec4(0.0);
    vec4 rb0 = vec4(0.0);
    vec4 rb1 = vec4(0.0);

    load_a(lid, m0, 0u, ra0, ra1);
    load_b(lid, n0, 0u, rb0, rb1);
    store_a(0, lid, ra0, ra1);
    store_b(0, lid, rb0, rb1);
    barrier();

    vec4 acc0 = vec4(0.0);
    vec4 acc1 = vec4(0.0);
    vec4 acc2 = vec4(0.0);
    vec4 acc3 = vec4(0.0);
    uint cur = 0u;
    for (uint t = 0u; t < ktiles; ++t) {
        const bool has_next = (t + 1u) < ktiles;
        if (has_next) {
            // Issue the next tile's global loads before consuming this one.
            load_a(lid, m0, (t + 1u) * uint(KT), ra0, ra1);
            load_b(lid, n0, (t + 1u) * uint(KT), rb0, rb1);
        }
        for (int kk = 0; kk < KT; ++kk) {
            const vec4 a = As[cur][kk][lrow];
            const vec4 b = Bs[cur][kk][lcol];
            acc0 += a.x * b;
            acc1 += a.y * b;
            acc2 += a.z * b;
            acc3 += a.w * b;
        }
        if (has_next) {
            store_a(int(cur ^ 1u), lid, ra0, ra1);
            store_b(int(cur ^ 1u), lid, rb0, rb1);
        }
        barrier();
        cur ^= 1u;
    }

    // 4x4 per-lane epilogue. The wrapper guarantees M, N are multiples of four,
    // so a lane's four rows/columns are all in range or all out of range.
    const uint row = m0 + lrow * 4u;
    const uint col = n0 + lcol * 4u;
    if (row < p.M && col < p.N) {
        C4[(row * p.N + col) >> 2] = acc0;
        C4[((row + 1u) * p.N + col) >> 2] = acc1;
        C4[((row + 2u) * p.N + col) >> 2] = acc2;
        C4[((row + 3u) * p.N + col) >> 2] = acc3;
    }
}
