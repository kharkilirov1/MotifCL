# Context (why this matters)

I'm porting MotifCL — a C++17/OpenCL neural compute framework for legacy AMD
GPUs (target device: Radeon RX 580 / gfx803 / Polaris, wave64) — from
OpenCL-first to Vulkan-first. The end state is a Vulkan-only training loop so
OpenCL becomes optional and eventually removable. This unblocks cross-platform
ports and resolves a growing split between the README/SPEC (still "an OpenCL
framework") and the code (already +4700 lines of Vulkan backend).

This is a multi-day, end-to-end engineering task. Take it as far as you can
autonomously; intervene points are under "Pause for the user."

# Critical constraint: optimize AS you port, not after

The previous Vulkan backend passes parity tests but is unusably slow: the
checked-in microbench shows a 64x64x64 F32 matmul at ~83 ms (0.006 GFLOP/s) —
roughly 10^4x off a reasonable implementation. Causes are known: each call
builds a fresh VkPipeline and the generated SPIR-V does scalar per-element
work. We will NOT accept "correct but scalar" ports that get fixed in a later
optimization phase. Every kernel you port must land with a workgroup design,
pipeline reuse, and a measured perf number inside its perf budget (see below).
A correct-but-slow port is a partial port, not a done port.

# Repo facts (verified 2026-07-03)

- Working dir: C:\Users\Kharki\Desktop\motifcl_production\motifcl_production
- Branch: fog-qkv-split. HEAD = 29f2702 ("...backward training segfault"). That
  segfault is real, uncovered by CTest, separate debt — see Pre-work.
- Plan: docs/VULKAN_PORT_PROTOCOL.md (read it first; defines slices 0-5 and
  invariants 1-5).
- Slices 0-3 DONE with OpenCL-free witnesses: F32 matmul (incl. M=1,
  transpose-B), softmax, RMSNorm, SwiGLU, add, scalar-scale Q8_0xQ8_0,
  non-causal batch=1 GQA, SGD update, CounterStateLinear decode/inference/
  backward-input. NOTE: these are correctness witnesses only — most are scalar
  and pipeline-per-call. Part of Slice 4/5 work is retrofitting the perf
  requirements onto already-ported ops as you touch their files.
- Slice 4 (autograd backward + training on Vulkan): NOT done.
- Slice 5 (OpenCL optional + CI/docs default to Vulkan): partial.
- Device: RX 580; vulkaninfo reports VK_KHR_8bit_storage with
  storageBuffer8BitAccess=true, uniformAndStorageBuffer8BitAccess=true.
  Wave size 64. Polaris has only basic subgroup support — do not assume
  subgroup reduction/scan primitives are available; fall back to workgroup
  shared-memory reductions.
- Env caveat: MinGW/GCC CTests fail to launch with 0xc0000139 because
  libgcc_s_seh-1.dll / libstdc++-6.dll / libwinpthread-1.dll are not on PATH
  when run from build/. 31/31 were green on 2026-06-28 under the same
  toolchain — this is env, not code. Fix it first.

# Working builds

GCC/Ninja (Strawberry g++ 13.2.0), tests on:

  cmake -S . -B build/port-vk -G Ninja \
    -DCMAKE_CXX_COMPILER=C:/Strawberry/c/bin/g++.exe \
    -DCMAKE_BUILD_TYPE=Release -DMOTIFCL_BUILD_TESTS=ON \
    -DMOTIFCL_BUILD_PYTHON=OFF -DMOTIFCL_BUILD_EXAMPLES=OFF \
    -DMOTIFCL_BUILD_TOOLS=OFF -DMOTIFCL_BUILD_BENCHMARKS=ON \
    -DMOTIFCL_WARNINGS_AS_ERRORS=OFF
  cmake --build build/port-vk -j 4
  ctest --test-dir build/port-vk --output-on-failure --timeout 300

OpenCL-off witness (must keep passing throughout):

  cmake -S . -B build/port-vk-off -G Ninja -DMOTIFCL_ENABLE_OPENCL=OFF \
    -DMOTIFCL_BUILD_TESTS=ON -DMOTIFCL_BUILD_PYTHON=ON
  cmake --build build/port-vk-off --target test_vulkan_runtime_standalone -j 4
  ninja -C build/port-vk-off -t commands test_vulkan_runtime_standalone | grep -i opencl
  # grep must print nothing

# Goal

Complete Slice 4 and Slice 5 of docs/VULKAN_PORT_PROTOCOL.md with every ported
kernel meeting both its parity assertion AND its perf budget. End state: a full
training step (forward -> backward -> optimizer) runs on Vulkan-backed Tensor
allocations with no OpenCL context, at >=40% of the OpenCL path's measured
throughput on the RX 580 for the op mix in the end-to-end witness.

# Non-negotiable invariants

Correctness invariants (from docs/VULKAN_PORT_PROTOCOL.md):

1. No OpenCL roundtrip masquerading as a Vulkan port. A host-download ->
   standalone Vulkan helper -> reupload path is a transition shim; if you ship
   one, label it a shim with a TODO for the missing native piece. Never present
   a shim as done.
2. Every ported slice needs an OpenCL-free witness (target/test that does not
   link OpenCL::OpenCL or include motifcl.hpp / Backend::create_opencl / Tensor
   / Buffer / Kernel / Program / OpenCLContext).
3. Fallback to OpenCL only behind an explicit, named env switch; the
   Vulkan-native test must cover the separate OpenCL-free path.
4. Vulkan-native means device-resident I/O; host roundtrip allowed only at the
   test upload/download boundary.
5. Parity: every Vulkan-native op matches a CPU reference (and the old OpenCL
   regression where present) within a stated tolerance, asserted in the
   witness test.

Performance invariants (NEW — do not violate):

6. No per-call VkPipeline creation in a hot path. Pipelines are keyed by
   (shader SPIR-V bytes + layout descriptor) and cached for the lifetime of
   the VulkanRuntime / Backend. Same for VkPipelineLayout and VkDescriptorSetLayout.
7. No per-call descriptor-set allocation in steady state. Use a descriptor pool
   with reuse, or push-constants for small uniform data (<=128 bytes) to avoid
   descriptor sets entirely where possible.
8. No scalar per-element SPIR-V in a compute- or bandwidth-bound op. Concretely:
   - matmul: tiled with Workgroup shared memory, register-blocked, one
     workgroup per output tile. Tile target 16x16 or 32x32 (match what
     MOTIFCL_MATMUL_F32_TILE=16 uses for OpenCL).
   - softmax / RMSNorm / row reductions: one workgroup (wave64) per row, or
     one subgroup if you have confirmed subgroup ops; otherwise shared-memory
     reduction in the workgroup.
   - elementwise (add/activation): coalesced 1-element-per-lane or vec2/vec4
     load when storageBuffer8BitAccess or layout allows.
9. Batch dispatches: a forward pass should record N dispatches into one
   primary command buffer and submit once, not submit-per-dispatch. (Start
   recording immediately; this becomes a hard requirement for the end-to-end
   training witness.)
10. Every ported op gets a benchmark with a recorded median before the port is
    called done. See the perf budget and the profiling harness below.

# Perf budget (per op, vs the existing OpenCL path on RX 580)

OpenCL is the regression baseline on the same device, so ratios are the honest
target. "Throughput" = bytes/s for memory-bound, FLOP/s for compute-bound,
measured at the op's representative shape.

| Op class                      | Examples                          | Budget (Vulkan / OpenCL) |
|-------------------------------|-----------------------------------|--------------------------|
| Elementwise / activation      | add, SwiGLU, GELU, scale          | >= 0.60                   |
| Row reductions                | softmax rows, RMSNorm, layernorm  | >= 0.40                   |
| F32 matmul (small/M=1)        | decode matmul, M<=8                | >= 0.33                   |
| F32 matmul (compute-bound)    | M,K,N >= 64, power-of-two          | >= 0.33 (stretch 0.50)    |
| Quantized matmul (Q8xQ8->F32) | scalar-scale path                 | >= 0.33                   |
| GQA forward / backward        | batch=1, head_dim<=64              | >= 0.25 (hardest)         |
| Optimizer step (SGD/Adam)     | per-row update                    | >= 0.40                   |
| End-to-end training step      | small GPT, Vulkan-only            | >= 0.40 of OpenCL wall    |

If you cannot reach a budget on Polaris after a genuine tiled/pooled attempt,
document why (specific Polaris/driver limitation, with the profiler output),
lower the bar explicitly in the perf record, and proceed — but never silently
ship something 100x off.

# Profiling harness (build this once, reuse for every op)

1. Use Vulkan timestamp queries (VK_QUERY_TYPE_TIMESTAMP) around each dispatch
   in the benchmark path. Convert to us using the device's timestampPeriod.
   Take the median of >=50 runs after >=5 warmup runs; report min/p50/p99.
2. Extend the existing microbench target (vk_microbench) — or add
   bench/vulkan_perf — so every ported op has an entry. Output a row per
   (op, shape) to reports/vulkan-perf/<op>.json with: op, shape, dtype,
   median_us, min_us, p99_us, achieved_gflops_or_gbs, opencl_baseline_us
   (if known), ratio, target_ratio, PASS/FAIL vs budget.
3. Also write a markdown summary reports/vulkan-perf/SUMMARY.md that the final
   report can cite. Update it each time an op lands.
4. Cross-check: the OpenCL baseline number comes from running the equivalent
   OpenCL op on the same RX 580 (the project already has OpenCL kernels and a
   tuner baseline). If you can't get an OpenCL number for an op, record
   "opencl_baseline: unavailable" rather than inventing one.

# Slice 4 work (autograd + training on Vulkan, optimized per op)

Port in dependency order; each item must land parity + perf before the next:

1. matmul backward (F32 tiled; F16 if F16 forward already exists) — reuse the
   same tiled BLAS-style kernel shape as forward with transposed loads.
2. softmax / RMSNorm / activation (SwiGLU, GELU) backward — wave-per-row.
3. attention (GQA) backward — extend the non-causal batch=1 F32 forward;
   causal/windowed/masked may stay OpenCL legacy, explicitly marked.
4. graph capture without recording cl_mem — capture/replay binds Vulkan
   device buffers; this is where batched command-buffer recording (invariant 9)
   becomes mandatory.
5. compact-counter fused state update on Vulkan (currently explicitly
   rejected) — port row-stats + stochastic counter tick, one workgroup per row.

End-to-end witness (Slice 4 done): one SGD training step on a small GPT or
transformer block, every Tensor allocated via Backend::create_vulkan(), no
OpenCL context, loss decreases over a few steps on a synthetic target, and the
whole step's wall time is >=40% of the OpenCL path on the same shape. Add as
test_vulkan_train_step + a bench entry; skip cleanly when no Vulkan device.

# Slice 5 work (OpenCL optional)

- Main motifcl library compiles under -DMOTIFCL_ENABLE_OPENCL=OFF for the
  surface Slice 4 covers.
- Gate OpenCL-only tests, Python bindings, examples, tools behind
  MOTIFCL_ENABLE_OPENCL in tests/CMakeLists.txt and related CMakeLists.
- Add a CI smoke job (OpenCL-off build + Vulkan-only tests). Mirror the
  existing workflow; don't invent a new CI provider.
- Update README/SPEC: MotifCL is "Vulkan-first, OpenCL optional." Update
  DeviceType. Move OpenCL docs to a legacy-backend section.

# Definition of done (measurable; do not claim done until all hold)

- [ ] ctest --test-dir build/port-vk green: every pre-existing test + new
      Slice 4 witnesses, on the RX 580.
- [ ] OpenCL-off build (build/port-vk-off) compiles the main library for the
      ported surface and passes test_vulkan_runtime_standalone,
      test_vulkan_backend, test_vulkan_train_step.
- [ ] Link line of every new Vulkan-native test contains no OpenCL symbol.
- [ ] No shim presented as a native port; every transition path labeled.
- [ ] Every Vulkan-native op has a parity assertion with a stated tolerance.
- [ ] Every Vulkan-native op has a perf record in reports/vulkan-perf/ meeting
      its budget (or a documented, explicit lowering with profiler evidence).
- [ ] Invariant audit: no per-call VkPipeline / descriptor-set allocation in
      any ported hot path (prove with a grep + a 1000-iteration timing showing
      flat or sublinear time, not linear-in-calls).
- [ ] docs/VULKAN_PORT_PROTOCOL.md Slice 4 and Slice 5 statuses updated to
      what is actually shipped, with exact witness + bench commands.

# Pre-work (before any Slice 4 code)

1. Fix the CTest launch env so 0xc0000139 stops. Prepend the Strawberry bin
   dir + mingw runtime DLL location to PATH for test runs. Confirm by running
   the existing suite green before touching anything.
2. Add a regression reproducer for the HEAD "backward training segfault"
   warning. Even if you can't fix it immediately, a test that reproduces it on
   the OpenCL path stops silent regression and tells us whether the Vulkan
   backward port also exhibits it. If the reproducer passes (segfault gone in
   current tree), say so plainly.
3. Build the profiling harness and the perf-record format once, before the
   first kernel port, so every port lands with a number from day one.

# Pause for the user only when

- a destructive/irreversible action is needed (force-push, deleting a build
  dir with uncommitted work, history rewrite, system package installs);
- a real scope change is needed (e.g. you discover Slice 4 needs a new IR for
  graph capture — that's a different project);
- only the user can supply input (device not attached, decision on which model
  family is the end-to-end witness).

Otherwise proceed. Do not ask "want me to...?" mid-task; do the work.

# Working style

- Operate autonomously. End your turn only when the task is complete or you're
  blocked on input only the user can give. Before ending, check your last
  paragraph: if it's a plan, analysis, question, next-steps list, or a promise
  about undone work, do that work now with tool calls.
- When you have enough to act, act. Don't re-derive established facts, don't
  re-litigate decided calls, don't survey options you won't pursue. If
  weighing a choice, give a recommendation, not a survey.
- Don't add features, refactor, or introduce abstractions beyond the port.
  A backward kernel doesn't need surrounding cleanup. Do the simplest thing
  that satisfies invariants 1-10 and the perf budget.
- Delegate independent subtasks to parallel subagents AFTER the shared
  plumbing exists (e.g. once pipeline-cache + descriptor-pool + perf harness
  are in, "matmul backward" and "RMSNorm backward" can proceed in parallel).
  Keep working while they run; intervene only if one goes off track.

# Ground your progress claims

Before reporting progress, audit each claim against a tool result from this
session: a build log, a green ctest run, a perf JSON with PASS vs budget, a
grep of the link line. Only report work you can point to evidence for. If
something isn't verified, say so. If tests fail, say so with the output. If
you skipped a step, say that. When something is done and verified, state it
plainly without hedging.

Self-check cadence: every op should land as parity -> bench -> audit. The audit
is a fresh-context verifier subagent that re-reads docs/VULKAN_PORT_PROTOCOL.md
and the relevant source, and checks your change against invariants 1-10, the
perf budget, and the definition of done. Treat its findings as authoritative;
fix before continuing to the next op.

# Memory

Store one lesson per file under reports/port-lessons/ (append to an existing
note rather than duplicating). Record: confirmed optimization approaches,
Polaris/SPIR-V gotchas (wave64 behavior, what tiled sizes won't compile,
whether push constants worked, descriptor-pool sizing), which op missed its
budget and why, anything surprising. Don't record what the repo or chat
already shows.

# Final summary (when you finish or stop)

Write it as a re-grounding for someone who hasn't watched you work. Open with
the outcome in one sentence ("Slice 4 matmul/softmax/RMSNorm/SwiGLU backward
are Vulkan-native, green, and within perf budget; GQA backward and graph
capture are not, here's why"). Then: what passed (exact ctest command + counts),
the perf table from reports/vulkan-perf/SUMMARY.md (op, ratio vs OpenCL,
PASS/FAIL), what's still a shim, what's blocked, and the one or two things you
need from me. Drop the working shorthand — spell out terms, no arrow chains, no
labels you invented mid-task. If you must choose between short and clear,
choose clear.
