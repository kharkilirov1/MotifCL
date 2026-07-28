# Vulkan fast-path lessons (RX 580 / Polaris / wave64)

- The 10^4x slowdown of the old backend decomposed into: per-call
  VkPipeline+descriptor+command-pool creation (~all of the 83 ms at 64^3),
  per-shape generated SPIR-V (made pipeline caching impossible by design),
  and scalar one-invocation-per-element kernels. Fixing the first two took
  64^3 matmul from 83,790 us to ~160 us wall before any kernel tuning.
- Submit+fence-wait floor on this driver is ~150-200 us wall per dispatch;
  GPU time (timestamps) for small ops is single-digit us. Per-op wall ratios
  vs OpenCL are therefore dominated by dispatch overhead on both sides at
  small shapes; the honest comparison at op level is wall-vs-wall measured
  identically (both dispatch+wait).
- glslc is NOT needed at build time: kernels/vulkan/*.comp are compiled
  offline by tools/gen_vulkan_spirv.py (glslc found in the Android NDK
  shader-tools; env MOTIFCL_GLSLC overrides) and committed as
  src/runtime/vulkan_spirv_kernels.inc. SPIR-V 1.0 target works on the
  Polaris Windows driver.
- Classic 16x16 shared-memory tile (256 threads, one output/thread, padded
  Bs[16][17]) already beats the tuned OpenCL path at 256^3..512^3
  (ratio 1.3-2.4) on this driver. Register blocking not yet needed to meet
  the 0.33 budget.
- The repo's legacy VK_MEMORY_PROPERTY_* constants are off by one bit
  (0x1 is really DEVICE_LOCAL, 0x2 HOST_VISIBLE): the old chooser therefore
  allocated everything in the 256MB BAR window and mapping still worked.
  New code uses spec-correct bits: device-local VRAM + staging transfers.
  Do not reuse the old constants in new code.
- Full VkPhysicalDeviceProperties/Limits struct transcription is guarded by
  static_assert sizeof==824/504 on x64 - if a field is ever mistyped the
  build fails instead of reading garbage timestampPeriod. RX 580 reports
  timestampPeriod=40ns, timestampValidBits=64.
- Timestamp queries: reset+write inside the same command buffer
  (vkCmdResetQueryPool core-1.0), TOP_OF_PIPE before / BOTTOM_OF_PIPE after,
  mask by timestampValidBits.
- Descriptor sets: pool of 256 sets / 2048 storage descriptors, free-list
  keyed by binding count, recycled only after the fence wait. A set created
  for layout (bindings=N, push=X) is safely reusable with any layout that
  has the same binding table (push range lives in the pipeline layout).
- Audit lesson: parity witnesses must cross the tile size (>=2 workgroups,
  >=2 K-tiles, non-multiples of 16) or cross-tile accumulation is untested;
  and device-op parity must ALSO live in the OpenCL-free standalone test,
  not only in the motifcl-linked test.
- Batch-mode footgun (open item): VulkanBuffer::upload during an open batch
  executes the copy immediately (own transfer submission) while recorded
  batch dispatches run later at batch_end - "compute then upload" ordering
  silently inverts. Don't upload into buffers written by an open batch.
- Batch keepalive is MANDATORY, not an optimization: op-internal temporaries
  (loss partials, backward scratch) legally destruct before batch_end; the
  recorded command buffer then references destroyed VkBuffers and reads
  zeros/garbage after submit. Symptom in practice: whole batched forward
  fine, loss exactly 0.0 (the CE partial died first). Fix: dispatch_cached
  pushes shared_ptr<VulkanBuffer::Impl> for every bound buffer into a
  batch_keepalive list cleared only after the fence wait.
- vkAllocateMemory per Tensor::empty is catastrophic for autograd churn on
  the Windows AMD driver: ~40 temporaries/step made a 1.4 ms training step
  take 26.8 ms. A power-of-two-bucketed buffer pool (release from the
  VulkanBuffer::Impl dtor, acquire in create_buffer, 1 GB cap) recovered
  18x. Keepalive also makes pooling safe: a pooled buffer can't be reissued
  while an unsubmitted batch still references it, because the keepalive
  shared_ptr delays the dtor (and thus the pool release) past the fence.
- GL_EXT_shader_8bit_storage in glslc (NDK r28) compiles fine to SPIR-V 1.0
  and byte-granular uint8_t stores work on the Polaris Windows driver -
  necessary for the 3-byte-per-group counter state (uint-word RMW would race
  across neighbouring groups).
- The counter SR tick is reproducible across backends: hash-seeded
  (seed ^ hash(elem)) stochastic rounding gave bit-identical state bytes
  OpenCL vs Vulkan on the first run - transcribe the hash and int-cast
  (trunc toward zero) exactly and parity is a byte compare, no tolerance.
- Vulkan train step ends up ~3.5-3.8x FASTER than the tuned OpenCL path on
  this driver, dominated by OpenCL's per-op queue overhead vs one-submit
  batches; per-op wall ratios (both sides ~200-600 us) are mostly dispatch
  overhead at these sizes, GPU time is single-digit us.
