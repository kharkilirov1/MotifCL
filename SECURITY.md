# Security policy

MotifCL is pre-1.0 research software. Security fixes target the current default development line;
older commits, binaries, model artifacts, and benchmark snapshots are not supported releases.

Report security-sensitive issues through GitHub private vulnerability reporting when available.
If private reporting is unavailable, contact the repository owner privately rather than opening a
public issue.

Include the affected commit, platform/backend, impact, minimal reproduction, and whether the issue
requires an untrusted model, tokenizer, GGUF/safetensors file, shader cache, environment variable,
or filesystem path.

## Trust boundary

MotifCL is not a sandbox. Treat untrusted model and tokenizer artifacts as hostile input. The
project includes native parsers, GPU kernels, file-backed caches, a local HTTP wrapper, and
optional Python bindings; malformed inputs may exercise memory-safety and denial-of-service
boundaries.

Security reports are especially useful for:

- integer overflow or bounds errors in GGUF/safetensors/tokenizer parsing;
- malformed shape or quantization metadata;
- path traversal or unsafe cache/model discovery;
- local server exposure beyond the documented loopback boundary;
- use-after-free or lifetime errors across backend/tensor ownership;
- shader or descriptor metadata causing out-of-bounds GPU access;
- silent checkpoint corruption.

Dependency or driver vulnerabilities should also be reported to the relevant upstream project.
