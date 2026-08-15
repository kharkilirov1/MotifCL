from dataclasses import dataclass


@dataclass
class FOGReasonerConfig:
    # Token / backbone geometry
    vocab_size: int = 8192
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 1024
    dropout: float = 0.0
    initializer_range: float = 0.02

    # Latent workspace geometry
    latent_slots: int = 8          # K: new thought slots per reasoning step
    reasoning_steps: int = 4       # R: recurrent latent refinement depth
    compare_rank: int = 64         # independent discrete rank budget for compare
    planner_ff: int = 512
    memory_slots: int = 32         # N: cap before learned compression; 0 = append-only

    # Route/control: generic reasoning modes. These can later be bootstrapped
    # from multilingual CoT, but they are NOT literal language token embeddings.
    n_reasoning_modes: int = 4

    # Optional auxiliary losses
    diversity_weight: float = 0.01
    route_entropy_weight: float = 0.0

    # Architecture contract.  ``legacy_v1`` preserves byte-for-byte loading of
    # the released checkpoints.  ``query_bound_v2`` keeps the final prompt
    # state as an explicit query anchor, protects the primary bound slot from
    # memory compression, and reads the first answer token directly from the
    # final latent state instead of asking a fresh BOS token to rediscover it.
    architecture_version: str = "legacy_v1"
    binding_mode: str = "summary_slots"
    readout_mode: str = "bos_decoder"
    protected_binding_slots: int = 0
    # Relative address→payload edges used only by query_bound_v2.  For each
    # offset o, token i is a candidate address for exact payload token i+o.
    binding_offsets: tuple[int, ...] = ()
    # Protected-binding query update rule. ``static`` preserves released v2
    # semantics. ``primary_recurrent`` feeds the previous protected payload
    # back as the next binding address while leaving auxiliary workspace
    # conditioning unchanged.
    binding_query_update: str = "static"

    # Register-machine v3. These fields are ignored by legacy/v2 and keep old
    # checkpoints source-compatible. v3 adds a shared finite operator bank on
    # top of the exact query-bound proposal path, typed persistent registers,
    # and an optional learned HALT signal.
    machine_operator_count: int = 0       # learned generated operators; READ/IDENTITY are implicit
    machine_operator_rank: int = 0
    machine_ff: int = 0
    machine_min_steps: int = 1
    machine_halt_threshold: float = 0.95
    machine_adaptive_halting: bool = False
    machine_hard_routing: bool = False

    def effective_memory_slots(self, reasoning_steps: int | None = None) -> int:
        """Maximum decoder-visible latent states for this architecture."""
        steps = self.reasoning_steps if reasoning_steps is None else reasoning_steps
        if steps <= 0:
            return 0
        if self.architecture_version in {"query_bound_v2", "register_machine_v3"}:
            return self.latent_slots
        size = steps * self.latent_slots
        return min(size, self.memory_slots) if self.memory_slots > 0 else size

    def validate(self) -> None:
        checks = {
            "vocab_size must be > 0": self.vocab_size > 0,
            "d_model must be > 0": self.d_model > 0,
            "n_heads must be > 0": self.n_heads > 0,
            "d_model must be divisible by n_heads": (
                self.n_heads > 0 and self.d_model % self.n_heads == 0
            ),
            "n_layers must be >= 0": self.n_layers >= 0,
            "compare_rank must be > 0": self.compare_rank > 0,
            "latent_slots must be > 0": self.latent_slots > 0,
            "reasoning_steps must be >= 0": self.reasoning_steps >= 0,
            "d_ff must be >= d_model": self.d_ff >= self.d_model,
            "planner_ff must be >= d_model": self.planner_ff >= self.d_model,
            "max_seq_len must be > 0": self.max_seq_len > 0,
            "memory_slots must be >= 0": self.memory_slots >= 0,
            "n_reasoning_modes must be >= 0": self.n_reasoning_modes >= 0,
            "dropout must be in [0, 1)": 0.0 <= self.dropout < 1.0,
            "initializer_range must be > 0": self.initializer_range > 0.0,
            "architecture_version is unsupported": self.architecture_version
            in {"legacy_v1", "query_bound_v2", "register_machine_v3"},
            "binding_mode is unsupported": self.binding_mode
            in {"summary_slots", "query_conditioned"},
            "readout_mode is unsupported": self.readout_mode
            in {"bos_decoder", "direct_latent"},
            "protected_binding_slots must be >= 0": self.protected_binding_slots >= 0,
            "protected_binding_slots cannot exceed latent_slots": (
                self.protected_binding_slots <= self.latent_slots
            ),
            "protected binding slots fit the bounded memory": (
                self.memory_slots == 0
                or self.protected_binding_slots <= self.memory_slots
            ),
            "binding_offsets must contain only positive integers": all(
                isinstance(offset, int) and offset > 0
                for offset in self.binding_offsets
            ),
            "binding_offsets must be unique": len(set(self.binding_offsets))
            == len(self.binding_offsets),
            "binding_query_update is unsupported": self.binding_query_update
            in {"static", "primary_recurrent"},
            "machine_operator_count must be >= 0": self.machine_operator_count >= 0,
            "machine_operator_rank must be >= 0": self.machine_operator_rank >= 0,
            "machine_ff must be >= 0": self.machine_ff >= 0,
            "machine_min_steps must be >= 1": self.machine_min_steps >= 1,
            "machine_min_steps cannot exceed reasoning_steps": (
                self.reasoning_steps == 0 or self.machine_min_steps <= self.reasoning_steps
            ),
            "machine_halt_threshold must be in (0,1)": 0.0 < self.machine_halt_threshold < 1.0,
        }
        for message, valid in checks.items():
            if not valid:
                raise ValueError(message)
        if self.architecture_version == "legacy_v1":
            if (
                self.binding_mode != "summary_slots"
                or self.readout_mode != "bos_decoder"
                or self.protected_binding_slots != 0
                or self.binding_offsets
                or self.binding_query_update != "static"
            ):
                raise ValueError(
                    "legacy_v1 requires summary_slots/bos_decoder and no protected slots"
                )
        else:
            prefix = (
                "register_machine_v3"
                if self.architecture_version == "register_machine_v3"
                else "query_bound_v2"
            )
            if self.binding_mode != "query_conditioned":
                raise ValueError(f"{prefix} requires binding_mode=query_conditioned")
            if self.readout_mode != "direct_latent":
                raise ValueError(f"{prefix} requires readout_mode=direct_latent")
            if self.protected_binding_slots < 1:
                raise ValueError(f"{prefix} requires at least one protected slot")
            if not self.binding_offsets:
                raise ValueError(f"{prefix} requires at least one binding offset")
            if self.architecture_version == "register_machine_v3":
                if self.binding_query_update != "primary_recurrent":
                    raise ValueError("register_machine_v3 requires primary_recurrent binding")
                if self.latent_slots < 2:
                    raise ValueError("register_machine_v3 requires value + control registers")
                if self.machine_operator_count < 1:
                    raise ValueError("register_machine_v3 requires machine_operator_count >= 1")
                if self.machine_operator_rank < 1:
                    raise ValueError("register_machine_v3 requires machine_operator_rank >= 1")
                if self.machine_ff < self.d_model:
                    raise ValueError("register_machine_v3 requires machine_ff >= d_model")
