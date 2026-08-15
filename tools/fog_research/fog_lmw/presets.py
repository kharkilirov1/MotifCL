from __future__ import annotations

from .config import FOGReasonerConfig


FOG_10M_VOCAB_SIZE = 8192
FOG_10M_PARAMETER_COUNT = 10_035_848
FOG_BINDING_V2_10M_PARAMETER_COUNT = 10_000_039
FOG_MACHINE_V3_10M_PARAMETER_COUNT = 10_245_433


def fog_10m_config(
    *,
    vocab_size: int = FOG_10M_VOCAB_SIZE,
    max_seq_len: int = 512,
    reasoning_steps: int = 4,
    dropout: float = 0.1,
) -> FOGReasonerConfig:
    """Balanced ~10M preset used by the real-data training pipeline.

    The exact reference count is 10,035,848 only for vocab=8192 and seq=512.
    Four slots are the only toy geometry with a positive mechanistic run so
    far.  An eight-slot memory cap makes compression active at steps 3+.
    """
    return FOGReasonerConfig(
        vocab_size=vocab_size,
        d_model=320,
        n_heads=5,
        n_layers=4,
        d_ff=1344,
        max_seq_len=max_seq_len,
        dropout=dropout,
        initializer_range=0.02,
        latent_slots=4,
        reasoning_steps=reasoning_steps,
        compare_rank=80,
        planner_ff=1344,
        memory_slots=8,
        n_reasoning_modes=8,
        diversity_weight=1e-3,
        route_entropy_weight=0.0,
    )


def fog_binding_v2_10m_config(
    *,
    vocab_size: int = FOG_10M_VOCAB_SIZE,
    max_seq_len: int = 512,
    reasoning_steps: int = 4,
    dropout: float = 0.1,
) -> FOGReasonerConfig:
    """10M query-bound revision with an exact protected payload channel.

    This is a distinct architecture, not a silent reinterpretation of the
    released legacy weights.  It keeps the lexical 10M geometry, expands only
    the auxiliary workspace FFN, and uses one protected binding carrier plus
    direct first-token readout.
    """

    return FOGReasonerConfig(
        vocab_size=vocab_size,
        d_model=320,
        n_heads=5,
        n_layers=4,
        d_ff=1344,
        max_seq_len=max_seq_len,
        dropout=dropout,
        initializer_range=0.02,
        latent_slots=4,
        reasoning_steps=reasoning_steps,
        compare_rank=80,
        planner_ff=2330,
        memory_slots=4,
        n_reasoning_modes=8,
        diversity_weight=1e-3,
        route_entropy_weight=0.0,
        architecture_version="query_bound_v2",
        binding_mode="query_conditioned",
        readout_mode="direct_latent",
        protected_binding_slots=1,
        binding_offsets=(2,),
    )


def fog_machine_v3_10m_config(
    *,
    vocab_size: int = FOG_10M_VOCAB_SIZE,
    max_seq_len: int = 512,
    reasoning_steps: int = 8,
    dropout: float = 0.1,
    adaptive_halting: bool = False,
) -> FOGReasonerConfig:
    """~10.25M model-ready register-machine preset.

    It preserves the v2 exact query-bound proposal path, but all recurrent
    steps after initialization pass through a shared typed-register cell with
    READ/IDENTITY plus four generated operators. The complete K=4 register file
    is fixed-size and JVP-probeable.
    """
    return FOGReasonerConfig(
        vocab_size=vocab_size,
        d_model=320,
        n_heads=5,
        n_layers=4,
        d_ff=1344,
        max_seq_len=max_seq_len,
        dropout=dropout,
        initializer_range=0.02,
        latent_slots=4,
        reasoning_steps=reasoning_steps,
        compare_rank=80,
        planner_ff=800,
        memory_slots=4,
        n_reasoning_modes=8,
        diversity_weight=1e-3,
        route_entropy_weight=0.0,
        architecture_version="register_machine_v3",
        binding_mode="query_conditioned",
        readout_mode="direct_latent",
        protected_binding_slots=1,
        binding_offsets=(2,),
        binding_query_update="primary_recurrent",
        machine_operator_count=4,
        machine_operator_rank=48,
        machine_ff=640,
        machine_min_steps=1,
        machine_halt_threshold=0.95,
        machine_adaptive_halting=adaptive_halting,
        machine_hard_routing=True,
    )
