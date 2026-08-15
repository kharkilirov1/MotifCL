from .config import FOGReasonerConfig
from .model import FOGLatentReasoner
from .resource import MotifBudget, allocate_equal_cost_discrete, compare_frobenius_error
from .presets import (
    FOG_10M_PARAMETER_COUNT,
    FOG_10M_VOCAB_SIZE,
    FOG_BINDING_V2_10M_PARAMETER_COUNT,
    FOG_MACHINE_V3_10M_PARAMETER_COUNT,
    fog_10m_config,
    fog_binding_v2_10m_config,
    fog_machine_v3_10m_config,
)

__all__ = [
    "FOGReasonerConfig",
    "FOGLatentReasoner",
    "MotifBudget",
    "allocate_equal_cost_discrete",
    "compare_frobenius_error",
    "FOG_10M_PARAMETER_COUNT",
    "FOG_10M_VOCAB_SIZE",
    "FOG_BINDING_V2_10M_PARAMETER_COUNT",
    "FOG_MACHINE_V3_10M_PARAMETER_COUNT",
    "fog_10m_config",
    "fog_binding_v2_10m_config",
    "fog_machine_v3_10m_config",
]
