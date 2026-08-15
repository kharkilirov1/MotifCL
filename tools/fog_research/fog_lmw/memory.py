import torch
from torch import nn

from .core import RMSNorm
from .motifs import LowRankCompareSelectAggregate


class PersistentLatentMemory(nn.Module):
    """
    memory motif.

    Append while under budget. Once the discrete slot budget N is exceeded,
    compress the combined old+new states back to N learned memory positions.
    """

    def __init__(self, d_model: int, compare_rank: int, memory_slots: int):
        super().__init__()
        self.memory_slots = memory_slots
        if memory_slots > 0:
            self.memory_query = nn.Parameter(torch.randn(memory_slots, d_model) * 0.02)
            self.compress = LowRankCompareSelectAggregate(d_model, compare_rank)
            self.norm = RMSNorm(d_model)
            self.gate = nn.Linear(2 * d_model, d_model)

    def forward(
        self, old: torch.Tensor | None, new: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        combined = new if old is None else torch.cat([old, new], dim=1)

        # N=0 means unbounded append-only MVP.
        if self.memory_slots <= 0 or combined.size(1) <= self.memory_slots:
            return combined, {"compressed": False, "compression_attention": None}

        b = combined.size(0)
        base = self.memory_query.unsqueeze(0).expand(b, -1, -1)
        update, attn = self.compress(self.norm(base), self.norm(combined))
        g = torch.sigmoid(self.gate(torch.cat([base, update], dim=-1)))
        memory = g * base + (1.0 - g) * update
        return memory, {"compressed": True, "compression_attention": attn}

    def forward_preserving(
        self,
        old: torch.Tensor | None,
        new: torch.Tensor,
        *,
        protected_slots: int,
    ) -> tuple[torch.Tensor, dict]:
        """Update memory while copying newest binding slots losslessly.

        Only the bounded-overflow case differs from :meth:`forward`.  The first
        ``protected_slots`` new states bypass learned compression; old memory
        and the remaining new workspace compete for the residual budget.
        """

        if protected_slots < 1 or protected_slots > new.size(1):
            raise ValueError("protected_slots must be in [1, new_slots]")
        protected = new[:, :protected_slots]
        remainder_parts = []
        if old is not None:
            remainder_parts.append(old)
        if new.size(1) > protected_slots:
            remainder_parts.append(new[:, protected_slots:])
        remainder = torch.cat(remainder_parts, dim=1) if remainder_parts else None
        # Newest protected bindings are always the prefix.  This is true even
        # before the capacity is reached, so memory[:, 0] has a stable meaning
        # at every recurrent depth.
        combined = protected if remainder is None else torch.cat([protected, remainder], dim=1)
        if self.memory_slots <= 0 or combined.size(1) <= self.memory_slots:
            return combined, {
                "compressed": False,
                "compression_attention": None,
                "protected_slots": protected_slots,
            }
        if protected_slots > self.memory_slots:
            raise ValueError("protected slots exceed bounded memory capacity")

        residual_budget = self.memory_slots - protected_slots
        if residual_budget == 0:
            return protected, {
                "compressed": True,
                "compression_attention": None,
                "protected_slots": protected_slots,
            }
        if remainder is None or remainder.size(1) <= residual_budget:
            tail = remainder
            attn = None
            compressed = False
        else:
            batch = new.size(0)
            base = self.memory_query[:residual_budget].unsqueeze(0).expand(batch, -1, -1)
            update, attn = self.compress(self.norm(base), self.norm(remainder))
            gate = torch.sigmoid(self.gate(torch.cat([base, update], dim=-1)))
            tail = gate * base + (1.0 - gate) * update
            compressed = True
        memory = protected if tail is None else torch.cat([protected, tail], dim=1)
        return memory, {
            "compressed": compressed,
            "compression_attention": attn,
            "protected_slots": protected_slots,
        }


class ReusableLatentMemory(nn.Module):
    """Fixed-size recurrent workspace used by query-bound v2.

    Slot identities are reused rather than appending ``K`` new states at every
    reasoning step.  Protected prefix slots are copied from the newest writer
    output exactly; only the auxiliary tail is softly gated against its prior
    value.  ``compress`` remains as a parameter-free compatibility attribute,
    but no content-independent compression can touch the exact payload carrier.
    """

    def __init__(self, d_model: int, latent_slots: int):
        super().__init__()
        if latent_slots < 1:
            raise ValueError("latent_slots must be positive")
        self.memory_slots = latent_slots
        self.norm = RMSNorm(d_model)
        self.gate = nn.Linear(2 * d_model, d_model)
        self.compress = nn.Identity()

    def forward_preserving(
        self,
        old: torch.Tensor | None,
        new: torch.Tensor,
        *,
        protected_slots: int,
    ) -> tuple[torch.Tensor, dict]:
        if new.ndim != 3 or new.size(1) != self.memory_slots:
            raise ValueError("new memory must have shape [batch, latent_slots, d_model]")
        if protected_slots < 1 or protected_slots > self.memory_slots:
            raise ValueError("protected_slots must be in [1, latent_slots]")
        if old is None:
            memory = new
            reused = False
        else:
            if old.shape != new.shape:
                raise ValueError("reusable old/new memory shapes must match")
            protected = new[:, :protected_slots]
            if protected_slots == self.memory_slots:
                memory = protected
            else:
                old_tail = old[:, protected_slots:]
                new_tail = new[:, protected_slots:]
                gate = torch.sigmoid(
                    self.gate(
                        torch.cat(
                            [self.norm(old_tail), self.norm(new_tail)], dim=-1
                        )
                    )
                )
                tail = gate * old_tail + (1.0 - gate) * new_tail
                memory = torch.cat([protected, tail], dim=1)
            reused = True
        return memory, {
            "compressed": False,
            "compression_attention": None,
            "protected_slots": protected_slots,
            "reused": reused,
        }
