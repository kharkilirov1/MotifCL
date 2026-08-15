from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .core import RMSNorm
from .motifs import ExpandCompressFFN, LowRankCompareSelectAggregate


class RegisterStateMemory(nn.Module):
    """Parameter-free fixed register file for register_machine_v3."""

    def __init__(self, latent_slots: int):
        super().__init__()
        self.memory_slots = latent_slots
        self.compress = nn.Identity()

    def forward_preserving(
        self,
        old: torch.Tensor | None,
        new: torch.Tensor,
        *,
        protected_slots: int,
    ) -> tuple[torch.Tensor, dict]:
        if new.ndim != 3 or new.size(1) != self.memory_slots:
            raise ValueError("register state must have [batch, latent_slots, d_model]")
        if protected_slots != self.memory_slots:
            raise ValueError("register_machine_v3 preserves the complete register file")
        if old is not None and old.shape != new.shape:
            raise ValueError("old/new register files must have identical shape")
        return new, {
            "compressed": False,
            "compression_attention": None,
            "protected_slots": protected_slots,
            "reused": old is not None,
        }


class GeneratedOperatorBank(nn.Module):
    """Finite bank of shared low-rank generated-value operators.

    READ and IDENTITY are implicit candidates. The learned operators transform
    the value register conditioned on the current addressed proposal and the
    control register. All candidates are applied with the same parameters at
    every recurrent depth.
    """

    def __init__(self, d_model: int, n_operators: int, rank: int, *, hard_routing: bool = False):
        super().__init__()
        if n_operators < 1 or rank < 1:
            raise ValueError("generated operator bank needs positive count/rank")
        self.d_model = d_model
        self.n_generated = n_operators
        self.rank = rank
        self.hard_routing = bool(hard_routing)
        scale = 1.0 / math.sqrt(d_model)
        self.state_down = nn.Parameter(torch.randn(n_operators, rank, d_model) * scale)
        self.read_down = nn.Parameter(torch.randn(n_operators, rank, d_model) * scale)
        self.control_down = nn.Parameter(torch.randn(n_operators, rank, d_model) * scale)
        self.bias = nn.Parameter(torch.zeros(n_operators, rank))
        self.up = nn.Parameter(torch.randn(n_operators, d_model, rank) * (1.0 / math.sqrt(rank)))
        self.delta_logit = nn.Parameter(torch.full((n_operators, 1), -2.0))
        if d_model % 2 != 0:
            raise ValueError("generated operator bank requires even d_model for block-product motif")
        # Candidate order: READ, IDENTITY, BLOCK_PRODUCT, then learned operators.
        self.router = nn.Sequential(
            RMSNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, n_operators + 3),
        )

    @property
    def total_candidates(self) -> int:
        return self.n_generated + 3

    def forward(
        self,
        value: torch.Tensor,
        addressed: torch.Tensor,
        control: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        for name, tensor in {"value": value, "addressed": addressed, "control": control}.items():
            if tensor.ndim != 2 or tensor.shape != value.shape:
                raise ValueError(f"{name} must have [batch, d_model]")
        state_feature = torch.einsum("bd,ord->bor", value, self.state_down)
        read_feature = torch.einsum("bd,ord->bor", addressed, self.read_down)
        control_feature = torch.einsum("bd,ord->bor", control, self.control_down)
        # Explicit low-rank bilinear motif. Purely additive projected features
        # can fit one-step tables but are a poor inductive bias for generated
        # composition; multiplicative interaction is the minimal generic form
        # that can represent operator-compatible products while remaining cheap.
        h = (
            state_feature * read_feature
            + 0.25 * (state_feature + read_feature)
            + control_feature
            + self.bias.unsqueeze(0)
        )
        h = F.silu(h)
        delta = torch.einsum("bor,odr->bod", h, self.up)
        alpha = torch.sigmoid(self.delta_logit).view(1, self.n_generated, 1)
        generated = value[:, None, :] + alpha * delta
        value_pairs = value.reshape(value.size(0), self.d_model // 2, 2)
        read_pairs = addressed.reshape(addressed.size(0), self.d_model // 2, 2)
        vr, vi = value_pairs.unbind(dim=-1)
        rr, ri = read_pairs.unbind(dim=-1)
        block_product = torch.stack(
            [vr * rr - vi * ri, vr * ri + vi * rr], dim=-1
        ).reshape_as(value)
        # Only direction is semantic for the tied cosine readout. Keeping a
        # fixed RMS prevents multiplicative norm drift across recurrent depth.
        block_product = F.normalize(block_product.float(), dim=-1).to(value.dtype) * math.sqrt(self.d_model)
        candidates = torch.cat(
            [
                addressed[:, None, :],
                value[:, None, :],
                block_product[:, None, :],
                generated,
            ],
            dim=1,
        )
        router_input = torch.cat([value, addressed, control], dim=-1)
        logits = self.router(router_input)
        probs = torch.softmax(logits.float(), dim=-1).to(candidates.dtype)
        if self.hard_routing:
            hard = torch.nn.functional.one_hot(
                probs.argmax(dim=-1), num_classes=probs.size(-1)
            ).to(probs.dtype)
            weights = hard + probs - probs.detach() if self.training else hard
        else:
            weights = probs
        new_value = torch.einsum("bo,bod->bd", weights, candidates)
        return new_value, {
            "operator_logits": logits,
            "operator_probs": probs,
            "operator_weights": weights,
            "selected_operator": weights.argmax(dim=-1),
            "read_probability": probs[:, 0],
            "identity_probability": probs[:, 1],
            "block_product_probability": probs[:, 2],
        }


class RegisterMachineCell(nn.Module):
    """Shared typed-register transition used by FOG register_machine_v3.

    Register roles:
      slot 0 -- value/address register (protected from workspace mixing)
      slot 1 -- control register
      slots 2+ -- scratch/workspace registers
    """

    def __init__(
        self,
        d_model: int,
        latent_slots: int,
        compare_rank: int,
        machine_ff: int,
        n_generated_operators: int,
        operator_rank: int,
        hard_routing: bool = False,
    ):
        super().__init__()
        if latent_slots < 2:
            raise ValueError("register machine requires value + control registers")
        self.d_model = d_model
        self.latent_slots = latent_slots
        self.operator_bank = GeneratedOperatorBank(
            d_model, n_generated_operators, operator_rank, hard_routing=hard_routing
        )
        self.tail_norm = RMSNorm(d_model)
        self.tail_mix = LowRankCompareSelectAggregate(d_model, compare_rank)
        self.tail_gate = nn.Sequential(
            RMSNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
        )
        self.tail_ff = ExpandCompressFFN(d_model, machine_ff)
        self.halt_norm = RMSNorm(2 * d_model)
        self.halt_head = nn.Linear(2 * d_model, 1)
        # A tiny explicit role code keeps control/scratch identities stable.
        self.tail_role = nn.Parameter(torch.randn(latent_slots - 1, d_model) * 0.02)

    def _halt(self, registers: torch.Tensor) -> torch.Tensor:
        value = registers[:, 0]
        control = registers[:, 1]
        return torch.sigmoid(
            self.halt_head(self.halt_norm(torch.cat([value, control], dim=-1)))
        ).squeeze(-1)

    def forward(
        self,
        old: torch.Tensor | None,
        proposal: torch.Tensor,
        *,
        initial_value: torch.Tensor | None = None,
        initial_control: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if proposal.ndim != 3 or proposal.size(1) != self.latent_slots or proposal.size(2) != self.d_model:
            raise ValueError("proposal must have [batch, latent_slots, d_model]")
        initialized = old is None
        if initialized:
            batch = proposal.size(0)
            if initial_value is None or initial_value.shape != (batch, self.d_model):
                raise ValueError("first machine tick requires initial_value [batch,d_model]")
            if initial_control is None:
                initial_control = proposal[:, 1]
            if initial_control.shape != (batch, self.d_model):
                raise ValueError("initial_control must have [batch,d_model]")
            tail = proposal[:, 1:].clone()
            tail[:, 0] = initial_control
            old = torch.cat([initial_value[:, None, :], tail], dim=1)
        if old.shape != proposal.shape:
            raise ValueError("old/proposal register files must match")

        value = old[:, 0]
        control = old[:, 1]
        addressed = proposal[:, 0]
        new_value, op_stats = self.operator_bank(value, addressed, control)

        old_tail = old[:, 1:]
        proposal_tail = proposal[:, 1:] + self.tail_role.unsqueeze(0)
        gate = torch.sigmoid(
            self.tail_gate(
                torch.cat(
                    [self.tail_norm(old_tail), self.tail_norm(proposal_tail)], dim=-1
                )
            )
        )
        tail = gate * old_tail + (1.0 - gate) * proposal_tail
        full_context = torch.cat([new_value[:, None, :], tail], dim=1)
        mixed, mix_weights = self.tail_mix(
            self.tail_norm(tail), self.tail_norm(full_context)
        )
        tail = self.tail_ff(tail + 0.1 * mixed)
        registers = torch.cat([new_value[:, None, :], tail], dim=1)
        return registers, {
            "initialized_from_query": initialized,
            **op_stats,
            "tail_mix_attention": mix_weights,
            "halt_probability": self._halt(registers),
        }

