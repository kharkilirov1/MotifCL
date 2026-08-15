import math
import torch
from torch import nn
from torch.nn import functional as F

from .core import RMSNorm
from .motifs import (
    AddressedPayloadBinding,
    ExpandCompressFFN,
    LowRankCompareSelectAggregate,
    ReasoningModeRouter,
)


class LatentPlanner(nn.Module):
    """
    Generates K latent thoughts in parallel, then lets those thoughts interact
    bidirectionally before reinjection into the model.
    """

    def __init__(
        self,
        d_model: int,
        latent_slots: int,
        compare_rank: int,
        planner_ff: int,
        n_reasoning_modes: int,
    ):
        super().__init__()
        self.latent_slots = latent_slots
        self.query = nn.Parameter(torch.randn(latent_slots, d_model) * 0.02)
        # A fixed learned query cannot perform content-dependent lookup. The
        # final causal context state is a prompt/memory summary; project it into
        # every slot before compare/select so the latent queries can represent
        # the current problem state while retaining their distinct slot biases.
        self.context_query = nn.Sequential(
            RMSNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model, bias=False),
        )
        self.router = ReasoningModeRouter(d_model, n_reasoning_modes)

        # context -> slots
        self.cross = LowRankCompareSelectAggregate(d_model, compare_rank)
        self.cross_norm = RMSNorm(d_model)

        # full slot<->slot interaction: no causal ordering inside a thought block
        self.slot_mix = LowRankCompareSelectAggregate(d_model, compare_rank)
        self.slot_norm = RMSNorm(d_model)

        self.ff = ExpandCompressFFN(d_model, planner_ff)
        self.reinject = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, d_model))

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        b = context.size(0)
        if context_mask is None:
            context_mask = torch.ones(
                context.shape[:2], dtype=torch.bool, device=context.device
            )
        else:
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)
        if context_mask.shape != context.shape[:2]:
            raise ValueError("context_mask must have shape [batch, sequence]")
        if not torch.all(context_mask.any(dim=-1)):
            raise ValueError("each context row must contain a valid state")
        # Mean pooling exposes lexical content directly; the final causal state
        # carries order-aware/global context. Their combination is a compact,
        # fully continuous interface from prompt/memory into the workspace.
        mask_f = context_mask.unsqueeze(-1).to(context.dtype)
        mean_context = (context * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1)
        positions = torch.arange(context.size(1), device=context.device).unsqueeze(0)
        last_index = positions.masked_fill(~context_mask, -1).max(dim=1).values
        last_context = context[
            torch.arange(b, device=context.device), last_index
        ]
        raw_summary = 0.5 * (mean_context + last_context)
        pooled = raw_summary + self.context_query(
            torch.cat([mean_context, last_context], dim=-1)
        )
        conditioned = pooled.unsqueeze(1)
        q = self.query.unsqueeze(0).expand(b, -1, -1) + conditioned
        q, route_probs = self.router(q, pooled)

        cross, cross_w = self.cross(
            self.cross_norm(q), context, context_mask=context_mask
        )
        z = q + cross

        # Bidirectional latent workspace. Because this module is not causal,
        # all K latent thoughts are formed as a set and can interact.
        mix, slot_w = self.slot_mix(self.slot_norm(z), self.slot_norm(z))
        z = z + mix
        z = self.ff(z)
        z = self.reinject(z)

        stats = {
            "cross_attention": cross_w,
            "slot_attention": slot_w,
            "route_probs": route_probs,
        }
        return z, stats


class QueryConditionedLatentPlanner(nn.Module):
    """Binding-preserving workspace with an explicit query anchor.

    The legacy planner asks learned generic slots to infer both *what is being
    asked* and *which context item answers it*.  Here those jobs are separated:
    the final valid prompt state is retained as ``query_state`` and every slot
    compares that state (plus a role offset) against context.  The primary slot
    bypasses routing and all slots retain an identity path around workspace
    mixing, preventing a selected value from being obligatorily pooled away.
    """

    def __init__(
        self,
        d_model: int,
        latent_slots: int,
        compare_rank: int,
        planner_ff: int,
        n_reasoning_modes: int,
        n_binding_relations: int,
    ):
        super().__init__()
        self.latent_slots = latent_slots
        self.slot_role = nn.Parameter(torch.randn(latent_slots, d_model) * 0.02)
        if n_binding_relations < 1:
            raise ValueError("query-conditioned planner needs binding relations")
        self.binding_role = nn.Parameter(
            torch.randn(n_binding_relations, d_model) * 0.02
        )
        self.query_norm = RMSNorm(d_model)
        self.context_norm = RMSNorm(d_model)
        self.router = ReasoningModeRouter(d_model, n_reasoning_modes)
        self.bind = AddressedPayloadBinding(d_model, compare_rank)
        self.bind_norm = RMSNorm(d_model)
        self.slot_mix = LowRankCompareSelectAggregate(d_model, compare_rank)
        self.slot_norm = RMSNorm(d_model)
        self.ff = ExpandCompressFFN(d_model, planner_ff)
        # Start close to the exact selected state.  Mixing can grow only when
        # the task loss demonstrates that it is useful.
        self.workspace_logit = nn.Parameter(torch.full((latent_slots, 1), -2.0))
        self.reinject = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, d_model))

    def forward(
        self,
        context: torch.Tensor,
        query_state: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        binding_mask: torch.Tensor | None = None,
        payload_context: torch.Tensor | None = None,
        binding_context: torch.Tensor | None = None,
        binding_payload: torch.Tensor | None = None,
        binding_query_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if query_state.ndim != 2 or query_state.shape != (
            context.size(0),
            context.size(2),
        ):
            raise ValueError("query_state must have shape [batch, d_model]")
        if payload_context is None:
            payload_context = context
        elif payload_context.shape != context.shape:
            raise ValueError("payload_context must match context shape")
        if binding_context is None:
            binding_context = context
        if binding_payload is None:
            binding_payload = payload_context
        if binding_context.shape != binding_payload.shape:
            raise ValueError("binding address/payload contexts must match")
        if binding_context.size(0) != context.size(0) or binding_context.size(2) != context.size(2):
            raise ValueError("binding context must match batch and model width")
        if binding_query_state is None:
            binding_query_state = query_state
        if binding_query_state.shape != query_state.shape:
            raise ValueError("binding_query_state must match query_state shape")
        if context_mask is None:
            context_mask = torch.ones(
                context.shape[:2], dtype=torch.bool, device=context.device
            )
        else:
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)
        if context_mask.shape != context.shape[:2]:
            raise ValueError("context_mask must have shape [batch, sequence]")
        if not torch.all(context_mask.any(dim=-1)):
            raise ValueError("each context row must contain a selectable state")
        if binding_mask is None:
            binding_mask = torch.ones(
                binding_context.shape[:2], dtype=torch.bool, device=context.device
            )
        else:
            binding_mask = binding_mask.to(device=context.device, dtype=torch.bool)
            if binding_mask.shape != binding_context.shape[:2]:
                raise ValueError("binding_mask must match binding candidates")
            if not torch.all(binding_mask.any(dim=-1)):
                raise ValueError("each binding_mask row must select at least one state")

        batch = context.size(0)
        anchor = self.query_norm(query_state)
        base = anchor[:, None, :] + self.slot_role[None, :, :]
        routed, route_probs = self.router(base, anchor)
        # Slot zero is the protected binding carrier: its compare query remains
        # the exact query anchor plus its role, never a route mixture.
        primary_query = (
            self.query_norm(binding_query_state)[:, None, :]
        )
        if self.latent_slots == 1:
            queries = primary_query
        else:
            queries = torch.cat([primary_query, routed[:, 1:]], dim=1)
        primary_selected, primary_weights = self.bind(
            queries[:, :1],
            self.context_norm(binding_context),
            binding_payload,
            context_mask=binding_mask,
        )
        if self.latent_slots == 1:
            selected = primary_selected
            binding_weights = primary_weights
        else:
            auxiliary_selected, auxiliary_weights = self.bind(
                queries[:, 1:],
                self.context_norm(context),
                payload_context,
                context_mask=context_mask,
            )
            selected = torch.cat([primary_selected, auxiliary_selected], dim=1)
            # The protected binding carrier may compare several address→payload
            # relations (and therefore have more candidates) than the generic
            # workspace slots.  Pad diagnostics only; this never changes the
            # selected payloads or their gradients.
            candidates = max(primary_weights.size(-1), auxiliary_weights.size(-1))
            primary_weights = F.pad(
                primary_weights, (0, candidates - primary_weights.size(-1))
            )
            auxiliary_weights = F.pad(
                auxiliary_weights, (0, candidates - auxiliary_weights.size(-1))
            )
            binding_weights = torch.cat([primary_weights, auxiliary_weights], dim=1)
        # The primary carrier is payload-only.  The query is an address, not a
        # shortcut into the answer: it determines attention weights but is not
        # added to the state that reaches the vocabulary head.  Auxiliary
        # slots retain the richer query residual for general workspace use.
        primary_bound = self.bind_norm(primary_selected)
        if self.latent_slots == 1:
            bound = primary_bound
        else:
            auxiliary_bound = self.bind_norm(
                queries[:, 1:] + selected[:, 1:]
            )
            bound = torch.cat([primary_bound, auxiliary_bound], dim=1)

        mixed, slot_weights = self.slot_mix(
            self.slot_norm(bound), self.slot_norm(bound)
        )
        workspace = self.ff(bound + mixed)
        alpha = torch.sigmoid(self.workspace_logit)[None, :, :]
        latent = bound + alpha * (workspace - bound)
        # Slot zero is the exact binding carrier.  Parallel workspace
        # interaction remains available to slots 1..K-1, but cannot overwrite
        # the primary query-selected payload before lexical readout.
        if self.latent_slots == 1:
            latent = bound
        else:
            latent = torch.cat([bound[:, :1], latent[:, 1:]], dim=1)
            refined_tail = latent[:, 1:] + 0.1 * self.reinject(latent[:, 1:])
            latent = torch.cat([latent[:, :1], refined_tail], dim=1)
        return latent, {
            "binding_attention": binding_weights,
            "slot_attention": slot_weights,
            "route_probs": route_probs,
            "query_state": query_state,
        }


class DirectLatentReadout(nn.Module):
    """Direction-preserving final state; no fresh BOS retrieval hop."""

    def __init__(self, d_model: int):
        super().__init__()
        self.scale = math.sqrt(d_model)

    def forward(self, primary_latent: torch.Tensor) -> torch.Tensor:
        if primary_latent.ndim != 2:
            raise ValueError("primary_latent must have shape [batch, d_model]")
        return F.normalize(primary_latent.float(), dim=-1).to(primary_latent.dtype) * self.scale


class CosineTiedHead(nn.Module):
    """Vocabulary projection whose own code is always decoded exactly.

    The weight is tied to the token embedding by :class:`FOGLatentReasoner`.
    Normalizing both sides removes the row-norm bias that made only ~16% of
    pretrained embeddings self-decode under a raw dot-product LM head.
    """

    def __init__(self, vocab_size: int, d_model: int, scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        self.scale = float(scale)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        normalized_state = F.normalize(state.float(), dim=-1)
        normalized_codebook = F.normalize(self.weight.float(), dim=-1)
        return self.scale * F.linear(normalized_state, normalized_codebook)
