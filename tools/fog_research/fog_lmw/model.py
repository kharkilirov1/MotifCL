from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import FOGReasonerConfig
from .core import TinyDecoderBackbone
from .losses import route_entropy, slot_diversity_loss
from .memory import PersistentLatentMemory, ReusableLatentMemory
from .registers import RegisterMachineCell, RegisterStateMemory
from .planner import (
    DirectLatentReadout,
    CosineTiedHead,
    LatentPlanner,
    QueryConditionedLatentPlanner,
)


class FOGLatentReasoner(nn.Module):
    """
    Finite-Operator-Grammar-inspired latent multi-workspace reasoner.

    Reasoning path:
      token embeddings
        -> recurrent backbone state
        -> parallel latent planner
        -> persistent/compressed memory
        -> repeat R times
        -> lexical decoding only at the end
    """

    def __init__(self, cfg: FOGReasonerConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.backbone = TinyDecoderBackbone(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        )
        planner_class = (
            QueryConditionedLatentPlanner
            if cfg.binding_mode == "query_conditioned"
            else LatentPlanner
        )
        self.planner = planner_class(
            d_model=cfg.d_model,
            latent_slots=cfg.latent_slots,
            compare_rank=cfg.compare_rank,
            planner_ff=cfg.planner_ff,
            n_reasoning_modes=cfg.n_reasoning_modes,
            **(
                {"n_binding_relations": len(cfg.binding_offsets)}
                if planner_class is QueryConditionedLatentPlanner
                else {}
            ),
        )
        if cfg.architecture_version == "register_machine_v3":
            self.memory = RegisterStateMemory(cfg.latent_slots)
            self.machine_cell = RegisterMachineCell(
                d_model=cfg.d_model,
                latent_slots=cfg.latent_slots,
                compare_rank=cfg.compare_rank,
                machine_ff=cfg.machine_ff,
                n_generated_operators=cfg.machine_operator_count,
                operator_rank=cfg.machine_operator_rank,
                hard_routing=cfg.machine_hard_routing,
            )
        elif cfg.architecture_version == "query_bound_v2":
            self.memory = ReusableLatentMemory(cfg.d_model, cfg.latent_slots)
        else:
            self.memory = PersistentLatentMemory(
                cfg.d_model, cfg.compare_rank, cfg.memory_slots
            )
        if cfg.readout_mode == "direct_latent":
            self.direct_readout = DirectLatentReadout(cfg.d_model)
            self.direct_head = CosineTiedHead(cfg.vocab_size, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token.weight
        if cfg.readout_mode == "direct_latent":
            self.direct_head.weight = self.token.weight
            if cfg.architecture_version == "register_machine_v3":
                # Start from the already-validated v2 behavior while keeping
                # generated computation available from tick one.
                with torch.no_grad():
                    final_router = self.machine_cell.operator_bank.router[-1]
                    final_router.bias.zero_()
                    final_router.bias[0] = 4.0  # READ candidate
            # Query and key begin in one address coordinate system.  They
            # remain separate trainable projections and may specialize later.
            with torch.no_grad():
                self.planner.bind.k_proj.weight.copy_(
                    self.planner.bind.q_proj.weight
                )
                self.planner.binding_role.zero_()

    def direct_vocab_logits(self, primary_latent: torch.Tensor) -> torch.Tensor:
        if self.cfg.readout_mode != "direct_latent":
            raise ValueError("direct vocabulary projection requires direct_latent mode")
        return self.direct_head(self.direct_readout(primary_latent))

    def _init_weights(self, module: nn.Module) -> None:
        """Small-model initialization; avoids unit-variance tied-embedding logits."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                nn.init.normal_(
                    module.in_proj_weight,
                    mean=0.0,
                    std=self.cfg.initializer_range,
                )
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)

    def _run_backbone(
        self,
        token_embeds: torch.Tensor,
        memory: torch.Tensor | None,
        token_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = token_embeds.shape
        if token_attention_mask is None:
            token_attention_mask = torch.ones(
                b, n, dtype=torch.bool, device=token_embeds.device
            )
        else:
            token_attention_mask = token_attention_mask.to(
                device=token_embeds.device, dtype=torch.bool
            )
            if token_attention_mask.shape != (b, n):
                raise ValueError(
                    "prompt_attention_mask must have shape [batch, prompt_length]"
                )
        if memory is None:
            x = token_embeds
            kinds = torch.zeros(b, n, dtype=torch.long, device=x.device)
            combined_mask = token_attention_mask
        else:
            x = torch.cat([token_embeds, memory], dim=1)
            kinds = torch.cat(
                [
                    torch.zeros(b, n, dtype=torch.long, device=x.device),
                    torch.ones(b, memory.size(1), dtype=torch.long, device=x.device),
                ],
                dim=1,
            )
            combined_mask = torch.cat(
                [
                    token_attention_mask,
                    torch.ones(
                        b,
                        memory.size(1),
                        dtype=torch.bool,
                        device=x.device,
                    ),
                ],
                dim=1,
            )
        return (
            self.backbone.forward_embeds(
                x, kinds, attention_mask=combined_mask
            ),
            combined_mask,
        )

    def _memory_size_after_steps(self, reasoning_steps: int) -> int:
        if self.cfg.architecture_version in {"query_bound_v2", "register_machine_v3"}:
            return self.cfg.latent_slots if reasoning_steps > 0 else 0
        size = reasoning_steps * self.cfg.latent_slots
        return min(size, self.cfg.memory_slots) if self.cfg.memory_slots > 0 else size

    def _check_decode_length(
        self, prompt_len: int, decoder_len: int, reasoning_steps: int
    ) -> None:
        total = prompt_len + self._memory_size_after_steps(reasoning_steps) + decoder_len
        if total > self.cfg.max_seq_len:
            raise ValueError(
                "prompt + latent memory + decoder prefix exceeds max_seq_len: "
                f"{prompt_len} + {self._memory_size_after_steps(reasoning_steps)} + "
                f"{decoder_len} = {total} > {self.cfg.max_seq_len}"
            )

    def _prepare_reasoning_prompt(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed a fixed lexical prompt and locate each example's query token."""
        x = self.token(prompt_ids)
        batch_size, prompt_length = prompt_ids.shape
        if prompt_attention_mask is None:
            lexical_mask = torch.ones(
                batch_size,
                prompt_length,
                dtype=torch.bool,
                device=prompt_ids.device,
            )
        else:
            lexical_mask = prompt_attention_mask.to(
                device=prompt_ids.device, dtype=torch.bool
            )
            if lexical_mask.shape != prompt_ids.shape:
                raise ValueError("prompt_attention_mask shape must match prompt_ids")
        if not torch.all(lexical_mask.any(dim=-1)):
            raise ValueError("each prompt must contain at least one valid token")
        positions = torch.arange(prompt_length, device=prompt_ids.device).unsqueeze(0)
        query_indices = positions.masked_fill(~lexical_mask, -1).max(dim=1).values
        batch_indices = torch.arange(batch_size, device=prompt_ids.device)
        return x, lexical_mask, query_indices, batch_indices

    def _reasoning_step_embedded(
        self,
        x: torch.Tensor,
        lexical_mask: torch.Tensor,
        query_indices: torch.Tensor,
        batch_indices: torch.Tensor,
        memory: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict, dict, torch.Tensor | None, torch.Tensor | None]:
        """Apply exactly one shared latent transition to an existing memory state.

        This explicit one-step boundary is intentionally public-facing through
        :meth:`transition_memory`: it makes recurrent FOG dynamics directly
        probeable with JVP/VJP sketches without reconstructing a full Jacobian.
        """
        prompt_length = x.size(1)
        h, context_mask = self._run_backbone(
            x, memory, token_attention_mask=lexical_mask
        )
        query_state: torch.Tensor | None = None
        binding_query_state: torch.Tensor | None = None

        if self.cfg.binding_mode == "query_conditioned":
            query_state = h[batch_indices, query_indices]
            raw_query_state = x[batch_indices, query_indices]
            if (
                self.cfg.binding_query_update == "primary_recurrent"
                and memory is not None
            ):
                if memory.size(1) < 1:
                    raise ValueError("recurrent binding query requires a primary memory slot")
                # ReusableLatentMemory preserves slot zero exactly, so the
                # protected carrier itself is the recurrent address register.
                binding_query_state = memory[:, 0]
            else:
                binding_query_state = raw_query_state

            payload_context = x if memory is None else torch.cat([x, memory], dim=1)
            workspace_selectable = context_mask.clone()
            workspace_selectable[batch_indices, query_indices] = False
            binding_addresses = []
            binding_payloads = []
            binding_masks = []
            for relation_index, offset in enumerate(self.cfg.binding_offsets):
                if offset >= prompt_length:
                    continue
                role = self.planner.binding_role[relation_index]
                binding_addresses.append(x[:, :-offset] + role)
                binding_payloads.append(x[:, offset:])
                binding_masks.append(
                    lexical_mask[:, :-offset] & lexical_mask[:, offset:]
                )
            if not binding_addresses:
                raise ValueError("prompt is too short for configured binding offsets")
            binding_context = torch.cat(binding_addresses, dim=1)
            binding_payload = torch.cat(binding_payloads, dim=1)
            binding_selectable = torch.cat(binding_masks, dim=1)
            if not torch.all(binding_selectable.any(dim=-1)):
                raise ValueError(
                    "query-conditioned binding requires context besides the query token"
                )
            z, pstats = self.planner(
                h,
                query_state,
                context_mask=workspace_selectable,
                binding_mask=binding_selectable,
                payload_context=payload_context,
                binding_context=binding_context,
                binding_payload=binding_payload,
                binding_query_state=binding_query_state,
            )
            if self.cfg.architecture_version == "register_machine_v3":
                z, machine_stats = self.machine_cell(
                    memory,
                    z,
                    initial_value=(binding_query_state if memory is None else None),
                    initial_control=(query_state if memory is None else None),
                )
                pstats = {**pstats, "machine": machine_stats}
                new_memory, mstats = self.memory.forward_preserving(
                    memory,
                    z,
                    protected_slots=self.cfg.latent_slots,
                )
            else:
                new_memory, mstats = self.memory.forward_preserving(
                    memory,
                    z,
                    protected_slots=self.cfg.protected_binding_slots,
                )
        else:
            z, pstats = self.planner(h, context_mask=context_mask)
            new_memory, mstats = self.memory(memory, z)

        return new_memory, z, pstats, mstats, query_state, binding_query_state

    def transition_memory(
        self,
        prompt_ids: torch.Tensor,
        memory: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Run one recurrent latent step from an explicit memory state.

        ``memory`` is the complete recurrent machine state. For query-bound v2
        it has fixed shape ``[B, K, D]``; when ``binding_query_update`` is
        ``primary_recurrent``, slot zero is used as the next binding address.
        This method performs no vocabulary projection and is differentiable
        with respect to ``memory`` for JVP/VJP instrumentation.
        """
        x, lexical_mask, query_indices, batch_indices = self._prepare_reasoning_prompt(
            prompt_ids, prompt_attention_mask
        )
        if memory is not None:
            if memory.ndim != 3 or memory.size(0) != prompt_ids.size(0) or memory.size(2) != self.cfg.d_model:
                raise ValueError("memory must have shape [batch, slots, d_model]")
            if (
                self.cfg.architecture_version == "query_bound_v2"
                and memory.size(1) != self.cfg.latent_slots
            ):
                raise ValueError("query_bound_v2 memory must contain exactly latent_slots states")
        new_memory, z, pstats, mstats, query_state, binding_query_state = (
            self._reasoning_step_embedded(
                x, lexical_mask, query_indices, batch_indices, memory
            )
        )
        aux = {
            "latent": z,
            "primary_latent": z[:, 0] if self.cfg.binding_mode == "query_conditioned" else None,
            "query_state": query_state,
            "binding_query_state": binding_query_state,
            "planner": pstats if return_diagnostics else None,
            "memory": mstats if return_diagnostics else None,
        }
        return new_memory, aux

    def reason(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        reasoning_steps: int | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor | None, dict]:
        """Build latent memory without emitting any vocabulary tokens."""
        steps = self.cfg.reasoning_steps if reasoning_steps is None else reasoning_steps
        if steps < 0:
            raise ValueError("reasoning_steps must be >= 0")
        if self.cfg.readout_mode == "direct_latent" and steps < 1:
            raise ValueError("direct_latent readout requires at least one reasoning step")
        x, lexical_mask, query_indices, batch_indices = self._prepare_reasoning_prompt(
            prompt_ids, prompt_attention_mask
        )
        memory = None
        latest_primary: torch.Tensor | None = None
        latest_query: torch.Tensor | None = None
        history: list[dict] = []
        diversity_terms = []
        entropy_terms = []
        halt_probabilities = []
        need_diversity = self.cfg.diversity_weight != 0.0
        need_entropy = self.cfg.route_entropy_weight != 0.0

        for t in range(steps):
            memory, z, pstats, mstats, query_state, binding_query_state = (
                self._reasoning_step_embedded(
                    x, lexical_mask, query_indices, batch_indices, memory
                )
            )
            if self.cfg.binding_mode == "query_conditioned":
                latest_primary = z[:, 0]
                latest_query = query_state

            if self.cfg.architecture_version == "register_machine_v3":
                halt_probabilities.append(pstats["machine"]["halt_probability"])
            if need_diversity:
                diversity_terms.append(slot_diversity_loss(z))
            if need_entropy:
                ent = route_entropy(pstats["route_probs"])
                if ent is not None:
                    entropy_terms.append(ent)

            if return_diagnostics:
                history.append(
                    {
                        "step": t,
                        "latent": z,
                        "memory_size": memory.size(1),
                        "planner": pstats,
                        "memory": mstats,
                        "binding_query_state": binding_query_state,
                    }
                )

        device = prompt_ids.device
        aux = {
            "history": history,
            "diversity_loss": (
                torch.stack(diversity_terms).mean()
                if diversity_terms
                else torch.tensor(0.0, device=device)
            ),
            "route_entropy": (
                torch.stack(entropy_terms).mean()
                if entropy_terms
                else torch.tensor(0.0, device=device)
            ),
            "primary_latent": latest_primary,
            "query_state": latest_query,
            "binding_query_update": self.cfg.binding_query_update,
            "halt_probabilities": (
                torch.stack(halt_probabilities, dim=1) if halt_probabilities else None
            ),
        }
        return memory, aux

    def reason_adaptive(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        *,
        max_steps: int | None = None,
        min_steps: int | None = None,
        halt_threshold: float | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Inference-only adaptive latent execution for register_machine_v3.

        The external loop is only a safety cap. Each example freezes its entire
        register file once the learned HALT probability crosses the threshold.
        """
        if self.cfg.architecture_version != "register_machine_v3":
            raise ValueError("reason_adaptive requires register_machine_v3")
        max_steps = self.cfg.reasoning_steps if max_steps is None else max_steps
        min_steps = self.cfg.machine_min_steps if min_steps is None else min_steps
        threshold = (
            self.cfg.machine_halt_threshold
            if halt_threshold is None
            else halt_threshold
        )
        if max_steps < 1 or min_steps < 1 or min_steps > max_steps:
            raise ValueError("adaptive step bounds are invalid")
        if not 0.0 < threshold < 1.0:
            raise ValueError("halt_threshold must be in (0,1)")

        x, lexical_mask, query_indices, batch_indices = self._prepare_reasoning_prompt(
            prompt_ids, prompt_attention_mask
        )
        batch = prompt_ids.size(0)
        memory = None
        halted = torch.zeros(batch, dtype=torch.bool, device=prompt_ids.device)
        steps_used = torch.full(
            (batch,), max_steps, dtype=torch.long, device=prompt_ids.device
        )
        halt_history = []
        history = []
        for t in range(max_steps):
            candidate, z, pstats, mstats, query_state, binding_query_state = (
                self._reasoning_step_embedded(
                    x, lexical_mask, query_indices, batch_indices, memory
                )
            )
            if memory is not None and torch.any(halted):
                candidate = torch.where(halted[:, None, None], memory, candidate)
            halt_prob = pstats["machine"]["halt_probability"]
            halt_history.append(halt_prob)
            if return_diagnostics:
                history.append(
                    {
                        "step": t,
                        "latent": z,
                        "memory": mstats,
                        "planner": pstats,
                        "binding_query_state": binding_query_state,
                    }
                )
            memory = candidate
            if t + 1 >= min_steps:
                newly = (~halted) & halt_prob.ge(threshold)
                steps_used = torch.where(
                    newly, torch.full_like(steps_used, t + 1), steps_used
                )
                halted = halted | newly
                if bool(torch.all(halted)):
                    break
        if memory is None:
            raise AssertionError("adaptive machine executed zero steps")
        return memory, {
            "primary_latent": memory[:, 0],
            "halt_probabilities": torch.stack(halt_history, dim=1),
            "steps_used": steps_used,
            "halted": halted,
            "history": history,
        }

    def _decode_direct_continuation(
        self,
        prompt_ids: torch.Tensor,
        memory: torch.Tensor | None,
        readout_state: torch.Tensor,
        previous_answer_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        previous_answer_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict tokens 2+ from real previous answer tokens.

        The first answer token is predicted directly from ``readout_state``.
        Subsequent tokens continue a single positional timeline containing the
        lexical context, latent memory, the contentful readout state, and actual
        previous output tokens.  There is no learned blank/BOS retrieval state.
        """

        if previous_answer_ids.size(1) < 1:
            raise ValueError("continuation requires at least one previous answer token")
        lexical = self.token(prompt_ids)
        previous = self.token(previous_answer_ids)
        batch = lexical.size(0)
        if readout_state.shape != (batch, self.cfg.d_model):
            raise ValueError("readout_state must have shape [batch, d_model]")
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones(
                batch, lexical.size(1), dtype=torch.bool, device=lexical.device
            )
        else:
            prompt_attention_mask = prompt_attention_mask.to(
                device=lexical.device, dtype=torch.bool
            )
        if previous_answer_mask is None:
            previous_answer_mask = torch.ones(
                batch, previous.size(1), dtype=torch.bool, device=lexical.device
            )
        else:
            previous_answer_mask = previous_answer_mask.to(
                device=lexical.device, dtype=torch.bool
            )
        if prompt_attention_mask.shape != prompt_ids.shape:
            raise ValueError("prompt_attention_mask shape must match prompt_ids")
        if previous_answer_mask.shape != previous_answer_ids.shape:
            raise ValueError("previous_answer_mask shape must match previous_answer_ids")

        parts = [lexical]
        kinds = [torch.zeros(batch, lexical.size(1), dtype=torch.long, device=lexical.device)]
        masks = [prompt_attention_mask]
        if memory is not None:
            parts.append(memory)
            kinds.append(torch.ones(batch, memory.size(1), dtype=torch.long, device=lexical.device))
            masks.append(torch.ones(batch, memory.size(1), dtype=torch.bool, device=lexical.device))
        parts.extend([readout_state[:, None, :], previous])
        kinds.extend(
            [
                torch.ones(batch, 1, dtype=torch.long, device=lexical.device),
                torch.zeros(batch, previous.size(1), dtype=torch.long, device=lexical.device),
            ]
        )
        masks.extend(
            [
                torch.ones(batch, 1, dtype=torch.bool, device=lexical.device),
                previous_answer_mask,
            ]
        )
        joined = torch.cat(parts, dim=1)
        kind_ids = torch.cat(kinds, dim=1)
        combined_mask = torch.cat(masks, dim=1)
        hidden = self.backbone.forward_embeds(
            joined,
            kind_ids,
            attention_mask=combined_mask,
        )
        return self.lm_head(hidden[:, -previous.size(1) :])

    def decode(
        self,
        prompt_ids: torch.Tensor,
        memory: torch.Tensor | None,
        decoder_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        decoder_ids are lexical answer-prefix tokens (e.g. BOS, then generated tokens).
        Memory is inserted BETWEEN prompt and answer so answer tokens can attend to it.
        """
        p = self.token(prompt_ids)
        a = self.token(decoder_ids)
        b = p.size(0)
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones(
                b, p.size(1), dtype=torch.bool, device=p.device
            )
        else:
            prompt_attention_mask = prompt_attention_mask.to(
                device=p.device, dtype=torch.bool
            )
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones(
                b, a.size(1), dtype=torch.bool, device=p.device
            )
        else:
            decoder_attention_mask = decoder_attention_mask.to(
                device=p.device, dtype=torch.bool
            )
        if prompt_attention_mask.shape != prompt_ids.shape:
            raise ValueError("prompt_attention_mask shape must match prompt_ids")
        if decoder_attention_mask.shape != decoder_ids.shape:
            raise ValueError("decoder_attention_mask shape must match decoder_ids")

        parts = [p]
        kinds = [torch.zeros(b, p.size(1), dtype=torch.long, device=p.device)]
        masks = [prompt_attention_mask]
        if memory is not None:
            parts.append(memory)
            kinds.append(torch.ones(b, memory.size(1), dtype=torch.long, device=p.device))
            masks.append(
                torch.ones(
                    b, memory.size(1), dtype=torch.bool, device=p.device
                )
            )
        parts.append(a)
        kinds.append(torch.zeros(b, a.size(1), dtype=torch.long, device=p.device))
        masks.append(decoder_attention_mask)

        x = torch.cat(parts, dim=1)
        kind = torch.cat(kinds, dim=1)
        combined_mask = torch.cat(masks, dim=1)
        h = self.backbone.forward_embeds(
            x, kind, attention_mask=combined_mask
        )
        answer_start = p.size(1) + (0 if memory is None else memory.size(1))
        # Avoid materializing vocabulary logits for prompt and latent positions.
        return self.lm_head(h[:, answer_start:, :])

    def loss(
        self,
        prompt_ids: torch.Tensor,
        answer_ids_with_bos: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        answer_attention_mask: torch.Tensor | None = None,
        decoder_prompt_ids: torch.Tensor | None = None,
        decoder_prompt_attention_mask: torch.Tensor | None = None,
        reasoning_steps: int | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Teacher forcing. answer_ids_with_bos = [BOS, y1, y2, ...].
        BOS predicts y1, y1 predicts y2, etc.
        """
        if answer_ids_with_bos.size(1) < 2:
            raise ValueError("answer_ids_with_bos must contain BOS + >=1 target token")
        steps = self.cfg.reasoning_steps if reasoning_steps is None else reasoning_steps
        lexical_prompt = prompt_ids if decoder_prompt_ids is None else decoder_prompt_ids
        lexical_prompt_mask = (
            prompt_attention_mask
            if decoder_prompt_ids is None
            else decoder_prompt_attention_mask
        )
        if lexical_prompt.size(0) != prompt_ids.size(0):
            raise ValueError("decoder_prompt_ids batch size must match prompt_ids")
        if answer_attention_mask is None:
            answer_attention_mask = torch.ones_like(
                answer_ids_with_bos, dtype=torch.bool
            )
        else:
            answer_attention_mask = answer_attention_mask.to(
                device=answer_ids_with_bos.device, dtype=torch.bool
            )
            if answer_attention_mask.shape != answer_ids_with_bos.shape:
                raise ValueError(
                    "answer_attention_mask shape must match answer_ids_with_bos"
                )
        self._check_decode_length(
            lexical_prompt.size(1), answer_ids_with_bos.size(1) - 1, steps
        )
        memory, aux = self.reason(
            prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            reasoning_steps=steps,
            return_diagnostics=return_diagnostics,
        )
        target = answer_ids_with_bos[:, 1:].masked_fill(
            ~answer_attention_mask[:, 1:], -100
        )
        if not torch.any(target.ne(-100)):
            raise ValueError("batch must contain at least one answer target token")
        if self.cfg.readout_mode == "direct_latent":
            primary = aux.get("primary_latent")
            if primary is None:
                raise AssertionError("query-bound reasoning did not return a primary latent")
            readout_state = self.direct_readout(primary)
            first_logits = self.direct_head(readout_state)[:, None, :]
            target_count = answer_ids_with_bos.size(1) - 1
            if target_count == 1:
                logits = first_logits
            else:
                previous_ids = answer_ids_with_bos[:, 1:-1]
                previous_mask = answer_attention_mask[:, 1:-1]
                continuation = self._decode_direct_continuation(
                    lexical_prompt,
                    memory,
                    readout_state,
                    previous_ids,
                    prompt_attention_mask=lexical_prompt_mask,
                    previous_answer_mask=previous_mask,
                )
                logits = torch.cat([first_logits, continuation], dim=1)
            aux["readout_state"] = readout_state
        else:
            decoder_in = answer_ids_with_bos[:, :-1]
            decoder_mask = answer_attention_mask[:, :-1]
            logits = self.decode(
                lexical_prompt,
                memory,
                decoder_in,
                prompt_attention_mask=lexical_prompt_mask,
                decoder_attention_mask=decoder_mask,
            )
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), target.reshape(-1)
        )
        active = target.ne(-100)
        token_accuracy = (
            logits.argmax(dim=-1).eq(target).masked_select(active).float().mean()
        )
        total = ce + self.cfg.diversity_weight * aux["diversity_loss"]
        # Positive coefficient here would MINIMIZE entropy. Keep zero by default;
        # users can instead negate it to encourage broad routing.
        total = total + self.cfg.route_entropy_weight * aux["route_entropy"]
        aux = {
            **aux,
            "ce_loss": ce,
            "loss": total,
            "token_accuracy": token_accuracy,
            "target_tokens": active.sum(),
        }
        return total, aux

    def causal_lm_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backbone-only lexical pretraining path (planner/memory are inactive)."""
        x = self.token(input_ids)
        kinds = torch.zeros_like(input_ids)
        h = self.backbone.forward_embeds(
            x, kinds, attention_mask=attention_mask
        )
        return self.lm_head(h)

    def causal_lm_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Next-token loss used to pretrain the shared lexical backbone."""
        if input_ids.size(1) < 2:
            raise ValueError("causal LM sequences must contain at least two tokens")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(
                device=input_ids.device, dtype=torch.bool
            )
        if labels is None:
            labels = input_ids.masked_fill(~attention_mask, -100)
        elif labels.shape != input_ids.shape:
            raise ValueError("labels shape must match input_ids")
        logits = self.causal_lm_logits(input_ids, attention_mask=attention_mask)
        shifted_logits = logits[:, :-1].float()
        shifted_labels = labels[:, 1:].clone()
        shifted_labels = shifted_labels.masked_fill(
            ~attention_mask[:, 1:], -100
        )
        if not torch.any(shifted_labels.ne(-100)):
            raise ValueError("batch must contain at least one next-token target")
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_labels.reshape(-1),
        )
        predicted = shifted_logits.argmax(dim=-1)
        active = shifted_labels.ne(-100)
        accuracy = predicted.eq(shifted_labels).masked_select(active).float().mean()
        return loss, {
            "loss": loss,
            "ce_loss": loss,
            "token_accuracy": accuracy,
            "target_tokens": active.sum(),
        }

    def forward(
        self,
        prompt_ids: torch.Tensor,
        answer_ids_with_bos: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        answer_attention_mask: torch.Tensor | None = None,
        decoder_prompt_ids: torch.Tensor | None = None,
        decoder_prompt_attention_mask: torch.Tensor | None = None,
        reasoning_steps: int | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Standard ``nn.Module`` entry point; equivalent to :meth:`loss`."""
        return self.loss(
            prompt_ids,
            answer_ids_with_bos,
            prompt_attention_mask=prompt_attention_mask,
            answer_attention_mask=answer_attention_mask,
            decoder_prompt_ids=decoder_prompt_ids,
            decoder_prompt_attention_mask=decoder_prompt_attention_mask,
            reasoning_steps=reasoning_steps,
            return_diagnostics=return_diagnostics,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int | None = None,
        max_new_tokens: int = 32,
        prompt_attention_mask: torch.Tensor | None = None,
        reasoning_steps: int | None = None,
        decoder_prompt_ids: torch.Tensor | None = None,
        decoder_prompt_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        steps = self.cfg.reasoning_steps if reasoning_steps is None else reasoning_steps
        lexical_prompt = prompt_ids if decoder_prompt_ids is None else decoder_prompt_ids
        lexical_mask = (
            prompt_attention_mask
            if decoder_prompt_ids is None
            else decoder_prompt_attention_mask
        )
        self._check_decode_length(lexical_prompt.size(1), max_new_tokens, steps)
        was_training = self.training
        self.eval()
        try:
            memory, aux = self.reason(
                prompt_ids,
                prompt_attention_mask=prompt_attention_mask,
                reasoning_steps=steps,
            )
            out = torch.full(
                (prompt_ids.size(0), 1), bos_token_id,
                dtype=torch.long,
                device=prompt_ids.device,
            )
            finished = torch.zeros(
                prompt_ids.size(0), dtype=torch.bool, device=prompt_ids.device
            )
            readout_state = None
            if self.cfg.readout_mode == "direct_latent":
                primary = aux.get("primary_latent")
                if primary is None:
                    raise AssertionError("query-bound reasoning did not return a primary latent")
                readout_state = self.direct_readout(primary)
                aux["readout_state"] = readout_state
            for token_index in range(max_new_tokens):
                if self.cfg.readout_mode == "direct_latent":
                    if token_index == 0:
                        next_logits = self.direct_head(readout_state)
                    else:
                        continuation = self._decode_direct_continuation(
                            lexical_prompt,
                            memory,
                            readout_state,
                            out[:, 1:],
                            prompt_attention_mask=lexical_mask,
                        )
                        next_logits = continuation[:, -1]
                else:
                    logits = self.decode(
                        lexical_prompt,
                        memory,
                        out,
                        prompt_attention_mask=lexical_mask,
                    )
                    next_logits = logits[:, -1]
                nxt = next_logits.argmax(dim=-1, keepdim=True)
                if eos_token_id is not None:
                    nxt = torch.where(
                        finished[:, None],
                        torch.full_like(nxt, eos_token_id),
                        nxt,
                    )
                out = torch.cat([out, nxt], dim=1)
                if eos_token_id is not None:
                    finished |= nxt.squeeze(1).eq(eos_token_id)
                    if torch.all(finished):
                        break
            return out, aux
        finally:
            self.train(was_training)
