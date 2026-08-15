from __future__ import annotations

import math

import torch

from binding_diagnostics import (
    exact_query_row,
    extract_frozen_features,
    oracle_answer_memory,
)
from matched_structured_lookup_experiment import (
    FOGStructuredLookup,
    StructuredBatch,
    StructuredModelConfig,
    StructuredTaskConfig,
    build_model,
)


def _manual_batch(mapping: tuple[int, ...], query: int) -> StructuredBatch:
    sources = tuple(range(len(mapping)))
    return StructuredBatch(
        row_sources=torch.tensor([sources]),
        row_values=torch.tensor([[mapping[source] for source in sources]]),
        query_keys=torch.tensor([query]),
        targets=torch.tensor([mapping[query]]),
        sample_indices=(0,),
        mappings=(mapping,),
        signatures=((*sources, *mapping, query, mapping[query]),),
    )


def _small_model() -> tuple[FOGStructuredLookup, StructuredTaskConfig]:
    task = StructuredTaskConfig(table_size=4)
    cfg = StructuredModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        max_seq_len=16,
        latent_slots=2,
        reasoning_steps=1,
        compare_rank=4,
        planner_ff=32,
        memory_slots=2,
        n_reasoning_modes=2,
        fixed_orthogonal_keys=True,
    )
    model = build_model("fog_strict", task, cfg, model_seed=7)
    assert isinstance(model, FOGStructuredLookup)
    return model, task


def test_pooling_is_mapping_invariant_but_exact_address_preserves_binding() -> None:
    model, task = _small_model()
    first = _manual_batch((0, 1, 2, 3), query=0)
    second = _manual_batch((1, 0, 2, 3), query=0)
    first_prompt = model.encode_prompt(first)
    second_prompt = model.encode_prompt(second)
    first_rows = first_prompt[:, : task.table_size]
    second_rows = second_prompt[:, : task.table_size]

    # Every key/value appears once, so the mean cannot identify the permutation.
    assert torch.allclose(first_rows.mean(1), second_rows.mean(1), atol=1e-7)
    # Query-conditioned addressing selects rows with different bound values.
    first_selected = exact_query_row(first_rows, first.row_sources, first.query_keys)
    second_selected = exact_query_row(
        second_rows, second.row_sources, second.query_keys
    )
    assert not torch.allclose(first_selected, second_selected)


def test_frozen_extraction_has_expected_shapes_and_does_not_mutate() -> None:
    model, task = _small_model()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    bank = extract_frozen_features(
        model,
        task,
        data_seed=31,
        split="train",
        examples=8,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert bank.features["raw_exact_query_row"].shape == (8, 16)
    assert bank.features["memory_step_1_flat"].shape == (8, 32)
    assert bank.slots["memory_step_1"].shape == (8, 2, 16)
    assert bank.targets.shape == (8,)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_oracle_answer_memory_is_exact_and_repeated() -> None:
    targets = torch.tensor([0, 2, 3])
    memory = oracle_answer_memory(
        targets,
        table_size=4,
        d_model=8,
        memory_slots=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert memory.shape == (3, 3, 8)
    assert torch.equal(memory[:, 0], memory[:, 1])
    assert memory[1, 0].argmax().item() == 2
    assert math.isclose(memory[1, 0, 2].item(), 8**0.5, rel_tol=1e-7)
