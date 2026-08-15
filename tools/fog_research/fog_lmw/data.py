"""Tokenizer and dataset utilities for real-data FOG training.

The core package deliberately depends only on PyTorch.  Hugging Face
``tokenizers`` and ``datasets`` are imported lazily by the functions that need
them, so importing :mod:`fog_lmw` keeps working in a minimal installation.

The collators in this module expose ordinary causal-LM tensors while the
prompt/response collator also returns the separate prompt and answer tensors
used by :class:`fog_lmw.model.FOGLatentReasoner`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
import json
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import torch


IGNORE_INDEX = -100
TargetMode = Literal["full", "final"]


@dataclass(frozen=True)
class SpecialTokens:
    """The four tokens required by the training and generation pipelines."""

    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"

    def as_list(self) -> list[str]:
        values = [self.pad, self.bos, self.eos, self.unk]
        if any(not value for value in values):
            raise ValueError("special tokens must be non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError("PAD/BOS/EOS/UNK special tokens must be distinct")
        return values


DEFAULT_SPECIAL_TOKENS = SpecialTokens()


def _require_tokenizers() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        tokenizers = import_module("tokenizers")
        models = import_module("tokenizers.models")
        trainers = import_module("tokenizers.trainers")
        pre_tokenizers = import_module("tokenizers.pre_tokenizers")
        decoders = import_module("tokenizers.decoders")
        normalizers = import_module("tokenizers.normalizers")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "BPE tokenizer support requires the optional Hugging Face "
            "dependency `tokenizers`. Install it with `pip install tokenizers`."
        ) from exc
    return tokenizers, models, trainers, pre_tokenizers, decoders, normalizers


def _require_datasets() -> Any:
    try:
        return import_module("datasets")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Hugging Face dataset loading requires the optional dependency "
            "`datasets`. Install it with `pip install datasets`."
        ) from exc


class BPETokenizer:
    """Small wrapper around a Hugging Face byte-level BPE tokenizer.

    ``encode`` returns Python integer lists, which keeps the wrapper usable in
    plain iterators and PyTorch ``DataLoader`` workers without introducing a
    Transformers dependency.
    """

    def __init__(
        self,
        tokenizer: Any,
        special_tokens: SpecialTokens = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        self._tokenizer = tokenizer
        self.special_tokens = special_tokens
        self._special_ids: dict[str, int] = {}
        for name, token in zip(
            ("pad", "bos", "eos", "unk"), special_tokens.as_list(), strict=True
        ):
            token_id = tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(
                    f"tokenizer is missing required {name.upper()} token {token!r}"
                )
            self._special_ids[name] = int(token_id)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 8192,
        min_frequency: int = 2,
        special_tokens: SpecialTokens = DEFAULT_SPECIAL_TOKENS,
        show_progress: bool = False,
        length: int | None = None,
    ) -> "BPETokenizer":
        """Train byte-level BPE from a restartable or one-shot text iterable."""

        (
            tokenizers,
            models,
            trainers,
            pre_tokenizers,
            decoders,
            normalizers,
        ) = _require_tokenizers()
        special = special_tokens.as_list()
        byte_alphabet = pre_tokenizers.ByteLevel.alphabet()
        minimum_vocab = len(byte_alphabet) + len(special)
        if vocab_size < minimum_vocab:
            raise ValueError(
                f"vocab_size must be >= {minimum_vocab} for byte-level BPE "
                f"with {len(special)} special tokens"
            )
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
        if length is not None and length < 0:
            raise ValueError("length must be >= 0")

        tokenizer = tokenizers.Tokenizer(models.BPE(unk_token=special_tokens.unk))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special,
            initial_alphabet=byte_alphabet,
            show_progress=show_progress,
        )

        def clean_texts() -> Iterator[str]:
            for index, text in enumerate(texts):
                if not isinstance(text, str):
                    raise TypeError(
                        f"tokenizer training item {index} must be str, "
                        f"got {type(text).__name__}"
                    )
                if text:
                    yield text

        tokenizer.train_from_iterator(clean_texts(), trainer=trainer, length=length)
        return cls(tokenizer, special_tokens)

    @classmethod
    def byte_fallback(
        cls,
        *,
        vocab_size: int = 8192,
        special_tokens: SpecialTokens = DEFAULT_SPECIAL_TOKENS,
    ) -> "BPETokenizer":
        """Build an offline-safe byte tokenizer padded to ``vocab_size``.

        Only the 256 byte symbols and four boundary tokens are emitted.  The
        remaining IDs are explicit reserved tokens, preserving the exact 8192
        embedding geometry until a corpus-trained BPE is available.  This is
        inefficient but lossless and works for arbitrary UTF-8 text.
        """

        (
            tokenizers,
            models,
            _,
            pre_tokenizers,
            decoders,
            normalizers,
        ) = _require_tokenizers()
        special = special_tokens.as_list()
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        minimum_vocab = len(special) + len(alphabet)
        if vocab_size < minimum_vocab:
            raise ValueError(f"vocab_size must be >= {minimum_vocab}")
        vocabulary: dict[str, int] = {
            token: index for index, token in enumerate([*special, *alphabet])
        }
        for index in range(minimum_vocab, vocab_size):
            vocabulary[f"<reserved_{index:05d}>"] = index
        tokenizer = tokenizers.Tokenizer(
            models.BPE(vocab=vocabulary, merges=[], unk_token=special_tokens.unk)
        )
        # Mark existing vocabulary entries as special so normal decode omits
        # BOS/EOS/PAD/UNK without allocating new IDs.
        tokenizer.add_special_tokens(special)
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        return cls(tokenizer, special_tokens)

    @classmethod
    def train_from_files(
        cls,
        paths: str | PathLike[str] | Sequence[str | PathLike[str]],
        *,
        text_field: str = "text",
        vocab_size: int = 8192,
        min_frequency: int = 2,
        special_tokens: SpecialTokens = DEFAULT_SPECIAL_TOKENS,
        show_progress: bool = False,
    ) -> "BPETokenizer":
        """Train from local ``.txt`` or ``.jsonl`` records without preloading."""

        records = iter_local_records(paths, text_field=text_field)

        def texts() -> Iterator[str]:
            for row in records:
                value = row.get(text_field)
                if not isinstance(value, str):
                    raise TypeError(
                        f"field {text_field!r} must be str, got "
                        f"{type(value).__name__}"
                    )
                yield value

        return cls.train(
            texts(),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=show_progress,
        )

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        *,
        special_tokens: SpecialTokens = DEFAULT_SPECIAL_TOKENS,
    ) -> "BPETokenizer":
        tokenizers, *_ = _require_tokenizers()
        tokenizer = tokenizers.Tokenizer.from_file(str(path))
        return cls(tokenizer, special_tokens)

    def save(self, path: str | PathLike[str]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(output))
        return output

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())

    @property
    def pad_token_id(self) -> int:
        return self._special_ids["pad"]

    @property
    def bos_token_id(self) -> int:
        return self._special_ids["bos"]

    @property
    def eos_token_id(self) -> int:
        return self._special_ids["eos"]

    @property
    def unk_token_id(self) -> int:
        return self._special_ids["unk"]

    def token_to_id(self, token: str) -> int | None:
        token_id = self._tokenizer.token_to_id(token)
        return None if token_id is None else int(token_id)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be >= 1")
        ids = [int(token_id) for token_id in self._tokenizer.encode(text).ids]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            if add_eos:
                ids[-1] = self.eos_token_id
        return ids

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
    ) -> list[list[int]]:
        return [
            self.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
                max_length=max_length,
            )
            for text in texts
        ]

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(
            [int(token_id) for token_id in ids],
            skip_special_tokens=skip_special_tokens,
        )


def _as_paths(
    paths: str | PathLike[str] | Sequence[str | PathLike[str]],
) -> list[Path]:
    if isinstance(paths, (str, PathLike)):
        resolved = [Path(paths)]
    else:
        resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("at least one local data path is required")
    return resolved


def iter_text_records(
    paths: str | PathLike[str] | Sequence[str | PathLike[str]],
    *,
    text_field: str = "text",
    encoding: str = "utf-8",
    keep_blank: bool = False,
) -> Iterator[dict[str, str]]:
    """Yield one text record per line from one or more UTF-8 ``.txt`` files."""

    for path in _as_paths(paths):
        with path.open("r", encoding=encoding) as handle:
            for line in handle:
                text = line.rstrip("\r\n")
                if keep_blank or text.strip():
                    yield {text_field: text}


def iter_jsonl_records(
    paths: str | PathLike[str] | Sequence[str | PathLike[str]],
    *,
    required_fields: Sequence[str] = (),
    encoding: str = "utf-8",
) -> Iterator[dict[str, Any]]:
    """Yield JSON objects and report malformed records with path and line."""

    for path in _as_paths(paths):
        with path.open("r", encoding=encoding) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path} at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"JSONL record in {path} at line {line_number} must be "
                        f"an object, got {type(value).__name__}"
                    )
                missing = [field for field in required_fields if field not in value]
                if missing:
                    raise ValueError(
                        f"JSONL record in {path} at line {line_number} is missing "
                        f"required field(s): {', '.join(missing)}"
                    )
                yield value


def iter_local_records(
    paths: str | PathLike[str] | Sequence[str | PathLike[str]],
    *,
    text_field: str = "text",
    required_fields: Sequence[str] = (),
    encoding: str = "utf-8",
) -> Iterator[dict[str, Any]]:
    """Dispatch local records by extension (``.txt`` or ``.jsonl``)."""

    for path in _as_paths(paths):
        suffix = path.suffix.lower()
        if suffix == ".txt":
            yield from iter_text_records(
                path, text_field=text_field, encoding=encoding
            )
        elif suffix in {".jsonl", ".ndjson"}:
            yield from iter_jsonl_records(
                path, required_fields=required_fields, encoding=encoding
            )
        else:
            raise ValueError(
                f"unsupported local dataset extension {suffix!r} for {path}; "
                "expected .txt, .jsonl, or .ndjson"
            )


def load_hf_dataset(
    dataset_id: str,
    *,
    config: str | None = None,
    split: str = "train",
    revision: str | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a Hub dataset with revision pinning and optional streaming.

    The return type is Hugging Face ``Dataset`` when ``streaming=False`` and
    ``IterableDataset`` when ``streaming=True``.
    """

    if not dataset_id:
        raise ValueError("dataset_id must be non-empty")
    if not split:
        raise ValueError("split must be non-empty")
    datasets = _require_datasets()
    options: dict[str, Any] = {
        "split": split,
        "streaming": streaming,
        **kwargs,
    }
    if revision is not None:
        options["revision"] = revision
    return datasets.load_dataset(dataset_id, name=config, **options)


@dataclass(frozen=True)
class PromptResponse:
    prompt: str
    response: str


def extract_gsm8k_final(answer: str, *, marker: str = "####") -> str:
    """Extract the canonical GSM8K final answer following ``####``."""

    if not isinstance(answer, str):
        raise TypeError(f"GSM8K answer must be str, got {type(answer).__name__}")
    if marker not in answer:
        raise ValueError(f"GSM8K answer is missing final-answer marker {marker!r}")
    final = answer.rsplit(marker, 1)[1].strip()
    if not final:
        raise ValueError("GSM8K final answer is empty")
    return final


def parse_prompt_response(
    record: Mapping[str, Any],
    *,
    prompt_field: str = "question",
    response_field: str = "answer",
    target_mode: TargetMode = "full",
    prompt_template: str = "{prompt}",
    final_marker: str = "####",
) -> PromptResponse:
    """Validate and format a generic prompt/response or GSM8K record."""

    if target_mode not in {"full", "final"}:
        raise ValueError("target_mode must be 'full' or 'final'")
    if prompt_field not in record:
        raise ValueError(f"record is missing prompt field {prompt_field!r}")
    if response_field not in record:
        raise ValueError(f"record is missing response field {response_field!r}")
    prompt = record[prompt_field]
    response = record[response_field]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"field {prompt_field!r} must be a non-empty string")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"field {response_field!r} must be a non-empty string")
    try:
        formatted_prompt = prompt_template.format(prompt=prompt.strip())
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            "prompt_template must be a valid format string containing {prompt}"
        ) from exc
    if "{prompt}" not in prompt_template:
        raise ValueError("prompt_template must contain the {prompt} placeholder")
    target = response.strip()
    if target_mode == "final":
        target = extract_gsm8k_final(target, marker=final_marker)
    return PromptResponse(formatted_prompt, target)


def _pad_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("cannot pad an empty batch")
    width = max(len(sequence) for sequence in sequences)
    if width < 1:
        raise ValueError("sequences must contain at least one token")
    ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        if sequence:
            length = len(sequence)
            ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
            mask[row, :length] = 1
    return ids, mask


def _encode_with_boundaries(
    tokenizer: Any,
    text: str,
    *,
    add_bos: bool,
    add_eos: bool,
    max_length: int | None,
) -> list[int]:
    ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    if not isinstance(ids, list):
        ids = list(ids)
    ids = [int(token_id) for token_id in ids]
    if max_length is not None:
        if max_length < 1:
            raise ValueError("maximum token lengths must be >= 1")
        if len(ids) > max_length:
            ids = ids[:max_length]
            if add_eos:
                ids[-1] = int(tokenizer.eos_token_id)
    return ids


class TextBlockCollator:
    """Pack text records into fixed causal-language-model blocks.

    Stories are separated by BOS/EOS tokens and concatenated.  Labels follow
    the Hugging Face causal-LM convention: they initially equal ``input_ids``
    and padding positions are replaced by ``ignore_index``.  The model loss is
    responsible for the single causal shift (logits ``[:-1]`` against labels
    ``[1:]``).  The final partial block is right-padded unless
    ``drop_last=True``.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        block_size: int = 256,
        text_field: str = "text",
        add_bos: bool = True,
        add_eos: bool = True,
        drop_last: bool = False,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if block_size < 2:
            raise ValueError("block_size must be >= 2 for shifted causal loss")
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_field = text_field
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.drop_last = drop_last
        self.ignore_index = ignore_index

    def __call__(self, examples: Sequence[str | Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        stream: list[int] = []
        for index, example in enumerate(examples):
            if isinstance(example, str):
                text = example
            elif isinstance(example, Mapping):
                if self.text_field not in example:
                    raise ValueError(
                        f"text example {index} is missing field {self.text_field!r}"
                    )
                text = example[self.text_field]
            else:
                raise TypeError(
                    f"text example {index} must be str or mapping, got "
                    f"{type(example).__name__}"
                )
            if not isinstance(text, str):
                raise TypeError(f"field {self.text_field!r} must be str")
            if not text.strip():
                continue
            stream.extend(
                _encode_with_boundaries(
                    self.tokenizer,
                    text,
                    add_bos=self.add_bos,
                    add_eos=self.add_eos,
                    max_length=None,
                )
            )

        if len(stream) < 2:
            raise ValueError("batch contains fewer than two usable text tokens")

        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for start in range(0, len(stream), self.block_size):
            chunk = stream[start : start + self.block_size]
            if len(chunk) < self.block_size and self.drop_last:
                break
            valid_tokens = len(chunk)
            if valid_tokens < 2:
                continue
            inputs = chunk + [self.tokenizer.pad_token_id] * (
                self.block_size - valid_tokens
            )
            labels = inputs.copy()
            labels[valid_tokens:] = [self.ignore_index] * (
                self.block_size - valid_tokens
            )
            mask = [1] * valid_tokens + [0] * (self.block_size - valid_tokens)
            input_rows.append(inputs)
            label_rows.append(labels)
            mask_rows.append(mask)

        if not input_rows:
            raise ValueError("batch does not contain a complete text block")
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }


TinyStoriesBlockCollator = TextBlockCollator


class PromptResponseCollator:
    """Collate generic prompt/answer rows, including GSM8K.

    Returned ``labels`` follow the Hugging Face causal-LM convention: they have
    the same shape as ``input_ids`` and prompt, answer-BOS, and padding
    positions are set to ``ignore_index``.  Separate padded prompt and answer
    tensors are included for the latent reasoner's native loss path.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        prompt_field: str = "question",
        response_field: str = "answer",
        target_mode: TargetMode = "full",
        prompt_template: str = "{prompt}",
        final_marker: str = "####",
        max_prompt_length: int = 256,
        max_answer_length: int = 256,
        max_sequence_length: int | None = None,
        add_prompt_bos: bool = True,
        add_prompt_eos: bool = True,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if target_mode not in {"full", "final"}:
            raise ValueError("target_mode must be 'full' or 'final'")
        if max_prompt_length < 1 or max_answer_length < 2:
            raise ValueError(
                "max_prompt_length must be >= 1 and max_answer_length must be >= 2"
            )
        if max_sequence_length is not None and max_sequence_length < 3:
            raise ValueError("max_sequence_length must be >= 3")
        self.tokenizer = tokenizer
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.target_mode = target_mode
        self.prompt_template = prompt_template
        self.final_marker = final_marker
        self.max_prompt_length = max_prompt_length
        self.max_answer_length = max_answer_length
        self.max_sequence_length = max_sequence_length
        self.add_prompt_bos = add_prompt_bos
        self.add_prompt_eos = add_prompt_eos
        self.ignore_index = ignore_index

    def _encode_record(
        self, record: Mapping[str, Any]
    ) -> tuple[list[int], list[int]]:
        parsed = parse_prompt_response(
            record,
            prompt_field=self.prompt_field,
            response_field=self.response_field,
            target_mode=self.target_mode,
            prompt_template=self.prompt_template,
            final_marker=self.final_marker,
        )
        prompt = _encode_with_boundaries(
            self.tokenizer,
            parsed.prompt,
            add_bos=self.add_prompt_bos,
            add_eos=self.add_prompt_eos,
            max_length=self.max_prompt_length,
        )
        response_tokens = _encode_with_boundaries(
            self.tokenizer,
            parsed.response,
            add_bos=False,
            add_eos=False,
            max_length=None,
        )
        room = self.max_answer_length - 2
        response_tokens = response_tokens[:room]
        answer = [self.tokenizer.bos_token_id, *response_tokens, self.tokenizer.eos_token_id]

        if self.max_sequence_length is not None:
            overflow = len(prompt) + len(answer) - self.max_sequence_length
            if overflow > 0:
                # Preserve the complete answer target and the beginning/end
                # boundary tokens of the prompt whenever possible.
                keep_prompt = len(prompt) - overflow
                if keep_prompt < 1:
                    answer_room = self.max_sequence_length - 1
                    if answer_room < 2:
                        raise ValueError("max_sequence_length leaves no room for an answer")
                    answer = answer[:answer_room]
                    answer[-1] = self.tokenizer.eos_token_id
                    keep_prompt = 1
                if keep_prompt < len(prompt):
                    prompt = prompt[:keep_prompt]
                    if self.add_prompt_eos:
                        prompt[-1] = self.tokenizer.eos_token_id
        return prompt, answer

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty prompt/response batch")
        prompts: list[list[int]] = []
        answers: list[list[int]] = []
        combined: list[list[int]] = []
        prompt_prefix_lengths: list[int] = []
        for index, example in enumerate(examples):
            if not isinstance(example, Mapping):
                raise TypeError(
                    f"prompt/response example {index} must be a mapping, got "
                    f"{type(example).__name__}"
                )
            prompt, answer = self._encode_record(example)
            if not prompt:
                raise ValueError(f"prompt/response example {index} encoded to no tokens")
            prompts.append(prompt)
            answers.append(answer)
            combined.append([*prompt, *answer])
            # Mask the lexical prompt and answer BOS.  With standard shifted
            # causal loss the first response token remains supervised.
            prompt_prefix_lengths.append(len(prompt) + 1)

        prompt_ids, prompt_mask = _pad_sequences(
            prompts, pad_token_id=self.tokenizer.pad_token_id
        )
        answer_ids, answer_mask = _pad_sequences(
            answers, pad_token_id=self.tokenizer.pad_token_id
        )
        input_ids, attention_mask = _pad_sequences(
            combined, pad_token_id=self.tokenizer.pad_token_id
        )

        labels = input_ids.clone()
        answer_labels = answer_ids.clone()
        for row, prefix_length in enumerate(prompt_prefix_lengths):
            labels[row, :prefix_length] = self.ignore_index
            answer_labels[row, 0] = self.ignore_index
        labels.masked_fill_(attention_mask.eq(0), self.ignore_index)
        answer_labels.masked_fill_(answer_mask.eq(0), self.ignore_index)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "answer_ids_with_bos": answer_ids,
            "answer_attention_mask": answer_mask,
            "answer_labels": answer_labels,
        }


__all__ = [
    "IGNORE_INDEX",
    "TargetMode",
    "SpecialTokens",
    "DEFAULT_SPECIAL_TOKENS",
    "BPETokenizer",
    "iter_text_records",
    "iter_jsonl_records",
    "iter_local_records",
    "load_hf_dataset",
    "PromptResponse",
    "extract_gsm8k_final",
    "parse_prompt_response",
    "TextBlockCollator",
    "TinyStoriesBlockCollator",
    "PromptResponseCollator",
]
