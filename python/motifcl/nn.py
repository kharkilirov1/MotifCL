from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence

from . import (
    GELU as _NativeGELU,
    GemmaConfig,
    GemmaTokenizer,
    HFArchitecture,
    HFChatMessage,
    HFChatTemplateKind,
    HFTransformerConfig,
    HFTokenizer,
    HFWeightLoadReport,
    HFWeightName,
    GPTModel as _NativeGPTModel,
    KVCache,
    PagedKVCache,
    DeltaStateCache,
    Linear as _NativeLinear,
    ModernGPTModel as _NativeModernGPTModel,
    ModernMLP as _NativeModernMLP,
    ModernSelfAttention as _NativeModernSelfAttention,
    ModernTransformerBlock as _NativeModernTransformerBlock,
    MoEFFN as _NativeMoEFFN,
    GatedDeltaNetLayer as _NativeGatedDeltaNetLayer,
    GatedAttentionLayer as _NativeGatedAttentionLayer,
    Parameter,
    QuantizationPolicy,
    QuantizedLinear as _NativeQuantizedLinear,
    Sequential as _NativeSequential,
    Tensor,
    TransformerConfig,
    GenerateOptions,
    disable_hf_transformer_quantized_inference,
    enable_hf_transformer_quantized_inference,
    expected_gemma_hf_weight_names,
    expected_hf_transformer_weight_names,
    generate,
    generate_batch,
    generate_batch_text,
    generate_hf_batch_text,
    generate_hf_text,
    generate_text,
    hf_architecture_name,
    infer_hf_chat_template_kind,
    apply_hf_chat_template,
    load_gemma_config_json,
    load_gemma_hf_weights,
    load_hf_tokenizer,
    load_hf_transformer_config_json,
    load_hf_transformer_config_gguf,
    load_hf_transformer_weights,
    load_hf_transformer_gguf_weights,
    load_parameters,
    make_gemma_model,
    make_hf_transformer_model,
    map_gemma_hf_weight_name,
    map_hf_transformer_weight_name,
    parse_hf_architecture,
    save_parameters,
    to_gemma_compatible_config,
    to_transformer_config,
)


def _unwrap_module(module):
    return module._native if isinstance(module, Module) and module._native is not None else module


class Module:
    """Small PyTorch-like Python facade over MotifCL native modules."""

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_native", None)
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value) -> None:
        object.__setattr__(self, name, value)
        if name.startswith("_"):
            return
        modules = self.__dict__.get("_modules")
        parameters = self.__dict__.get("_parameters")
        if modules is None or parameters is None:
            return
        modules.pop(name, None)
        parameters.pop(name, None)
        if isinstance(value, Module):
            modules[name] = value
        elif isinstance(value, Parameter):
            parameters[name] = value

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - abstract-style method
        native = self.__dict__.get("_native")
        if native is None:
            raise NotImplementedError(f"{type(self).__name__}.forward is not implemented")
        return native.forward(*args, **kwargs)

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        native = self.__dict__.get("_native")
        if native is not None and hasattr(native, "parameters"):
            params.extend(native.parameters())
        params.extend(self.__dict__.get("_parameters", {}).values())
        for module in self.__dict__.get("_modules", {}).values():
            params.extend(module.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        out: list[tuple[str, Parameter]] = []
        native = self.__dict__.get("_native")
        if native is not None and hasattr(native, "parameters"):
            for i, param in enumerate(native.parameters()):
                out.append((f"{prefix}param_{i}", param))
        for name, param in self.__dict__.get("_parameters", {}).items():
            out.append((prefix + name, param))
        for name, module in self.__dict__.get("_modules", {}).items():
            out.extend(module.named_parameters(prefix + name + "."))
        return out

    def zero_grad(self) -> None:
        native = self.__dict__.get("_native")
        if native is not None and hasattr(native, "zero_grad"):
            native.zero_grad()
        for param in self.__dict__.get("_parameters", {}).values():
            param.zero_grad()
        for module in self.__dict__.get("_modules", {}).values():
            module.zero_grad()

    def train(self, mode: bool = True):
        object.__setattr__(self, "training", bool(mode))
        for module in self.__dict__.get("_modules", {}).values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict((name, param.data) for name, param in self.named_parameters())

    def load_state_dict(self, state: dict[str, Tensor], strict: bool = True) -> None:
        named = OrderedDict(self.named_parameters())
        missing = [name for name in named if name not in state]
        unexpected = [name for name in state if name not in named]
        if strict and (missing or unexpected):
            raise KeyError(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")
        for name, param in named.items():
            if name in state:
                param.data = state[name]

    def save(self, path: str) -> None:
        save_parameters(self.parameters(), path)

    def load(self, backend, path: str) -> None:
        load_parameters(self.parameters(), backend, path)


class Linear(Module):
    def __init__(self, backend, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        object.__setattr__(self, "_native", _NativeLinear(backend, in_features, out_features, bias))

    @property
    def weight(self) -> Parameter:
        return self._native.weight

    @property
    def bias(self) -> Parameter:
        return self._native.bias

    def has_bias(self) -> bool:
        return self._native.has_bias()

    def enable_quantized_inference(self, qdtype=None) -> None:
        if qdtype is None:
            self._native.enable_quantized_inference()
        else:
            self._native.enable_quantized_inference(qdtype)

    def set_quantized_weight(self, weight: Tensor) -> None:
        self._native.set_quantized_weight(weight)

    def disable_quantized_inference(self) -> None:
        self._native.disable_quantized_inference()

    def quantized_inference_enabled(self) -> bool:
        return self._native.quantized_inference_enabled()

    def quantized_weight_dtype(self):
        return self._native.quantized_weight_dtype()

    def quantized_weight(self):
        return self._native.quantized_weight()


class GELU(Module):
    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_native", _NativeGELU())


class Sequential(Module):
    def __init__(self, modules: Sequence[Module] | None = None) -> None:
        super().__init__()
        for i, module in enumerate(modules or []):
            self.add(module, name=str(i))

    def add(self, module: Module, name: str | None = None) -> None:
        setattr(self, name if name is not None else str(len(self._modules)), module)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class QuantizedLinear(Module):
    def __init__(self, weight: Tensor, bias_or_qdtype=None, qdtype=None) -> None:
        super().__init__()
        if bias_or_qdtype is None and qdtype is None:
            native = _NativeQuantizedLinear(weight)
        elif isinstance(bias_or_qdtype, Tensor):
            native = _NativeQuantizedLinear(weight, bias_or_qdtype, qdtype) if qdtype is not None else _NativeQuantizedLinear(weight, bias_or_qdtype)
        else:
            native = _NativeQuantizedLinear(weight, bias_or_qdtype)
        object.__setattr__(self, "_native", native)

    @staticmethod
    def from_linear(linear: Linear, qdtype=None):
        out = QuantizedLinear.__new__(QuantizedLinear)
        Module.__init__(out)
        native_linear = _unwrap_module(linear)
        native = _NativeQuantizedLinear.from_linear(native_linear) if qdtype is None else _NativeQuantizedLinear.from_linear(native_linear, qdtype)
        object.__setattr__(out, "_native", native)
        return out

    def quantized_weight(self):
        return self._native.quantized_weight()

    def bias(self):
        return self._native.bias()

    def has_bias(self) -> bool:
        return self._native.has_bias()

    def weight_dtype(self):
        return self._native.weight_dtype()

    def in_features(self) -> int:
        return self._native.in_features()

    def out_features(self) -> int:
        return self._native.out_features()


class GPTModel(Module):
    def __init__(self, backend, vocab_size: int, block_size: int, n_embd: int, n_head: int, n_layer: int, mlp_hidden: int) -> None:
        super().__init__()
        object.__setattr__(self, "_native", _NativeGPTModel(backend, vocab_size, block_size, n_embd, n_head, n_layer, mlp_hidden))


ModernMLP = _NativeModernMLP
MoEFFN = _NativeMoEFFN
GatedDeltaNetLayer = _NativeGatedDeltaNetLayer
ModernSelfAttention = _NativeModernSelfAttention
GatedAttentionLayer = _NativeGatedAttentionLayer
ModernTransformerBlock = _NativeModernTransformerBlock
ModernGPTModel = _NativeModernGPTModel
