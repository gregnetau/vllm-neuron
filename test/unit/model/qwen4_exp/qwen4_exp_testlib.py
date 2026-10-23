# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the Qwen3.8-Flash-Next (qwen4_exp) CPU unit tests.

Test modules load it as a plugin (``pytest_plugins = ["qwen4_exp_testlib"]``) for the
fixtures and import the helpers from it. There is deliberately no ``conftest.py`` here: test
directories without ``__init__.py`` share the module name ``conftest``, so a second one breaks
``from conftest import ...`` in the other model test directories of the same pytest session.

Run with the Neuron vLLM venv and this checkout first on the path::

    PYTHONPATH=$PWD python -m pytest test/unit/model/qwen4_exp

The oracle is the Hugging Face implementation (transformers >= 5.17 ships ``qwen4_exp``).
``tiny_text_config_dict`` keeps the real config's structure (layer layout, PLE on layer 2, QSA,
hyper-connections, MoE with a shared expert) at sizes that run in seconds on CPU. Its QSA
budget (16 tokens = 4 blocks) is small so that sequences of a few dozen tokens already take
the sparse path.

The plugin harness runs the Neuron model at tp=1 with the parallel layers replaced by plain
torch and the NKI-backed ops (``NF.*``) used by the prefill paths replaced by torch versions.
"""

import json
import os
import pathlib
import sys
import types

import pytest
import torch
import torch.nn.functional as F

REAL_CONFIG = pathlib.Path(__file__).with_name("qwen3_8_flash_next_config.json")

# Real checkpoint for the optional tests that read it (skipped when unset or missing).
MODEL_PATH_ENV = "QWEN4_EXP_MODEL_PATH"

TINY_OVERRIDES = dict(
    hidden_size=256,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    num_experts=16,
    num_experts_per_tok=4,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    hc_lowrank=32,
    ple_embed_dim=256,
    ngram_vocab_size_base=1000,
    split_ngram_parts=4,
    indexer_head_dim=32,
    indexer_budget=16,
    vocab_size=1024,
    eos_token_id=1,
    bos_token_id=1,
)


def real_text_config_dict() -> dict:
    return json.loads(REAL_CONFIG.read_text())["text_config"]


def tiny_text_config_dict(**overrides) -> dict:
    cfg = dict(real_text_config_dict())
    for k in ("mtp", "layer_types"):
        cfg.pop(k, None)
    cfg.update(TINY_OVERRIDES)
    cfg.update(overrides)
    # Rotary dim = 64 * 0.25 = 16 -> 8 frequencies; mrope sections must sum to 8.
    cfg["rope_parameters"] = dict(cfg["rope_parameters"], mrope_section=[4, 2, 2])
    return cfg


def checkpoint_dir() -> pathlib.Path | None:
    """``$QWEN4_EXP_MODEL_PATH`` if it names a checkpoint directory with an index, else None."""
    path = os.environ.get(MODEL_PATH_ENV)
    if not path:
        return None
    path = pathlib.Path(path)
    return path if (path / "model.safetensors.index.json").is_file() else None


requires_checkpoint = pytest.mark.skipif(
    checkpoint_dir() is None,
    reason=f"set {MODEL_PATH_ENV} to a Qwen3.8-Flash-Next checkpoint directory",
)


# ---------------------------------------------------------------------------
# Plugin harness (tp=1, torch stand-ins for the parallel layers and NKI ops)
# ---------------------------------------------------------------------------


class FakeTP:
    world_size, rank_in_group, device_group = 1, 0, None


def fake_nf():
    """Torch versions of the ``NF`` ops the Qwen3.5 prefill attention calls."""

    def qkv_proj(hidden, qkv_weights, bias=None, **kw):
        return hidden @ qkv_weights

    def o_proj(active, weight, bias=None, **kw):
        if active.dim() == 4:  # [B, N, D, S]
            B, N, D, S = active.shape
            active = active.reshape(B, N * D, S).transpose(1, 2)
        return active @ weight

    def flash_attention(q, k, v, scale=None, tp_q=True, tp_out=False, **kw):
        S = q.shape[-1]
        scores = torch.einsum("bds,bdt->bst", q.float(), k.float()) * scale
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
        p = scores.masked_fill(~causal, float("-inf")).softmax(-1)
        return torch.einsum("bst,btd->bds", p, v.float()).to(q.dtype)

    return types.SimpleNamespace(
        qkv_proj=qkv_proj, o_proj=o_proj, flash_attention=flash_attention
    )


class FakeEmbed(torch.nn.Module):
    """``VocabDimShardedEmbedding`` at tp=1."""

    def __init__(self, vocab_size, embed_dim, dtype, tp_group=None):
        super().__init__()
        self.vocab_size_per_rank = vocab_size
        self.weight = torch.nn.Parameter(
            torch.empty(vocab_size, embed_dim, dtype=dtype)
        )

    def forward(self, ids, scatter_tokens=False, rank=None):
        return F.embedding(ids, self.weight)


class FakeColumnParallel(torch.nn.Module):
    """``ColumnParallelLinear`` at tp=1."""

    def __init__(
        self,
        i,
        o,
        bias=False,
        dtype=None,
        gather_output=True,
        tp_group=None,
        weight_layout="out_in",
    ):
        super().__init__()
        self.weight_layout = weight_layout
        shape = (o, i) if weight_layout == "out_in" else (i, o)
        self.weight = torch.nn.Parameter(torch.empty(*shape, dtype=dtype))

    def forward(self, x):
        return x @ (self.weight.t() if self.weight_layout == "out_in" else self.weight)


class FakeSlice:
    """A full tensor posing as a safetensors slice for the weight loaders."""

    def __init__(self, t):
        self.t = t

    def get_shape(self):
        return list(self.t.shape)

    def __getitem__(self, idx):
        return self.t[idx]


def patch_plugin(monkeypatch, *modules: str):
    """Import ``vllm_neuron.model.qwen4_exp.{modules}`` fresh with the tp=1 stand-ins.

    Returns the imported modules; call ``unload_plugin(*modules)`` when done.
    """
    import vllm_neuron.model.qwen3_5.model as Q
    import vllm_neuron.model.qwen4_exp.moe as moe

    unload_plugin(*modules)
    monkeypatch.setattr(Q, "get_tp_group", lambda: FakeTP())
    monkeypatch.setattr(Q, "NF", fake_nf())
    monkeypatch.setattr(moe, "get_tp_group", lambda: FakeTP())
    out = []
    for name in modules:
        mod = __import__(f"vllm_neuron.model.qwen4_exp.{name}", fromlist=["_"])
        monkeypatch.setattr(mod, "get_tp_group", lambda: FakeTP())
        monkeypatch.setattr(mod, "VocabDimShardedEmbedding", FakeEmbed)
        out.append(mod)
    import vllm_neuron.nn as neuron_nn

    monkeypatch.setattr(neuron_nn, "ColumnParallelLinear", FakeColumnParallel)
    return out


def unload_plugin(*modules: str) -> None:
    for name in modules:
        sys.modules.pop(f"vllm_neuron.model.qwen4_exp.{name}", None)


def hf_checkpoint_state_dict(hf) -> dict:
    """HF ``Qwen4ExpForCausalLM`` state dict under the published checkpoint's names."""
    return {
        (
            k.replace("model.", "model.language_model.", 1)
            if k.startswith("model.")
            else k
        ): v
        for k, v in hf.state_dict().items()
    }


def load_from_hf(model, hf) -> None:
    """Load ``model`` from an HF model (see ``load_state``)."""
    load_state(model, hf_checkpoint_state_dict(hf))


def load_state(model, sd: dict) -> None:
    """Load ``model`` through its own mappings and weight loaders from checkpoint tensors.

    Asserts every parameter is mapped to existing tensors and loads to its shape.
    """
    from vllm_neuron.utils import weight_loader as wl

    mappings = model._weight_mappings()
    params = dict(model.named_parameters())
    assert set(mappings) == set(params), set(mappings) ^ set(params)
    for name, keys in mappings.items():
        keys = [keys] if isinstance(keys, str) else keys
        missing = [k for k in keys if k not in sd]
        assert not missing, (name, missing)
        loader = wl.get_weight_loader(params[name])
        slices = [FakeSlice(sd[k]) for k in keys]
        loaded = loader.load(slices, 0) if loader is not None else slices[0].t
        assert loaded.shape == params[name].shape, (
            name,
            loaded.shape,
            params[name].shape,
        )
        params[name].data = loaded.to(params[name].dtype)
    model._materialize_buffers()


def ple_rows(hf, ids):
    """HF n-gram embedding rows for a whole sequence [T, E] (stateless full-sequence call)."""
    with torch.no_grad():
        return hf.model.layers[1].ple.ple_embedding(ids[None], None)[0]


def state_pages(spec, num_pages: int, fill: int = 0) -> dict:
    """Paired-half state pages for every state layer of ``spec``, padded to rows of 128
    floats per half as on device; ``fill`` sets every byte (dirty pages)."""
    from vllm_neuron.model import state_cache

    out = {}
    for sl in spec.state_layers:
        page_bytes = state_cache.page_layout(sl.shapes, sl.dtypes).page_bytes + 1024
        raw = torch.full((num_pages * page_bytes,), fill, dtype=torch.int8)
        out[sl.name] = list(state_cache.paired_half_views(raw, page_bytes))
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_text_config():
    from transformers import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(**tiny_text_config_dict())


@pytest.fixture
def tiny_hf_model(tiny_text_config):
    """Seeded tiny ``Qwen4ExpForCausalLM`` (fp32, eval). Norm weights get noise so that
    zero-centered-norm handling is exercised (HF initialises them to 0)."""
    from transformers import Qwen4ExpForCausalLM

    torch.manual_seed(0)
    model = Qwen4ExpForCausalLM(tiny_text_config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.dim() == 1:
                p.add_(0.1 * torch.randn_like(p))
            if name.endswith("conv1d.weight") and ".ple." in name:
                p.normal_(0, 0.2)  # HF inits the PLE conv to zero
    return model


@pytest.fixture
def qwen4_config(tiny_text_config):
    """The plugin's Qwen4ExpConfig built from the tiny HF config."""
    from vllm_neuron.model.qwen4_exp.config import Qwen4ExpConfig

    return Qwen4ExpConfig.from_configs(tiny_text_config)


@pytest.fixture
def plugin(monkeypatch):
    """``qwen4_exp.model`` imported with the tp=1 torch stand-ins."""
    (model,) = patch_plugin(monkeypatch, "model")
    yield model
    unload_plugin("model")


@pytest.fixture
def plugin_mtp(monkeypatch):
    """``qwen4_exp.model`` and ``qwen4_exp.mtp`` imported with the tp=1 torch stand-ins."""
    model, mtp = patch_plugin(monkeypatch, "model", "mtp")
    yield model, mtp
    unload_plugin("model", "mtp")
