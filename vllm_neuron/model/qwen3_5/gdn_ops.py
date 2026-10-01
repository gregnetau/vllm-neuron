# SPDX-License-Identifier: Apache-2.0
"""
Gated DeltaNet reference ops (pure PyTorch, single sequence, explicit state)
============================================================================

These are the correctness oracles / CPU fallbacks for the NKI kernels that will
back Qwen3.5 linear-attention layers. They depend only on ``torch`` so they can be
unit-tested without a Neuron install.

Layouts (one sequence, no batch dim; the runner loops/packs requests):
    q, k    [L, H, Dk]   (already repeated up to the value-head count H)
    v       [L, H, Dv]
    g, beta [L, H]       g = log decay (<= 0), beta in (0, 1)
    state   [H, Dk, Dv]  recurrent state, float32
    conv    x [L, C], weight [C, K], conv_state [C, K-1]

The delta rule and its chunked (WY) form follow the FLA / HF ``modeling_qwen3_5``
reference (Apache-2.0); the chunk kernel here solves the intra-chunk triangular
system with ``solve_triangular`` instead of HF's row-by-row loop.
"""

import torch
import torch.nn.functional as F

CHUNK_SIZE = 64


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def gdn_gating(
    a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """a, b: [L, H] projections. Returns (g, beta), both float32 [L, H]."""
    beta = b.float().sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    return g, beta


def causal_conv1d(
    x: torch.Tensor, weight: torch.Tensor, conv_state: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv + SiLU over one sequence, carrying ``conv_state``.

    x [L, C], weight [C, K], conv_state [C, K-1] (the last K-1 raw inputs of the
    previous call, oldest first) or None (zeros). Returns (y [L, C], new_state).
    """
    L, C = x.shape
    K = weight.shape[1]
    if conv_state is None:
        conv_state = x.new_zeros(C, K - 1)
    xt = torch.cat([conv_state.to(x.dtype), x.t()], dim=-1)  # [C, K-1+L]
    new_state = xt[:, -(K - 1) :].clone() if K > 1 else xt[:, :0]
    y = F.conv1d(xt.unsqueeze(0), weight.unsqueeze(1).to(x.dtype), groups=C)[0]
    return F.silu(y).t().contiguous(), new_state


def recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token delta rule (decode path). Returns (out [L, H, Dv], state)."""
    dtype = q.dtype
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    q, k, v = (t.float() for t in (q, k, v))
    L, H, Dk = k.shape
    q = q * Dk**-0.5
    S = (
        v.new_zeros(H, Dk, v.shape[-1])
        if state is None
        else state.to(torch.float32).clone()
    )
    out = torch.empty(L, H, v.shape[-1], dtype=torch.float32, device=v.device)
    for t in range(L):
        S = S * g[t].exp()[:, None, None]
        kv_mem = (S * k[t][:, :, None]).sum(1)  # [H, Dv]
        delta = (v[t] - kv_mem) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        out[t] = (S * q[t][:, :, None]).sum(1)
    return out.to(dtype), S


def inverse_unit_lower(m: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    """(I + m)^-1 for strictly lower-triangular ``m`` [..., C, C] (C a power of two).

    ``m`` is nilpotent (m^C = 0), so the inverse is the finite series sum_j (-m)^j, which
    factors exactly as (I - m)(I + m^2)(I + m^4)...(I + m^(C/2)): log2(C) batched matmuls.
    Used instead of ``torch.linalg.solve_triangular``, which the Neuron XLA bridge cannot
    lower (custom call) — and matmuls are what the hardware is good at anyway.
    """
    C = m.shape[-1]
    assert C & (C - 1) == 0, "chunk size must be a power of two"
    out = eye - m
    p = m
    for _ in range(C.bit_length() - 2):  # m^2, m^4, ..., m^(C/2)
        p = p @ p
        out = out @ (eye + p)
    return out


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
    use_qk_l2norm: bool = True,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked delta rule (prefill path). Same contract as the recurrent form.

    Written for the Neuron compiler: masks come from ``arange`` comparisons and
    ``torch.where`` (no bool ``tril``/``masked_fill``), the exponent is clamped before
    ``exp``, and chunk outputs are stacked rather than slice-assigned into a buffer.
    """
    dtype = q.dtype
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    # [L, H, D] -> [H, L, D] float32
    q, k, v = (t.transpose(0, 1).float() for t in (q, k, v))
    g, beta = g.transpose(0, 1).float(), beta.transpose(0, 1).float()
    H, L, Dk = k.shape
    Dv = v.shape[-1]

    pad = (chunk_size - L % chunk_size) % chunk_size
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    n = (L + pad) // chunk_size
    q = q * Dk**-0.5

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, v, k_beta, v_beta = (
        t.reshape(H, n, chunk_size, t.shape[-1]) for t in (q, k, v, k_beta, v_beta)
    )
    g = g.reshape(H, n, chunk_size).cumsum(-1)

    r = torch.arange(chunk_size, device=q.device)
    incl = r[:, None] >= r[None, :]  # i >= j
    strict = r[:, None] > r[None, :]  # i > j
    eye = (r[:, None] == r[None, :]).to(torch.float32)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2)  # g_i - g_j (<= 0 on/below the diagonal)
    decay = torch.where(incl, torch.exp(torch.where(incl, diff, torch.zeros_like(diff))),
                        torch.zeros_like(diff))
    kk = (k_beta @ k.transpose(-1, -2)) * decay
    m = torch.where(strict, kk, torch.zeros_like(kk))
    # T = (I + m)^-1 : intra-chunk WY correction (matmul-only; see inverse_unit_lower)
    T = inverse_unit_lower(m, eye)
    v_corr = T @ v_beta
    k_cumdecay = T @ (k_beta * g.exp().unsqueeze(-1))

    S = (
        q.new_zeros(H, Dk, Dv)
        if state is None
        else state.to(torch.float32).clone()
    )
    outs = []
    for i in range(n):
        q_i, k_i = q[:, i], k[:, i]
        qk = q_i @ k_i.transpose(-1, -2) * decay[:, i]
        attn = torch.where(incl, qk, torch.zeros_like(qk))
        v_new = v_corr[:, i] - k_cumdecay[:, i] @ S
        outs.append((q_i * g[:, i, :, None].exp()) @ S + attn @ v_new)
        g_last = g[:, i, -1]
        S = S * g_last.exp()[:, None, None] + (
            k_i * (g_last[:, None] - g[:, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new
    out = torch.stack(outs, dim=1).reshape(H, n * chunk_size, Dv)[:, :L].transpose(0, 1)
    return out.contiguous().to(dtype), S


def gated_rmsnorm(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Per-head RMSNorm (plain ``weight``, not zero-centered) times SiLU(gate)."""
    dtype = x.dtype
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    x = weight * x32.to(dtype)
    return (x * F.silu(gate.float())).to(dtype)


def gdn_layer_reference(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    cfg,
    conv_state: torch.Tensor | None,
    ssm_state: torch.Tensor | None,
    decode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Whole linear-attention block for one sequence (spec for the device layer).

    hidden [L, hidden_size]; ``w`` holds HF-named tensors: in_proj_qkv, in_proj_z,
    in_proj_b, in_proj_a, conv1d (either [C, 1, K] or [C, K]), A_log, dt_bias,
    norm (plain-weight gated RMSNorm), out_proj. ``cfg`` needs the ``linear_*``
    fields, ``rms_norm_eps``. Returns (out [L, hidden_size], conv_state, ssm_state).
    """
    L = hidden.shape[0]
    Hk, Hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    Dk, Dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    mixed = hidden @ w["in_proj_qkv"].t()
    z = (hidden @ w["in_proj_z"].t()).reshape(L, Hv, Dv)
    b = hidden @ w["in_proj_b"].t()
    a = hidden @ w["in_proj_a"].t()

    conv_w = w["conv1d"]
    if conv_w.dim() == 3:
        conv_w = conv_w.squeeze(1)
    mixed, conv_state = causal_conv1d(mixed, conv_w, conv_state)
    q, k, v = torch.split(mixed, [Hk * Dk, Hk * Dk, Hv * Dv], dim=-1)
    q, k = q.reshape(L, Hk, Dk), k.reshape(L, Hk, Dk)
    v = v.reshape(L, Hv, Dv)
    rep = Hv // Hk
    if rep > 1:
        q, k = q.repeat_interleave(rep, dim=1), k.repeat_interleave(rep, dim=1)

    g, beta = gdn_gating(a, b, w["A_log"], w["dt_bias"])
    rule = recurrent_gated_delta_rule if decode else chunk_gated_delta_rule
    o, ssm_state = rule(q, k, v, g, beta, ssm_state)
    o = gated_rmsnorm(o, z, w["norm"], cfg.rms_norm_eps).reshape(L, Hv * Dv)
    return o @ w["out_proj"].t(), conv_state, ssm_state


# ---------------------------------------------------------------------------
# Batched single-token (decode) forms: one token per request, per-request state.
# ---------------------------------------------------------------------------
def causal_conv1d_step_batched(
    x: torch.Tensor, weight: torch.Tensor, conv_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """x [B, C], weight [C, K], conv_state [B, C, K-1] -> (silu(conv) [B, C], new_state)."""
    xt = torch.cat([conv_state.to(x.dtype), x.unsqueeze(-1)], dim=-1)  # [B, C, K]
    y = (xt * weight.unsqueeze(0).to(x.dtype)).sum(-1)
    return F.silu(y), xt[:, :, 1:]


def gated_delta_step_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One delta-rule step for B requests. q,k [B,H,Dk]; v [B,H,Dv]; g,beta [B,H];
    state [B,H,Dk,Dv] float32. Returns (out [B,H,Dv] in q.dtype, new_state)."""
    dtype = q.dtype
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    q, k, v = q.float() * q.shape[-1] ** -0.5, k.float(), v.float()
    S = state.float() * g.float().exp()[:, :, None, None]
    kv_mem = (S * k[..., None]).sum(-2)  # [B, H, Dv]
    delta = (v - kv_mem) * beta.float()[..., None]
    S = S + k[..., None] * delta[:, :, None, :]
    out = (S * q[..., None]).sum(-2)
    return out.to(dtype), S
