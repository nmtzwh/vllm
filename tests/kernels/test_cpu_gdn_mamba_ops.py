# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as custom_ops
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skipif(
        not hasattr(torch.ops._C, "cpu_causal_conv1d_update"),
        reason="CPU GDN mamba ops are not built",
    ),
]


def _l2norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


def _reference_causal_conv1d_update(
    x: torch.Tensor,
    conv_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state_indices: torch.Tensor,
    silu: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = weight.size(1)
    state_ref = conv_states.clone()
    out = torch.empty_like(x)
    for batch_idx, state_idx in enumerate(state_indices.tolist()):
        window = torch.cat(
            [state_ref[state_idx], x[batch_idx].unsqueeze(-1)], dim=-1
        ).to(torch.float32)
        curr = (window * weight.to(torch.float32)).sum(dim=-1)
        if bias is not None:
            curr = curr + bias.to(torch.float32)
        if silu:
            curr = curr * torch.sigmoid(curr)
        out[batch_idx] = curr.to(x.dtype)
        state_ref[state_idx] = window[:, 1:width].to(conv_states.dtype)
    return out, state_ref


def _reference_causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    silu: bool,
) -> torch.Tensor:
    # x is expected in transposed layout [B, D, T]
    x_pad = torch.nn.functional.pad(x.to(torch.float32), (weight.size(1) - 1, 0))
    y = torch.nn.functional.conv1d(
        x_pad,
        weight.to(torch.float32).unsqueeze(1),
        bias=bias.to(torch.float32) if bias is not None else None,
        groups=x.size(1),
    )
    if silu:
        y = torch.nn.functional.silu(y)
    return y.to(x.dtype)


def _reference_fused_sigmoid_update(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len, batch_size, num_heads, head_dim = q.shape
    v_num_heads = v.size(2)
    v_head_dim = v.size(3)
    group_size = v_num_heads // num_heads
    scale = head_dim**-0.5

    state_ref = initial_state_source.clone()
    out = torch.empty((batch_size, seq_len, v_num_heads, v_head_dim), dtype=q.dtype)

    for bi, cache_index in enumerate(initial_state_indices.tolist()):
        for si in range(seq_len):
            for vi in range(v_num_heads):
                qi = vi // group_size
                q_vec = q[si, bi, qi].to(torch.float32)
                k_vec = k[si, bi, qi].to(torch.float32)
                if use_qk_l2norm_in_kernel:
                    q_vec = _l2norm(q_vec)
                    k_vec = _l2norm(k_vec)

                state = state_ref[cache_index, vi]
                g_val = -torch.exp(A_log[vi].to(torch.float32)) * torch.nn.functional.softplus(
                    a[bi, vi].to(torch.float32) + dt_bias[vi].to(torch.float32),
                    beta=1.0,
                    threshold=20.0,
                )
                g_exp = torch.exp(g_val)
                scaled_state = state * g_exp
                kv_mem = (scaled_state * k_vec[:, None]).sum(dim=0)
                dt = (
                    v[si, bi, vi].to(torch.float32) - kv_mem
                ) * torch.sigmoid(b[bi, vi].to(torch.float32))
                updated_state = scaled_state + k_vec[:, None] * dt[None, :]
                out[bi, si, vi] = (
                    (updated_state * q_vec[:, None]).sum(dim=0) * scale
                ).to(q.dtype)
                state_ref[cache_index, vi] = updated_state

    return out, state_ref


@pytest.mark.parametrize("silu", [False, True])
def test_cpu_causal_conv1d_fwd_matches_reference(silu: bool) -> None:
    torch.manual_seed(7)

    batch = 2
    dim = 64
    seqlen = 5
    width = 4

    x = torch.randn(batch, seqlen, dim, dtype=torch.bfloat16).transpose(1, 2)
    weight = torch.randn(dim, width, dtype=torch.bfloat16)
    bias = torch.randn(dim, dtype=torch.bfloat16)

    expected_out = _reference_causal_conv1d_fwd(x, weight, bias, silu)
    actual_out = custom_ops.cpu_causal_conv1d_fwd(
        x,
        weight,
        bias,
        None,
        None,
        None,
        None,
        silu,
        -1,
        False,
    )

    torch.testing.assert_close(
        actual_out.to(torch.float32),
        expected_out.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.parametrize("silu", [False, True])
def test_cpu_causal_conv1d_update_matches_reference(silu: bool) -> None:
    torch.manual_seed(0)

    batch = 2
    dim = 64
    width = 4
    num_states = 3

    x = torch.randn(batch, dim, dtype=torch.bfloat16)
    conv_states = torch.randn(num_states, dim, width - 1, dtype=torch.bfloat16)
    weight = torch.randn(dim, width, dtype=torch.bfloat16)
    bias = torch.randn(dim, dtype=torch.bfloat16)
    state_indices = torch.tensor([2, 0], dtype=torch.int32)

    expected_out, expected_states = _reference_causal_conv1d_update(
        x, conv_states, weight, bias, state_indices, silu
    )
    actual_states = conv_states.clone()
    actual_out = custom_ops.cpu_causal_conv1d_update(
        x,
        actual_states,
        weight,
        bias,
        silu,
        None,
        state_indices,
        -1,
        False,
    )

    torch.testing.assert_close(
        actual_out.to(torch.float32),
        expected_out.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        actual_states.to(torch.float32),
        expected_states.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )


def test_cpu_fused_gdn_gating_matches_reference() -> None:
    torch.manual_seed(1)

    batch = 4
    heads = 8
    A_log = torch.randn(heads, dtype=torch.bfloat16)
    a = torch.randn(batch, heads, dtype=torch.bfloat16)
    b = torch.randn(batch, heads, dtype=torch.bfloat16)
    dt_bias = torch.randn(heads, dtype=torch.bfloat16)

    actual_g, actual_beta = custom_ops.cpu_fused_gdn_gating(A_log, a, b, dt_bias)
    expected_g = (
        -torch.exp(A_log.to(torch.float32)).unsqueeze(0)
        * torch.nn.functional.softplus(
            a.to(torch.float32) + dt_bias.to(torch.float32),
            beta=1.0,
            threshold=20.0,
        )
    ).unsqueeze(0)
    expected_beta = torch.sigmoid(b.to(torch.float32)).to(torch.bfloat16).unsqueeze(0)

    torch.testing.assert_close(actual_g, expected_g, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        actual_beta.to(torch.float32),
        expected_beta.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )


def test_cpu_chunk_gated_delta_rule_smoke() -> None:
    torch.manual_seed(11)

    bsz = 1
    seq_len = 4
    qk_heads = 2
    v_heads = 4
    qk_dim = 32
    v_dim = 32
    num_states = 2

    q = torch.randn(bsz, seq_len, qk_heads, qk_dim, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(bsz, seq_len, v_heads, v_dim, dtype=torch.bfloat16)
    g = torch.randn(bsz, seq_len, v_heads, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(bsz, seq_len, v_heads)).to(torch.bfloat16)
    initial_state = torch.randn(
        num_states, v_heads, qk_dim, v_dim, dtype=torch.float32
    )
    cu_seqlens = torch.tensor([0, seq_len, seq_len], dtype=torch.int32)

    out, final_state = custom_ops.cpu_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        True,
        cu_seqlens,
        False,
        True,
    )

    assert out.shape == v.shape
    assert final_state.shape == initial_state.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(final_state).all()


def test_cpu_fused_sigmoid_gating_delta_rule_update_matches_reference() -> None:
    torch.manual_seed(2)

    seq_len = 2
    batch = 3
    num_heads = 2
    v_num_heads = 4
    head_dim = 32
    v_head_dim = 32
    num_states = 5

    q = torch.randn(seq_len, batch, num_heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(seq_len, batch, v_num_heads, v_head_dim, dtype=torch.bfloat16)
    a = torch.randn(batch, v_num_heads, dtype=torch.bfloat16)
    b = torch.randn(batch, v_num_heads, dtype=torch.bfloat16)
    A_log = torch.randn(v_num_heads, dtype=torch.bfloat16)
    dt_bias = torch.randn(v_num_heads, dtype=torch.bfloat16)
    state_indices = torch.tensor([3, 0, 4], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 4, 6], dtype=torch.int32)

    expected_state = torch.randn(
        num_states, v_num_heads, head_dim, v_head_dim, dtype=torch.float32
    )
    actual_state = expected_state.clone()

    expected_out, expected_state = _reference_fused_sigmoid_update(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        expected_state,
        state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    actual_out = custom_ops.cpu_fused_sigmoid_gating_delta_rule_update(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        actual_state,
        state_indices,
        cu_seqlens,
        True,
    )

    torch.testing.assert_close(
        actual_out.to(torch.float32),
        expected_out.to(torch.float32),
        atol=6e-2,
        rtol=6e-2,
    )
    torch.testing.assert_close(
        actual_state,
        expected_state,
        atol=6e-2,
        rtol=6e-2,
    )
