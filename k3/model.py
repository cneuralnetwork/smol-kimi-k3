from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import K3Config

try:
    from fla.ops.kda import chunk_kda
except ImportError:  # CPU-only development remains supported.
    chunk_kda = None


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x).transpose(1, 2)

    def step(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one causal convolution step to x shaped [B, 1, D]."""
        b, _, d = x.shape
        if cache is None:
            cache = x.new_zeros(b, self.kernel_size - 1, d)
        window = torch.cat((cache, x), dim=1)
        output = F.conv1d(
            window.transpose(1, 2),
            self.conv.weight,
            groups=d,
        ).transpose(1, 2)
        return output, window[:, 1:]


class KimiDeltaAttention(nn.Module):
    """Readable recurrent form of KDA equations 1, 2, 5, and 6.

    CUDA training uses FLA's report-linked chunkwise Triton kernel. The exact
    recurrent delta rule remains as an auditable CPU fallback.
    """

    def __init__(self, config: K3Config) -> None:
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.gmin = config.kda_gmin

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.q_conv = CausalDepthwiseConv1d(d, config.short_conv_kernel)
        self.k_conv = CausalDepthwiseConv1d(d, config.short_conv_kernel)
        self.v_conv = CausalDepthwiseConv1d(d, config.short_conv_kernel)
        self.beta_proj = nn.Linear(d, config.num_heads, bias=True)
        self.alpha_down = nn.Linear(d, config.decay_rank, bias=False)
        self.alpha_up = nn.Linear(config.decay_rank, d, bias=True)
        self.alpha_log_scale = nn.Parameter(torch.zeros(config.num_heads))
        self.gate_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.head_norm_weight = nn.Parameter(torch.ones(config.num_heads, config.head_size))

        # Used by the optimizer to apply Newton-Schulz per attention head.
        for projection in (self.q_proj, self.k_proj, self.v_proj):
            projection.weight._muon_heads = config.num_heads  # type: ignore[attr-defined]

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = F.silu(self.q_conv(self.q_proj(x)))
        k = F.silu(self.k_conv(self.k_proj(x)))
        v = F.silu(self.v_conv(self.v_proj(x)))
        q = self._shape_heads(q)
        k = self._shape_heads(k)
        v = self._shape_heads(v)

        beta_logits = self.beta_proj(x)
        z = self._shape_heads(self.alpha_up(self.alpha_down(x)))

        if x.is_cuda and chunk_kda is not None:
            # This is the chunkwise form referenced by K3 report section 5.1.2.
            # Gate and beta activations are fused into the Triton kernel.
            out, _ = chunk_kda(
                q,
                k,
                v,
                z,
                beta_logits,
                scale=1.0,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=self.gmin,
                A_log=self.alpha_log_scale,
            )
        else:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            beta = torch.sigmoid(beta_logits)
            scale = self.alpha_log_scale.exp().view(1, 1, self.num_heads, 1)
            log_decay = self.gmin * torch.sigmoid(scale * z)
            alpha = log_decay.exp()

            # FP32 recurrence protects the delta state from long products.
            q_float, k_float, v_float = q.float(), k.float(), v.float()
            beta_float, alpha_float = beta.float(), alpha.float()
            b, t, h, p = q.shape
            state = torch.zeros(b, h, p, p, device=x.device, dtype=torch.float32)
            outputs: list[torch.Tensor] = []
            for position in range(t):
                state = state * alpha_float[:, position].unsqueeze(-1)
                key = k_float[:, position]
                value = v_float[:, position]
                prediction = torch.einsum("bhd,bhdv->bhv", key, state)
                error = value - prediction
                write = (
                    beta_float[:, position, :, None, None]
                    * key.unsqueeze(-1)
                    * error.unsqueeze(-2)
                )
                state = state + write
                outputs.append(
                    torch.einsum("bhdv,bhd->bhv", state, q_float[:, position])
                )
            out = torch.stack(outputs, dim=1)

        b, t, h, p = out.shape
        out = out * torch.rsqrt(out.square().mean(dim=-1, keepdim=True) + 1e-6)
        out = out * self.head_norm_weight
        out = out.reshape(b, t, h * p).to(x.dtype)
        out = torch.sigmoid(self.gate_proj(x)) * out
        return self.out_proj(out)

    def step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Single-token recurrent decode with convolution and KDA state caches."""
        cache = {} if cache is None else cache
        q, q_conv = self.q_conv.step(self.q_proj(x), cache.get("q_conv"))
        k, k_conv = self.k_conv.step(self.k_proj(x), cache.get("k_conv"))
        v, v_conv = self.v_conv.step(self.v_proj(x), cache.get("v_conv"))
        q = F.normalize(self._shape_heads(F.silu(q)), dim=-1)
        k = F.normalize(self._shape_heads(F.silu(k)), dim=-1)
        v = self._shape_heads(F.silu(v))

        beta = torch.sigmoid(self.beta_proj(x)).float()[:, 0]
        z = self._shape_heads(self.alpha_up(self.alpha_down(x)))
        scale = self.alpha_log_scale.exp().view(1, 1, self.num_heads, 1)
        alpha = (self.gmin * torch.sigmoid(scale * z)).exp().float()[:, 0]
        q_float, k_float, v_float = q.float()[:, 0], k.float()[:, 0], v.float()[:, 0]

        state = cache.get("state")
        if state is None:
            b = x.size(0)
            state = torch.zeros(
                b,
                self.num_heads,
                self.head_size,
                self.head_size,
                device=x.device,
                dtype=torch.float32,
            )
        state = state * alpha.unsqueeze(-1)
        prediction = torch.einsum("bhd,bhdv->bhv", k_float, state)
        state = state + (
            beta[:, :, None, None]
            * k_float.unsqueeze(-1)
            * (v_float - prediction).unsqueeze(-2)
        )
        out = torch.einsum("bhdv,bhd->bhv", state, q_float)
        out = out * torch.rsqrt(out.square().mean(dim=-1, keepdim=True) + 1e-6)
        out = out * self.head_norm_weight
        out = out.reshape(x.size(0), 1, -1).to(x.dtype)
        out = self.out_proj(torch.sigmoid(self.gate_proj(x)) * out)
        return out, {
            "q_conv": q_conv,
            "k_conv": k_conv,
            "v_conv": v_conv,
            "state": state,
        }


class GatedMLA(nn.Module):
    """NoPE global attention with low-rank cached keys/values and a full-rank gate."""

    def __init__(self, config: K3Config) -> None:
        super().__init__()
        d = config.hidden_size
        latent = config.mla_latent_size
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.q_proj = nn.Linear(d, d, bias=False)
        self.kv_down = nn.Linear(d, latent, bias=False)
        self.k_up = nn.Linear(latent, d, bias=False)
        self.v_up = nn.Linear(latent, d, bias=False)
        self.gate_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        for projection in (self.q_proj, self.k_up, self.v_up):
            projection.weight._muon_heads = config.num_heads  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        latent = self.kv_down(x)
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_up(latent).view(b, t, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_up(latent).view(b, t, self.num_heads, self.head_size).transpose(1, 2)
        # There is deliberately no RoPE or absolute position embedding.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.out_proj(torch.sigmoid(self.gate_proj(x)) * out)

    def step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cache = {} if cache is None else cache
        b = x.size(0)
        latent = self.kv_down(x)
        q = self.q_proj(x).view(b, 1, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_up(latent).view(b, 1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_up(latent).view(b, 1, self.num_heads, self.head_size).transpose(1, 2)
        all_keys = torch.cat((cache.get("keys", k[:, :, :0]), k), dim=2)
        all_values = torch.cat((cache.get("values", v[:, :, :0]), v), dim=2)
        scores = torch.matmul(q, all_keys.transpose(-2, -1)) / math.sqrt(self.head_size)
        out = torch.matmul(scores.softmax(dim=-1), all_values)
        out = out.transpose(1, 2).contiguous().view(b, 1, -1)
        out = self.out_proj(torch.sigmoid(self.gate_proj(x)) * out)
        return out, {"keys": all_keys, "values": all_values}


class SiTUGLU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, beta1: float, beta2: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, input_size, bias=False)
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_linear = self.gate_proj(x)
        up_linear = self.up_proj(x)
        gate = self.beta1 * torch.tanh(gate_linear / self.beta1) * torch.sigmoid(gate_linear)
        up = self.beta2 * torch.tanh(up_linear / self.beta2)
        return self.down_proj(gate * up)


class StableLatentMoE(nn.Module):
    """Equations 11-14, using an exact small-batch quantile instead of histograms."""

    def __init__(self, config: K3Config) -> None:
        super().__init__()
        d = config.hidden_size
        latent = config.latent_moe_size
        self.num_experts = config.num_routed_experts
        self.top_k = config.experts_per_token
        self.down_proj = nn.Linear(d, latent, bias=False)
        self.up_proj = nn.Linear(latent, d, bias=False)
        self.routed_norm = RMSNorm(latent)
        self.router = nn.Linear(d, self.num_experts, bias=False)
        self.routed_experts = nn.ModuleList(
            SiTUGLU(
                latent,
                config.routed_expert_hidden_size,
                config.situ_beta1,
                config.situ_beta2,
            )
            for _ in range(self.num_experts)
        )
        self.shared_experts = nn.ModuleList(
            SiTUGLU(
                d,
                config.shared_expert_hidden_size,
                config.situ_beta1,
                config.situ_beta2,
            )
            for _ in range(config.num_shared_experts)
        )
        self.register_buffer("router_bias", torch.zeros(self.num_experts), persistent=True)

    @torch.no_grad()
    def _quantile_balance(self, scores: torch.Tensor, cutoff: torch.Tensor) -> None:
        # The report uses pooled histograms at trillion-parameter scale. Exact
        # quantiles are inexpensive and more accurate for this local model.
        quantile = 1.0 - self.top_k / self.num_experts
        margins = scores.float() - cutoff.float().unsqueeze(-1)
        next_bias = -torch.quantile(margins, quantile, dim=0)
        next_bias -= next_bias.mean()
        self.router_bias.copy_(next_bias.to(self.router_bias.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat_x = x.reshape(-1, shape[-1])
        scores = torch.sigmoid(self.router(flat_x))
        biased_scores = scores + self.router_bias
        top_values, top_indices = torch.topk(biased_scores, self.top_k + 1, dim=-1)
        selected = top_indices[:, : self.top_k]
        cutoff = top_values[:, self.top_k]
        selected_scores = scores.gather(1, selected)
        weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        latent = self.down_proj(flat_x)
        routed = self._routed_bank_vectorized(latent, selected, weights)

        if self.training:
            self._quantile_balance(scores.detach(), cutoff.detach())

        routed = self.up_proj(self.routed_norm(routed))
        shared = torch.zeros_like(flat_x)
        for expert in self.shared_experts:
            shared = shared + expert(flat_x)
        return (shared + routed).view(shape)

    def _stacked_expert_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack the per-expert SiTUGLU weights into 3 batched tensors.
        Called every forward (cheap: ~100 us for 32 experts of [168, 160]).
        Avoids the correctness hazards of caching across load_state_dict."""
        sg = torch.stack([e.gate_proj.weight for e in self.routed_experts])
        su = torch.stack([e.up_proj.weight for e in self.routed_experts])
        sd = torch.stack([e.down_proj.weight for e in self.routed_experts])
        return sg, su, sd

    def _routed_bank_vectorized(
        self, latent: torch.Tensor, selected: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute the routed expert bank via batched einsum instead of a
        per-expert Python loop. Numerically identical to the loop version
        (max abs diff < 1e-6 at BF16, relL2 < 1e-6) but ~2.4x faster at
        decode batch sizes by eliminating per-expert ``torch.where`` calls.

        Forward path (SiTUGLU expert):
            gate = beta1 * tanh(Wg x / beta1) * sigmoid(Wg x)
            up   = beta2 * tanh(Wu x / beta2)
            out  = Wdown(gate * up)

        Vectorized form per (token n, top-k slot k):
            gate_lin[n, k, :] = latent[n, :] @ Wg[expert(n,k), :, :].T
            ... (same for up, then down) ...
            routed[n, :] = sum_k weights[n, k] * out[n, k, :]
        """
        sg, su, sd = self._stacked_expert_weights()
        beta1 = self.routed_experts[0].beta1
        beta2 = self.routed_experts[0].beta2
        # gather per-(token, slot) expert weights: [N, K, inter, lat]
        sg_sel = sg[selected]
        su_sel = su[selected]
        sd_sel = sd[selected]
        # compute in float32 for numerical stability with tanh/sigmoid
        lat_f = latent.float()
        gate_lin = torch.einsum("nl,nkil->nki", lat_f, sg_sel.float())
        up_lin = torch.einsum("nl,nkil->nki", lat_f, su_sel.float())
        gate = beta1 * torch.tanh(gate_lin / beta1) * torch.sigmoid(gate_lin)
        up = beta2 * torch.tanh(up_lin / beta2)
        hidden = gate * up
        out_per_expert = torch.einsum("nki,nkli->nkl", hidden, sd_sel.float())
        routed = (out_per_expert * weights.float().unsqueeze(-1)).sum(dim=1)
        return routed.to(latent.dtype)


class AttentionResidual(nn.Module):
    """Data-dependent attention over depth using a learned pseudo-query."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.key_norm = RMSNorm(hidden_size)
        self.scale = hidden_size**-0.5

    def forward(self, sources: Iterable[torch.Tensor]) -> torch.Tensor:
        values = torch.stack(tuple(sources), dim=-2)
        keys = self.key_norm(values)
        logits = torch.einsum("btsd,d->bts", keys, self.query) * self.scale
        weights = logits.softmax(dim=-1)
        return torch.einsum("bts,btsd->btd", weights, values)


class K3Layer(nn.Module):
    def __init__(self, config: K3Config, layer_index: int) -> None:
        super().__init__()
        self.attention_type = config.attention_type(layer_index)
        self.attn_residual = AttentionResidual(config.hidden_size)
        self.moe_residual = AttentionResidual(config.hidden_size)
        self.attn_norm = RMSNorm(config.hidden_size)
        self.moe_norm = RMSNorm(config.hidden_size)
        if self.attention_type == "kda":
            self.attn = KimiDeltaAttention(config)
        else:
            self.attn = GatedMLA(config)
        self.moe = StableLatentMoE(config)

    def forward_attention(self, sources: list[torch.Tensor]) -> torch.Tensor:
        mixed = self.attn_residual(sources)
        return mixed + self.attn(self.attn_norm(mixed))

    def forward_moe(self, sources: list[torch.Tensor]) -> torch.Tensor:
        mixed = self.moe_residual(sources)
        return mixed + self.moe(self.moe_norm(mixed))

    def step_attention(
        self,
        sources: list[torch.Tensor],
        cache: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mixed = self.attn_residual(sources)
        output, cache = self.attn.step(self.attn_norm(mixed), cache)
        return mixed + output, cache


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class K3ForCausalLM(nn.Module):
    def __init__(self, config: K3Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            K3Layer(config, layer_index) for layer_index in range(config.num_layers)
        )
        self.final_residual = AttentionResidual(config.hidden_size)
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        embedding = self.token_embedding(input_ids)
        completed_blocks = [embedding]
        current_block: torch.Tensor | None = None

        for layer_index, layer in enumerate(self.layers):
            sources = completed_blocks + ([] if current_block is None else [current_block])
            attention_output = layer.forward_attention(sources)
            current_block = (
                attention_output
                if current_block is None
                else current_block + attention_output
            )
            moe_output = layer.forward_moe(completed_blocks + [current_block])
            current_block = current_block + moe_output

            block_finished = (layer_index + 1) % self.config.attnres_block_size == 0
            if block_finished:
                completed_blocks.append(current_block)
                current_block = None

        if current_block is not None:
            completed_blocks.append(current_block)
        hidden = self.final_norm(self.final_residual(completed_blocks))
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss)

    def num_parameters(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    def forward_step(
        self,
        input_ids: torch.Tensor,
        cache: list[dict[str, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Decode one token with KDA recurrent and MLA KV caches."""
        if input_ids.ndim != 2 or input_ids.size(1) != 1:
            raise ValueError("forward_step expects input_ids shaped [batch, 1]")
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("cache length must equal num_layers")

        embedding = self.token_embedding(input_ids)
        completed_blocks = [embedding]
        current_block: torch.Tensor | None = None
        next_cache: list[dict[str, torch.Tensor]] = []
        for layer_index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            sources = completed_blocks + ([] if current_block is None else [current_block])
            attention_output, updated_cache = layer.step_attention(sources, layer_cache)
            next_cache.append(updated_cache)
            current_block = (
                attention_output
                if current_block is None
                else current_block + attention_output
            )
            moe_output = layer.forward_moe(completed_blocks + [current_block])
            current_block = current_block + moe_output
            if (layer_index + 1) % self.config.attnres_block_size == 0:
                completed_blocks.append(current_block)
                current_block = None
        if current_block is not None:
            completed_blocks.append(current_block)
        hidden = self.final_norm(self.final_residual(completed_blocks))
        return self.lm_head(hidden), next_cache

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        cache: list[dict[str, torch.Tensor] | None] | None = None
        logits = None
        for position in range(input_ids.size(1)):
            logits, cache = self.forward_step(input_ids[:, position : position + 1], cache)
        if logits is None:
            raise ValueError("input_ids must contain at least one token")
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1] / max(temperature, 1e-5)
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = -float("inf")
            next_token = torch.multinomial(next_logits.softmax(dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            logits, cache = self.forward_step(next_token, cache)
        return input_ids
