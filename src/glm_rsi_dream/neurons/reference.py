"""Differentiable frozen GLM suffix for answer-level consolidation experiments.

Geometry follows the inspected TensorFold GLM implementation and the official
GLM5Next reference. Native parity is required before this becomes a trainer.
The tensor reader and EXL3 tile decoder are supplied by the owning checkpoint.
"""

import math
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def rmsnorm(inputs: Any, weight: Any, epsilon: Any) -> Any:
    values = inputs.float()
    scale = torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
    normalized = (values * scale).to(inputs.dtype)
    return (normalized.float() * weight.float()).to(inputs.dtype)


def hyper_pre(streams: Any, fn: Any, base: Any, scale: Any, norm: Any, epsilon: Any = 1e-5,
              hc_epsilon: Any = 1e-6, iterations: Any = 20) -> Any:
    """Compute all four-stream mixing coefficients, preserving input gradients."""
    values = streams.float()
    flattened = values.flatten(1)
    inverse = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + epsilon)
    mixing = F.linear(flattened, fn.float()) * inverse
    pre = torch.sigmoid(mixing[:, :4] * scale[0] + base[:4]) + hc_epsilon
    post = 2 * torch.sigmoid(mixing[:, 4:8] * scale[1] + base[4:8])
    logits = (mixing[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4)
    combine = logits.softmax(-1) + hc_epsilon
    combine = combine / (combine.sum(-2, keepdim=True) + hc_epsilon)
    for _ in range(iterations - 1):
        combine = combine / (combine.sum(-1, keepdim=True) + hc_epsilon)
        combine = combine / (combine.sum(-2, keepdim=True) + hc_epsilon)
    collapsed = (pre[:, 0, None] * values[:, 0] + pre[:, 1, None] * values[:, 1])
    collapsed = collapsed + pre[:, 2, None] * values[:, 2]
    collapsed = collapsed + pre[:, 3, None] * values[:, 3]
    return rmsnorm(collapsed.to(streams.dtype), norm, epsilon), post, combine


def hyper_post(streams: Any, branch: Any, post: Any, combine: Any) -> Any:
    """Mix residuals and an already reduced FFN or attention branch."""
    values = streams.float()
    outputs = []
    branch = branch.to(streams.dtype).float()
    for index in range(4):
        current = branch[:, index] if branch.ndim == 3 else branch
        mixed = values[:, 0] * combine[:, 0, index, None]
        mixed = mixed + values[:, 1] * combine[:, 1, index, None]
        mixed = mixed + values[:, 2] * combine[:, 2, index, None]
        mixed = mixed + values[:, 3] * combine[:, 3, index, None]
        outputs.append((mixed + post[:, index, None] * current).to(streams.dtype))
    return torch.stack(outputs, dim=1)


def rotate128(inputs: Any) -> Any:
    """Normalized block Hadamard rotation, expressed with differentiable adds."""
    shape = inputs.shape
    if shape[-1] % 128:
        raise ValueError('EXL3 rotations require complete 128-channel blocks')
    values = inputs.float().reshape(*shape[:-1], -1, 128)
    step = 1
    while step < 128:
        groups = values.reshape(*values.shape[:-1], -1, 2, step)
        left = groups[..., 0, :]
        right = groups[..., 1, :]
        values = torch.stack((left + right, left - right), dim=-2).reshape(values.shape)
        step *= 2
    return (values / math.sqrt(128)).reshape(shape)


def exl3_linear(inputs: Any, rotated_weight: Any, input_scale: Any, output_scale: Any) -> Any:
    """Differentiate activations through a fixed decoded EXL3 matrix."""
    rotated = rotate128(inputs.float() * input_scale.float()).to(torch.float16)
    multiplied = rotated.float() @ rotated_weight.float()
    return rotate128(multiplied) * output_scale.float()


class FrozenPackedLinear(torch.autograd.Function):
    # PyTorch supplies Function.apply without a typed call signature.
    """Exact input VJP of fixed EXL3 basis arithmetic, with streamed weights."""

    @staticmethod
    def forward(context: Any, inputs: Any, loader: Any) -> Any:
        weight, input_scale, output_scale = loader()
        context.loader = loader
        context.save_for_backward(inputs)
        return exl3_linear(inputs, weight, input_scale, output_scale)

    @staticmethod
    def backward(context: Any, gradient: Any) -> Any:
        weight, input_scale, output_scale = context.loader()
        with torch.enable_grad():
            inputs = context.saved_tensors[0].detach().requires_grad_(True)
            output = exl3_linear(inputs, weight, input_scale, output_scale)
            input_gradient = torch.autograd.grad(output, inputs, gradient)[0]
        return input_gradient, None


def kda_recurrence(query: Any, key: Any, value: Any, log_decay: Any, beta: Any, state: Any = None) -> Any:
    """Fresh causal delta rule with a real gradient through every token's state."""
    heads, channels = key.shape[1:]
    if state is None:
        state = torch.zeros(heads, channels, channels, dtype=query.dtype, device=query.device)
    outputs = []
    for position in range(len(query)):
        state = state * log_decay[position].exp()[:, None, :]
        predicted = (state * key[position][:, None, :]).sum(-1)
        delta = (value[position] - predicted) * beta[position][:, None]
        state = state + delta[:, :, None] * key[position][:, None, :]
        outputs.append((state * query[position][:, None, :]).sum(-1))
    return torch.stack(outputs), state


def kda_chunks(query: Any, key: Any, value: Any, log_decay: Any, beta: Any, chunk_size: Any = 32) -> Any:
    """Differentiable native WY chunks, with immutable forward substitution."""
    rows, heads, channels = query.shape
    state = torch.zeros(heads, channels, channels, dtype=query.dtype, device=query.device)
    outputs = []
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        count = stop - start
        padded = []
        for index, tensor in enumerate((query, key, value, log_decay.exp(), beta)):
            fill = 1 if index == 3 else 0
            tail = torch.full((chunk_size - count, *tensor.shape[1:]), fill,
                              dtype=tensor.dtype, device=tensor.device)
            padded.append(torch.cat((tensor[start:stop], tail)).transpose(0, 1))
        q, k, v, decay, strength = padded
        decay_rows = []
        transition = torch.zeros_like(decay)
        identity = torch.eye(chunk_size, dtype=query.dtype, device=query.device)
        for row in range(chunk_size):
            # Advance every earlier write by the same current decay.
            transition = transition * decay[:, row, None, :]
            transition = transition + identity[row][None, :, None]
            decay_rows.append(transition)
        transitions = torch.stack(decay_rows, dim=1)
        kk = torch.einsum('hsc,htc,hstc->hst', k, k, transitions)
        qk = torch.einsum('hsc,htc,hstc->hst', q, k, transitions)
        lower = torch.tril(strength[:, :, None] * kk, diagonal=-1)
        mixing = torch.tril(qk)
        inverse_rows: list[Any] = []
        for row in range(chunk_size):
            solved = identity[row].expand(heads, -1)
            if row:
                previous = torch.stack(inverse_rows, dim=1)
                solved = solved - (lower[:, row, :row, None] * previous).sum(1)
            inverse_rows.append(solved)
        inverse = torch.stack(inverse_rows, dim=1)
        forward_decay = decay.cumprod(1)
        backward_decay = decay[:, 1:].flip(1).cumprod(1).flip(1)
        backward_decay = torch.cat((backward_decay, torch.ones_like(decay[:, :1])), dim=1)
        entering = state.transpose(1, 2)
        target = strength[:, :, None] * (v - (k * forward_decay) @ entering)
        writes = inverse @ target
        output = (q * forward_decay) @ entering + mixing @ writes
        outputs.append(output[:, :count].transpose(0, 1))
        state = state * forward_decay[:, -1:, :] + writes.transpose(1, 2) @ (k * backward_decay)
    return torch.cat(outputs), state


def fp8_rows(inputs: Any) -> Any:
    """Native power-of-two FP8 cache rows, with PyTorch's cast derivative."""
    maximum = inputs.detach().float().abs().amax(-1).contiguous()
    bits = maximum.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).to(torch.int32)
    scale = ((exponent - 8).clamp(1, 254) << 23).view(torch.float32)
    return (inputs.float() / scale[:, None]).to(torch.float8_e4m3fn).float() * scale[:, None]


class FrozenSuffix:
    """Lazy frozen tensors avoid creating a second full 320B model instance."""

    def __init__(self, reader: Any, unpack: Any, config: Any, device: Any = 'cuda') -> None:
        self.reader = reader
        self.unpack = unpack
        self.config = config
        self.device = device
        self.prefix = 'model.language_model.'
        linear = config['linear_attn_config']
        self.heads = linear['num_heads']
        self.channels = linear['head_dim']
        self.full_attention = set(linear['full_attn_layers'])
        self.epsilon = config['rms_norm_eps']

    def tensor(self, name: Any) -> Any:
        tensor = self.reader(name).to(self.device)
        if tensor.requires_grad:
            raise ValueError('Frozen checkpoint tensor has gradients enabled')
        return tensor

    def linear(self, name: Any, inputs: Any, packed: Any = False, output_float: Any = False) -> Any:
        if packed:
            if output_float:
                width = inputs.shape[-1] // 2
                def load_left() -> Any:
                    weight = self.unpack(self.reader(name + '.trellis')).to(self.device)
                    return weight[:width], self.tensor(name + '.suh')[:width], self.tensor(name + '.svh')
                def load_right() -> Any:
                    weight = self.unpack(self.reader(name + '.trellis')).to(self.device)
                    return weight[width:], self.tensor(name + '.suh')[width:], self.tensor(name + '.svh')
                left = FrozenPackedLinear.apply(inputs[:, :width], load_left)  # type: ignore[no-untyped-call]
                right = FrozenPackedLinear.apply(inputs[:, width:], load_right)  # type: ignore[no-untyped-call]
                return left + right
            def load_weight() -> Any:
                weight = self.unpack(self.reader(name + '.trellis')).to(self.device)
                return weight, self.tensor(name + '.suh'), self.tensor(name + '.svh')
            return FrozenPackedLinear.apply(inputs, load_weight)  # type: ignore[no-untyped-call]
        weight = self.tensor(name + '.weight')
        if output_float:
            width = inputs.shape[-1] // 2
            left = F.linear(inputs[:, :width].float(), weight[:, :width].float())
            right = F.linear(inputs[:, width:].float(), weight[:, width:].float())
            return left + right
        return F.linear(inputs.float(), weight.float()).to(inputs.dtype)

    def hyper(self, streams: Any, layer: Any, site: Any, norm: Any) -> Any:
        prefix = self.prefix + 'layers.' + str(layer) + '.'
        return hyper_pre(streams, self.tensor(prefix + 'hc_' + site + '_fn'),
                         self.tensor(prefix + 'hc_' + site + '_base'),
                         self.tensor(prefix + 'hc_' + site + '_scale'),
                         self.tensor(prefix + norm + '.weight'), self.epsilon,
                         self.config['hc_eps'], self.config['hc_sinkhorn_iters'])

    def kda(self, inputs: Any, layer: Any) -> Any:
        prefix = self.prefix + 'layers.' + str(layer) + '.self_attn.'
        rows = len(inputs)
        width = self.heads * self.channels
        convolved = []
        for name in ('q', 'k', 'v'):
            projected = self.linear(prefix + name + '_proj', inputs)
            weight = self.tensor(prefix + name + '_conv1d.weight').reshape(width, -1).float()
            history = F.pad(projected.float(), (0, 0, weight.shape[-1] - 1, 0))
            output = torch.zeros_like(projected.float())
            for tap in range(weight.shape[-1]):
                output = output + history[tap:tap + rows] * weight[:, tap]
            convolved.append(F.silu(output).to(inputs.dtype).float().reshape(rows, self.heads, self.channels))
        query, key, value = convolved
        query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        query = query / math.sqrt(self.channels)
        forget = self.linear(prefix + 'f_b_proj', self.linear(prefix + 'f_a_proj', inputs)).float()
        forget = forget + self.tensor(prefix + 'dt_bias').float()
        forget = forget.reshape(rows, self.heads, self.channels)
        rate = self.tensor(prefix + 'A_log').float().exp()[None, :, None]
        lower = self.config['linear_attn_config']['gate_lower_bound']
        log_decay = lower * torch.sigmoid(rate * forget)
        beta = torch.sigmoid(self.linear(prefix + 'b_proj', inputs).float()).to(inputs.dtype).float()
        output, _ = kda_chunks(query, key, value, log_decay, beta)
        output = output.to(inputs.dtype).float()
        inverse = torch.rsqrt(output.square().mean(-1, keepdim=True) + self.epsilon)
        output = output * inverse * self.tensor(prefix + 'o_norm.weight').float()
        gate = self.linear(prefix + 'g_b_proj', self.linear(prefix + 'g_a_proj', inputs))
        output = output * torch.sigmoid(gate.float().reshape(rows, self.heads, self.channels))
        return self.linear(prefix + 'o_proj', output.to(inputs.dtype).reshape(rows, width), output_float=True)

    def dsa(self, inputs: Any, layer: Any) -> Any:
        if len(inputs) > self.config['index_topk']:
            raise ValueError('Reference suffix admits only full visible short-context attention')
        prefix = self.prefix + 'layers.' + str(layer) + '.self_attn.'
        heads = self.config['num_attention_heads']
        channels = self.config['qk_nope_head_dim']
        value_channels = self.config['v_head_dim']
        query = self.linear(prefix + 'q_a_proj', inputs)
        query = rmsnorm(query, self.tensor(prefix + 'q_a_layernorm.weight'), self.epsilon)
        query = self.linear(prefix + 'q_b_proj', query).reshape(-1, heads, channels)
        latent = self.linear(prefix + 'kv_a_proj_with_mqa', inputs)
        latent = rmsnorm(latent, self.tensor(prefix + 'kv_a_layernorm.weight'), self.epsilon)
        weight = self.tensor(prefix + 'kv_b_proj.weight')
        weight = weight.reshape(heads, channels + value_channels, -1)
        key_weight = weight[:, :channels].float()
        value_weight = weight[:, channels:].float()
        absorbed = torch.einsum('thd,hdl->thl', query.float(), key_weight).to(inputs.dtype)
        cached = fp8_rows(latent)
        scores = torch.einsum('thl,sl->hts', absorbed.float(), cached) / math.sqrt(channels)
        causal = torch.ones(len(inputs), len(inputs), dtype=torch.bool, device=inputs.device).tril()
        scores = scores.masked_fill(~causal[None], float('-inf'))
        attention = scores.softmax(-1)
        readout = torch.einsum('hts,sl->thl', attention, cached).to(inputs.dtype)
        output = torch.einsum('thl,hdl->thd', readout.float(), value_weight).to(inputs.dtype)
        return self.linear(prefix + 'o_proj', output.to(inputs.dtype).flatten(1), output_float=True)

    def expert(self, inputs: Any, prefix: Any, packed: Any = False) -> Any:
        gate = self.linear(prefix + 'gate_proj', inputs, packed).to(inputs.dtype).float().clamp(max=10)
        up = self.linear(prefix + 'up_proj', inputs, packed).to(inputs.dtype).float().clamp(-10, 10)
        activation = (F.silu(gate).to(inputs.dtype).float() * up).to(inputs.dtype)
        return self.linear(prefix + 'down_proj', activation, packed, output_float=True)

    def moe(self, inputs: Any, layer: Any) -> Any:
        prefix = self.prefix + 'layers.' + str(layer) + '.mlp.'
        router = self.tensor(prefix + 'gate.weight').to(torch.bfloat16).float()
        logits = F.linear(inputs.float(), router)
        scores = logits.sigmoid()
        correction = self.tensor(prefix + 'gate.e_score_correction_bias').float()
        selected = (scores + correction).topk(self.config['num_experts_per_tok'], dim=-1).indices
        weights = scores.gather(1, selected)
        weights = weights / weights.sum(-1, keepdim=True)
        weights = weights * self.config['routed_scaling_factor']
        output = self.expert(inputs, prefix + 'shared_experts.')
        for expert in selected.unique().tolist():
            positions, slots = torch.where(selected == expert)
            value = self.expert(inputs[positions], prefix + 'experts.' + str(expert) + '.', packed=True)
            weighted = value * weights[positions, slots, None]
            output = output.index_add(0, positions, weighted)
        return output

    def layer(self, streams: Any, index: Any) -> Any:
        inputs, post, combine = self.hyper(streams, index, 'attn', 'input_layernorm')
        branch = self.dsa(inputs, index) if index in self.full_attention else self.kda(inputs, index)
        streams = hyper_post(streams, branch, post, combine)
        inputs, post, combine = self.hyper(streams, index, 'ffn', 'post_attention_layernorm')
        return hyper_post(streams, self.moe(inputs, index), post, combine)

    def __call__(self, streams: Any, start: Any, checkpoint_layers: Any = True) -> Any:
        for index in range(start, self.config['num_hidden_layers']):
            if checkpoint_layers:
                def forward_layer(value: Any, layer: Any = index) -> Any:
                    return self.layer(value, layer)
                streams = checkpoint(forward_layer, streams, use_reentrant=False)
            else:
                streams = self.layer(streams, index)
        hidden = streams[:, 0].float()
        for index in range(1, 4):
            hidden = hidden + streams[:, index].float()
        hidden = (hidden / 4).to(streams.dtype)
        hidden = rmsnorm(hidden, self.tensor(self.prefix + 'norm.weight'), self.epsilon)
        return self.linear('lm_head', hidden).float()

    def batch_layer(self, sequences: Any, index: Any) -> Any:
        """Independent causal attention, with one grouped frozen MoE read."""
        residuals = []
        inputs = []
        posts = []
        combinations = []
        for streams in sequences:
            normalized, post, combine = self.hyper(streams, index, 'attn', 'input_layernorm')
            if index in self.full_attention:
                branch = self.dsa(normalized, index)
            else:
                branch = self.kda(normalized, index)
            residual = hyper_post(streams, branch, post, combine)
            normalized, post, combine = self.hyper(residual, index, 'ffn', 'post_attention_layernorm')
            residuals.append(residual)
            inputs.append(normalized)
            posts.append(post)
            combinations.append(combine)
        branches = self.moe(torch.cat(inputs), index)
        outputs = []
        offset = 0
        for number, residual in enumerate(residuals):
            count = len(residual)
            branch = branches[offset:offset + count]
            outputs.append(hyper_post(residual, branch, posts[number], combinations[number]))
            offset += count
        return tuple(outputs)

    def batch(self, sequences: Any, start: Any) -> Any:
        """Share fixed-weight traffic without sharing any sequence's attention state."""
        sequences = tuple(sequences)
        for index in range(start, self.config['num_hidden_layers']):
            def forward_layer(*values: Any, layer: Any = index) -> Any:
                return self.batch_layer(values, layer)
            sequences = checkpoint(forward_layer, *sequences, use_reentrant=False)
        hidden_rows = []
        for streams in sequences:
            hidden = streams[:, 0].float()
            for stream in range(1, 4):
                hidden = hidden + streams[:, stream].float()
            hidden_rows.append((hidden / 4).to(streams.dtype))
        hidden = torch.cat(hidden_rows)
        hidden = rmsnorm(hidden, self.tensor(self.prefix + 'norm.weight'), self.epsilon)
        logits = self.linear('lm_head', hidden).float()
        outputs = []
        offset = 0
        for sequence in sequences:
            count = len(sequence)
            outputs.append(logits[offset:offset + count])
            offset += count
        return outputs
