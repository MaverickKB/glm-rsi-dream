"""Check causal-state derivatives and reversible EXL3 basis arithmetic."""

import torch

from glm_rsi_dream.neurons.reference import (
    FrozenPackedLinear,
    FrozenSuffix,
    exl3_linear,
    kda_chunks,
    kda_recurrence,
    rotate128,
)


def test_reference_recurrence_backpropagates_from_later_answer_to_earlier_write():
    torch.manual_seed(23)
    query = torch.randn(4, 2, 3, dtype=torch.float64)
    key = torch.randn_like(query) * .1
    value = torch.randn_like(query, requires_grad=True)
    decay = torch.full_like(query, -.2)
    beta = torch.full((4, 2), .5, dtype=torch.float64)
    output, _ = kda_recurrence(query, key, value, decay, beta)
    objective = output[-1].square().sum()
    gradient = torch.autograd.grad(objective, value)[0]
    direction = torch.zeros_like(value)
    direction[0] = torch.randn_like(direction[0])
    epsilon = 1e-5
    positive, _ = kda_recurrence(query, key, value.detach() + epsilon * direction, decay, beta)
    negative, _ = kda_recurrence(query, key, value.detach() - epsilon * direction, decay, beta)
    numerical = (positive[-1].square().sum() - negative[-1].square().sum()) / (2 * epsilon)
    analytic = (gradient * direction).sum()
    torch.testing.assert_close(analytic, numerical, rtol=1e-6, atol=1e-9)
    assert gradient[0].norm() > 0


def test_chunked_state_and_gradient_match_the_causal_delta_rule():
    torch.manual_seed(28)
    query = torch.randn(7, 2, 3, dtype=torch.float64) * .2
    key = torch.randn_like(query) * .1
    value = torch.randn_like(query, requires_grad=True)
    decay = torch.full_like(query, -.2)
    beta = torch.full((7, 2), .5, dtype=torch.float64)
    expected, expected_state = kda_recurrence(query, key, value, decay, beta)
    actual, actual_state = kda_chunks(query, key, value, decay, beta, chunk_size=4)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-12, atol=1e-12)
    expected_gradient = torch.autograd.grad(expected[-1].sum(), value)[0]
    actual_gradient = torch.autograd.grad(actual[-1].sum(), value)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_chunked_backward_matches_recurrence_for_every_state_input():
    torch.manual_seed(42)
    inputs = []
    for index in range(4):
        value = torch.randn(9, 2, 3, dtype=torch.float64) * .1
        if index == 3:
            value = -value.abs() - .01
        inputs.append(value.requires_grad_())
    inputs.append(torch.full((9, 2), .5, dtype=torch.float64, requires_grad=True))
    expected, expected_state = kda_recurrence(*inputs)
    actual, actual_state = kda_chunks(*inputs, chunk_size=4)
    expected_gradients = torch.autograd.grad(expected[-1].square().sum() + expected_state.square().sum(), inputs)
    actual_gradients = torch.autograd.grad(actual[-1].square().sum() + actual_state.square().sum(), inputs)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-13)


def test_block_rotation_is_involutive_and_preserves_input_gradient():
    torch.manual_seed(24)
    inputs = torch.randn(3, 256, requires_grad=True)
    restored = rotate128(rotate128(inputs))
    torch.testing.assert_close(restored, inputs, atol=1e-6, rtol=1e-5)
    restored.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))


def test_fixed_exl3_projection_has_input_gradient_and_frozen_weights():
    torch.manual_seed(25)
    inputs = torch.randn(2, 128, requires_grad=True)
    weight = torch.randn(128, 128, dtype=torch.float16)
    scale = torch.ones(128, dtype=torch.float16)
    expected = weight.clone()
    output = exl3_linear(inputs, weight, scale, scale)
    output.square().mean().backward()
    assert inputs.grad is not None and inputs.grad.norm() > 0
    assert weight.grad is None
    assert torch.equal(weight, expected)


def test_streamed_packed_vjp_matches_eager_autograd():
    torch.manual_seed(26)
    inputs = torch.randn(2, 256, requires_grad=True)
    weight = torch.randn(256, 128, dtype=torch.float16)
    input_scale = torch.randn(256, dtype=torch.float16)
    output_scale = torch.randn(128, dtype=torch.float16)
    eager = exl3_linear(inputs, weight, input_scale, output_scale)
    gradient = torch.randn_like(eager)
    expected = torch.autograd.grad(eager, inputs, gradient)[0]
    streamed = FrozenPackedLinear.apply(inputs, lambda: (weight, input_scale, output_scale))
    actual = torch.autograd.grad(streamed, inputs, gradient)[0]
    torch.testing.assert_close(streamed, eager, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_batched_suffix_preserves_independent_attention_and_input_gradients():
    class SmallSuffix(FrozenSuffix):
        def __init__(self):
            self.config = {'num_hidden_layers': 2}
            self.full_attention = {1}
            self.prefix = 'fixture.'
            self.epsilon = 1e-5
        def hyper(self, streams, index, site, norm):
            rows = len(streams)
            return streams.mean(1), torch.ones(rows, 4), torch.eye(4).expand(rows, -1, -1)
        def kda(self, inputs, index):
            return inputs.cumsum(0) * .1
        def dsa(self, inputs, index):
            counts = torch.arange(1, len(inputs) + 1)[:, None]
            return inputs.cumsum(0) / counts
        def moe(self, inputs, index):
            return inputs.sin() * .2
        def tensor(self, name):
            return torch.ones(3)
        def linear(self, name, inputs):
            return inputs @ torch.tensor([[1., 2.], [3., 1.], [2., -1.]])
    torch.manual_seed(39)
    suffix = SmallSuffix()
    sequences = [torch.randn(3, 4, 3, requires_grad=True),
                 torch.randn(5, 4, 3, requires_grad=True)]
    expected = [suffix(sequence, 0) for sequence in sequences]
    actual = suffix.batch(sequences, 0)
    for one, two in zip(actual, expected, strict=True):
        torch.testing.assert_close(one, two)
    gradients = torch.autograd.grad(actual[0].sum(), sequences, allow_unused=True)
    torch.testing.assert_close(gradients[0], torch.autograd.grad(expected[0].sum(), sequences[0])[0])
    assert gradients[1] is None or torch.count_nonzero(gradients[1]) == 0
