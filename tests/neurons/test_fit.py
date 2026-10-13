"""Numerical checks of the FFN objective and its paired native geometry."""

import pytest
import torch

from glm_rsi_dream.neurons.fit import (
    AddedSharedNeurons,
    fit_packets,
    fit_site,
    partition_shared,
    projected_correction,
)


def test_projection_recovers_expressible_two_rank_correction():
    torch.manual_seed(4)
    student = torch.randn(6, 4, 8)
    post = torch.rand(6, 4) + 0.2
    delta = torch.randn(6, 8)
    teacher = student + post.unsqueeze(-1) * delta.unsqueeze(1)
    correction, unexplained = projected_correction(student, teacher, post)
    torch.testing.assert_close(correction * 2, delta)
    assert float(unexplained.abs().max()) < 1e-6


def test_new_ffn_derivative_matches_finite_difference():
    torch.manual_seed(3)
    model = AddedSharedNeurons(4, 6).double()
    with torch.no_grad():
        model.down.weight.normal_()
        model.gate.weight.fill_(0.2)
        model.up.weight.fill_(0.3)
    inputs = torch.randn(2, 4, dtype=torch.float64)
    targets = torch.randn(2, 4, dtype=torch.float64)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    predicted = float(model.gate.weight.grad[1, 2])
    epsilon = 1e-5
    with torch.no_grad():
        original = float(model.gate.weight[1, 2])
        model.gate.weight[1, 2] = original + epsilon
        upper = float((model(inputs) - targets).square().mean())
        model.gate.weight[1, 2] = original - epsilon
        lower = float((model(inputs) - targets).square().mean())
    assert predicted == pytest.approx((upper - lower) / (2 * epsilon), rel=1e-6)


def test_gradient_fit_learns_corrections_and_limits_negative_changes():
    inputs = torch.eye(8)
    inputs = torch.cat((inputs, -inputs))
    targets = torch.zeros_like(inputs)
    targets[:8] = inputs[:8] * 2
    positive = torch.arange(16) < 8
    site, report = fit_site(inputs, targets, positive, width=16, steps=1000, rate=0.02)
    gate = (inputs @ site["gate"].T).clamp(max=10)
    up = (inputs @ site["up"].T).clamp(-10, 10)
    output = (torch.nn.functional.silu(gate) * up) @ site["down"].T
    assert report["relative_correction_error"] < 0.03
    assert float(output[~positive].abs().max()) < 0.05
    assert report["negative_max_correction"] < 0.05


def test_fit_rejects_missing_preservation_rows():
    with pytest.raises(ValueError, match="preservation"):
        fit_site(torch.eye(4), torch.eye(4), torch.ones(4, dtype=torch.bool))


def test_fit_rejects_relabeled_rank_before_optimization(tmp_path):
    packet = tmp_path / 'training.pt'
    torch.save({'schema_version': 1, 'role': 'train', 'rank': 1, 'world': 2}, packet)
    with pytest.raises(ValueError, match='rank/world'):
        fit_packets(packet, tmp_path / 'neurons.pt', 0, device='cpu')


def test_joint_shared_expert_matches_the_sum_of_its_native_rank_partitions():
    torch.manual_seed(11)
    model = AddedSharedNeurons(8, 16)
    with torch.no_grad():
        model.down.weight.normal_()
    inputs = torch.randn(7, 8)
    site = {'gate': model.gate.weight, 'up': model.up.weight, 'down': model.down.weight}
    actual = torch.zeros_like(inputs)
    for rank in range(2):
        part = partition_shared(site, rank)
        gate = (inputs @ part['gate'].T).clamp(max=10)
        up = (inputs @ part['up'].T).clamp(-10, 10)
        actual += (torch.nn.functional.silu(gate) * up) @ part['down'].T
    torch.testing.assert_close(actual, model(inputs), atol=1e-6, rtol=1e-6)


def test_question_weighting_rejects_a_negative_row_as_a_question_target():
    positive = torch.tensor([True, True, False, False])
    first = torch.tensor([True, False, True, False])
    with pytest.raises(ValueError, match='Bare-question'):
        fit_site(torch.eye(4), torch.eye(4), positive, first_question=first)
