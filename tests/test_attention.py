import torch

from glm_rsi_dream.attention import attention


def test_real_attention_gradient_matches_directional_derivative_and_optimizes():
    torch.manual_seed(8)
    queries = torch.randn(3, 2, 4, dtype=torch.float64)
    teacher = torch.randn(5, 4, dtype=torch.float64)
    history = torch.randn(3, 4, dtype=torch.float64)
    positions = torch.tensor([0, 1, 2])
    target = attention(queries, teacher, history, positions).detach()
    memory = torch.nn.Parameter(torch.randn(3, 4, dtype=torch.float64))
    loss = (attention(queries, memory, history, positions) - target).square().mean()
    loss.backward()
    direction = torch.randn_like(memory)
    direction /= direction.norm()
    derivative = (memory.grad * direction).sum()
    epsilon = 1e-5
    plus = (attention(queries, memory + epsilon * direction, history, positions) - target).square().mean()
    minus = (attention(queries, memory - epsilon * direction, history, positions) - target).square().mean()
    numeric = (plus - minus) / (2 * epsilon)
    assert torch.allclose(derivative, numeric, rtol=1e-5, atol=1e-8)
    assert memory.grad.norm() > 0
    initial = float(loss.detach())
    optimizer = torch.optim.Adam([memory], lr=0.1)
    for _ in range(80):
        optimizer.zero_grad()
        loss = (attention(queries, memory, history, positions) - target).square().mean()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial / 10


def test_causal_history_masks_future_rows():
    queries = torch.zeros(2, 1, 2)
    memory = torch.tensor([[2., 4.]])
    history = torch.tensor([[4., 8.], [100., 200.]])
    output = attention(queries, memory, history, torch.tensor([0, 1]))
    assert torch.equal(output[0, 0], torch.tensor([3., 6.]))
    assert torch.allclose(output[1, 0], torch.tensor([106/3, 212/3]))
