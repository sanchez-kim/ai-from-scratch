
import torch

from charformer.data import batch
from charformer.model import Config, CharTransformer, attention_weights, causal_mask

INF = float("inf")


def random_scores(seed, t=6):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 3, t, t, generator=g, dtype=torch.float64) * 3


def test_mask_fills_the_future_with_minus_inf():
    assert causal_mask(4).tolist() == [
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False],
    ]
    scores = torch.tensor([[2.0, 1.0, 0.5, 3.0]] * 4)
    masked = scores.masked_fill(causal_mask(4), -INF)
    assert masked[0].tolist() == [2.0, -INF, -INF, -INF]
    assert masked[2].tolist() == [2.0, 1.0, 0.5, -INF]
    # Row 0 has one finite score, so softmax gives it all the weight.
    assert attention_weights(scores, "causal")[0].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_causal_rows_sum_to_one_and_future_is_exactly_zero():
    w = attention_weights(random_scores(0), "causal")
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 6, dtype=torch.float64), atol=1e-12)
    assert (w[..., causal_mask(6)] == 0).all()   # exactly 0, not just small
    assert (w[..., ~causal_mask(6)] > 0).all()   # the past keeps some weight


def test_mask_after_softmax_leaves_rows_that_do_not_sum_to_one():
    sums = attention_weights(random_scores(0), "after").sum(-1)
    # Only the last row, which has no future, still adds up to 1.
    assert torch.allclose(sums[..., -1], torch.ones(2, 3, dtype=torch.float64), atol=1e-12)
    assert (sums[..., :-1] < 1 - 1e-3).all()
    # This is the check that catches the bug in real code.
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_no_mask_rows_sum_to_one_but_read_the_future():
    w = attention_weights(random_scores(0), "none")
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 6, dtype=torch.float64), atol=1e-12)
    assert (w[..., causal_mask(6)] > 0).all()


def outputs_before_and_after_changing_the_future(mask, t=10):
    torch.manual_seed(0)
    model = CharTransformer(Config(vocab_size=20, context=16, d_model=32, n_heads=4, n_layers=2, mask=mask)).eval()
    g = torch.Generator().manual_seed(1)
    a = torch.randint(20, (1, 16), generator=g)
    b = a.clone()
    b[0, t + 1:] = torch.randint(20, (16 - t - 1,), generator=g)  # new tokens after t only
    assert (a[0, t + 1:] != b[0, t + 1:]).any()
    with torch.no_grad():
        return model(a)[0][0, :t + 1], model(b)[0][0, :t + 1]


def test_output_at_t_ignores_later_tokens_only_with_the_mask():
    before, after = outputs_before_and_after_changing_the_future("causal")
    assert torch.equal(before, after)  # bit for bit: the future adds exact zeros

    before, after = outputs_before_and_after_changing_the_future("none")
    assert (before - after).abs().max() > 1e-3

    # Masking after softmax does not stop the leak either: future keys still
    # sit in the softmax denominator and change the weights on the past.
    before, after = outputs_before_and_after_changing_the_future("after")
    assert (before - after).abs().max() > 1e-3


def test_targets_are_inputs_shifted_by_one():
    ids = torch.arange(100)
    x, y = batch(ids, 4, 8, torch.Generator().manual_seed(0))
    assert torch.equal(y, x + 1)
    assert x.shape == (4, 8)
