
import torch
import torch.nn.functional as F

from charformer.data import batch, load
from charformer.model import (Block, Config, CharTransformer, SelfAttention, attention_weights,
                              causal_mask, generate)

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


def tiny_config(**kw):
    return Config(**{"vocab_size": 20, "context": 16, "d_model": 32, "n_heads": 4, "n_layers": 2, **kw})


def test_self_attention_matches_torch_sdpa():
    # Reference: split the qkv projection into q | k | v, each into 4 heads of
    # 8 columns, and let torch's own causal attention (scale 1/sqrt(8)) do the rest.
    torch.manual_seed(0)
    cfg = tiny_config()
    attn = SelfAttention(cfg).double()
    x = torch.randn(2, 10, 32, dtype=torch.float64)
    b, t, d, h = 2, 10, 32, 4
    with torch.no_grad():
        q, k, v = attn.qkv(x).split(d, dim=-1)
        q, k, v = (m.view(b, t, h, d // h).transpose(1, 2) for m in (q, k, v))
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref = attn.out(ref.transpose(1, 2).reshape(b, t, d))
        got = attn(x)
    assert torch.allclose(got, ref, atol=1e-6, rtol=0)


def test_positions_change_the_output():
    # Without a position embedding, a run of one repeated character gives
    # every position the same inputs to attend over, so every output row
    # would be the same. With it, the rows differ.
    torch.manual_seed(0)
    model = CharTransformer(tiny_config()).eval()
    with torch.no_grad():
        logits, _ = model(torch.full((1, 8), 5))
    assert ((logits[0, 1:] - logits[0, 0]).abs().amax(-1) > 1e-3).all()


def test_block_adds_each_sublayer_to_its_input():
    torch.manual_seed(0)
    block = Block(tiny_config()).eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        # x + attn(ln1(x)), then that + mlp(ln2(that)).
        mid = x + block.attn(block.ln1(x))
        assert torch.allclose(block(x), mid + block.mlp(block.ln2(mid)), atol=1e-6)
        # With both sublayers' last projections at zero, the block is the identity.
        for layer in (block.attn.out, block.mlp[2]):
            layer.weight.zero_()
            layer.bias.zero_()
        assert torch.equal(block(x), x)


def test_loss_is_mean_cross_entropy_of_next_characters():
    # -log softmax(logits)[target], averaged over all 2 x 12 positions.
    torch.manual_seed(0)
    model = CharTransformer(tiny_config()).eval()
    g = torch.Generator().manual_seed(1)
    x = torch.randint(20, (2, 12), generator=g)
    y = torch.randint(20, (2, 12), generator=g)
    with torch.no_grad():
        logits, loss = model(x, y)
    expected = -torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).mean()
    assert logits.shape == (2, 12, 20)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_greedy_generate_appends_the_argmax_of_the_last_position():
    torch.manual_seed(0)
    model = CharTransformer(tiny_config()).eval()
    prompt = torch.tensor([[3, 1, 4]])
    out = generate(model, prompt, n_new=6, temperature=0)
    assert out.shape == (1, 9)
    assert torch.equal(out[:, :3], prompt)
    seq = prompt
    with torch.no_grad():
        for _ in range(6):
            nxt = model(seq)[0][0, -1].argmax()
            seq = torch.cat([seq, nxt.view(1, 1)], dim=1)
    assert torch.equal(out, seq)


def test_batches_stay_inside_the_data():
    # 4,000 windows of 8 from arange(100): the value is the position, so
    # x[:, 0] is the start. Every window is whole and in range, y is x + 1,
    # and start 0 shows up (chance it is missed: (90/91)^4000, about e^-44).
    ids = torch.arange(100)
    x, y = batch(ids, 4000, 8, torch.Generator().manual_seed(0))
    assert x.shape == (4000, 8) and y.shape == (4000, 8)
    assert torch.equal(x, x[:, :1] + torch.arange(8))
    assert torch.equal(y, x + 1)
    assert x.min() == 0 and y.max() <= 99


def test_load_keeps_the_last_tenth_for_validation(tmp_path):
    # 100 characters: int(100 * 0.9) = 90 for training, the last 10 for validation.
    text = "".join(chr(ord("a") + i % 26) for i in range(100))
    path = tmp_path / "book.txt"
    path.write_text(text, encoding="utf-8")
    chars, train, val = load(path)
    assert len(train) == 90 and len(val) == 10
    assert chars.decode(train.tolist()) == text[:90]
    assert chars.decode(val.tolist()) == text[90:]
