import random

from bpe.tokenizer import decode, encode, train, vocab

# The word-frequency example from Sennrich et al. (2016), written out as text:
# low x5, lower x2, newest x6, widest x3.
TOY = " ".join(["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3)


def test_known_merge_sequence_and_an_unseen_word():
    merges = train(TOY, 4)
    table = vocab(merges)
    # (e,s) and (s,t) both occur 9 times (newest 6 + widest 3); the tie goes to
    # the smaller pair, (e,s). Then "es"+"t" (9), then (l,o) and (o,w) tie at 7
    # (low 5 + lower 2), so "lo", then "lo"+"w".
    assert [table[i] for i in merges.values()] == [b"es", b"est", b"lo", b"low"]
    # "lowest" never appears in the training text, but its pieces do.
    assert [table[i] for i in encode("lowest", merges)] == [b"low", b"est"]
    assert [table[i] for i in encode("lower", merges)] == [b"low", b"e", b"r"]


def test_round_trip_for_any_utf8_text():
    merges = train(TOY + " 한국어 토큰 비용 😀 café", 40)
    samples = ["", "lowest lower", "한국어는 토큰을 더 쓴다.", "emoji 😀👍🏽 and 漢字",
               "tabs\tand\n\nnewlines  ", "def f(x):\n    return x ** 2\n"]
    rng = random.Random(0)
    for _ in range(200):
        # Random code points from every plane, skipping surrogates (not valid in UTF-8).
        cps = [rng.choice([rng.randrange(0x20, 0x80), rng.randrange(0xAC00, 0xD7A4),
                           rng.randrange(0x80, 0xD800), rng.randrange(0xE000, 0x110000)])
               for _ in range(rng.randrange(1, 30))]
        samples.append("".join(map(chr, cps)))
    for text in samples:
        assert decode(encode(text, merges), merges) == text
        assert decode(encode(text, {}), {}) == text


def test_english_merges_never_join_hangul_bytes():
    # Every Hangul syllable is 3 bytes in UTF-8.
    assert {len(chr(cp).encode("utf-8")) for cp in range(0xAC00, 0xD7A4)} == {3}
    english_only = train(TOY * 3, 30)
    mixed = train(TOY + " 한국어" * 20, 30)
    assert len(encode("한국어", english_only)) == 9  # 3 syllables, 9 raw bytes
    assert len(encode("한국어", mixed)) < 9
