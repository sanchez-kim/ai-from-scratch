import random
import re

from bpe.tokenizer import chunks, decode, encode, train, vocab

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
    korean = "한국어"
    raw = list(korean.encode("utf-8"))
    english_only = train(TOY * 3, 30)
    mixed = train(TOY + " 한국어" * 20, 30)
    # No Hangul byte pair is in the English text, so no merge touches a byte >= 0x80
    # and the Korean word stays as its 9 raw bytes.
    english_table = vocab(english_only)
    assert all(max(english_table[i]) < 0x80 for i in english_only.values())
    assert encode(korean, english_only) == raw
    # Mixed training learns merges inside the Hangul bytes. At least one merge is
    # a 2-byte piece of a single syllable, so Korean gets cheaper.
    mixed_table = vocab(mixed)
    syllables = [ch.encode("utf-8") for ch in korean]
    pieces = [mixed_table[i] for i in mixed.values()]
    assert any(len(b) == 2 and b[0] >= 0x80 and any(b in syl for syl in syllables)
               for b in pieces)
    assert len(encode(korean, mixed)) < 9


def test_encode_applies_the_earliest_learned_merge_first():
    a, b, c = 97, 98, 99
    # Hand-built merges: (a,b) was learned first (256), (b,c) second (257).
    # "abc" has both pairs. Earliest first: (a,b) -> [256, c]. Then no pair is left.
    # Applying the later merge first would give [a, 257] instead.
    merges = {(a, b): 256, (b, c): 257}
    assert encode("abc", merges) == [256, c]
    # With a third merge (256,c) -> 258 the earliest-first path continues: [258].
    assert encode("abc", {(a, b): 256, (b, c): 257, (256, c): 258}) == [258]
    # Trained version, run by hand: "abab" -> pair (a,b) x2 is most common -> 256,
    # chunk becomes [256, 256]; pair (256,256) x1 -> 257. Zero-count pairs
    # (a,b) and (b,a) must be dropped, so training stops after 2 merges.
    trained = train("abab", 10)
    assert trained == {(a, b): 256, (256, 256): 257}
    assert encode("abab", trained) == [257]
    assert encode("ab", trained) == [256]


def test_training_stops_when_no_pair_is_left():
    # "aa" has one pair, (a,a). After it merges the chunk is a single token and
    # no pair remains, so asking for 10 merges must learn exactly 1.
    assert train("aa", 10) == {(97, 97): 256}


def test_chunks_split_on_spaces_and_punctuation():
    # Hand-run of the pattern: a word keeps one leading space, punctuation runs
    # are their own chunk, whitespace before a non-space is split off but the
    # last space stays with the next word.
    assert chunks("a, b.\n c") == ["a", ",", " b", ".", "\n", " c"]
    assert chunks("hi  there") == ["hi", " ", " there"]
    assert "".join(chunks("a, b.\n c  \n")) == "a, b.\n c  \n"


def test_no_merge_spans_a_space_or_punctuation_boundary():
    text = "the cat, the cat. the cat! (the cat) " * 20
    merges = train(text, 50)
    table = vocab(merges)
    assert merges  # something was learned
    for new_id in merges.values():
        token = table[new_id].decode("utf-8")
        # One word, or one run of punctuation, with at most one leading space.
        assert re.fullmatch(r" ?[A-Za-z]+| ?[^A-Za-z\s]+", token), repr(token)
