"""Write figures/bpe-merges-v2.svg: the animated merge diagram.

    uv run python figures/make_merge_animation.py

Part 1 trains the toy text from tests/test_tokenizer.py and shows "lowest"
turning from 6 bytes into low + est, one merge at a time. Part 2 shows the
Korean word 한국어 (9 bytes) under the two trained tokenizers: English-only
merges leave 9 raw bytes; mixed merges turn each syllable into one token.
All token boundaries come from the real tokenizers, not from hand-written lists.

v2 colours mark where a token came from: blue = merged from English text,
orange = merged from Korean text (series palette v2: English slot 1, Korean
slot 2). Raw single bytes stay neutral outlines. The merge that just happened
gets a dark outline.

CSS animation only (no script, no SMIL), so it plays inside an <img> tag.
Every element's static attributes are the final frame, so
prefers-reduced-motion switches the animations off on the final frame.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from bpe.tokenizer import encode, merge, train, vocab  # noqa: E402

TOY = " ".join(["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3)
W, H, X0 = 460, 352, 20
DUR = 16.0  # seconds per loop


def pct(seconds):
    return round(100 * seconds / DUR, 2)


def window(name, start, end, fade=0.3):
    """Keyframes that show an element from `start` to `end` seconds of the loop."""
    frames = []
    if start <= 0:
        frames += [(0, 1)]
    else:
        frames += [(0, 0), (pct(start - fade), 0), (pct(start), 1)]
    if end >= DUR:
        frames += [(100, 1)]
    else:
        frames += [(pct(end), 1), (pct(end + fade), 0), (100, 0)]
    body = " ".join(f"{p}% {{opacity:{o}}}" for p, o in frames)
    return f"@keyframes {name} {{ {body} }} #{name}{{animation-name:{name}}}"


def merge_states(word_bytes, merges):
    """Token sequences after each merge that changes this word, in learning order."""
    seq = list(word_bytes)
    states = [(seq, None)]
    for pair, new_id in sorted(merges.items(), key=lambda kv: kv[1]):
        nxt = merge(seq, pair, new_id)
        if nxt != seq:
            seq = nxt
            states.append((seq, new_id))
    return states


def row(ids, table, y, h, unit, new_id=None, label=None, lang="en"):
    parts, x = [], X0
    for i in ids:
        b = table[i]
        w = len(b) * unit - 4
        cls = (lang if len(b) > 1 else "byt") + (" cur" if i == new_id else "")
        parts.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="3"/>')
        text = label(b) if label else b.decode("utf-8")
        tcls = "tw" if len(b) > 1 else "ink"
        parts.append(f'<text class="{tcls}" x="{x + w / 2:.1f}" y="{y + h / 2 + 5:.1f}" '
                     f'text-anchor="middle" font-size="14">{text}</text>')
        x += len(b) * unit
    return "".join(parts)


def text(x, y, s, cls="ink", size=13, weight=400, ident=None, opacity=1):
    i = f' id="{ident}" class="a {cls}"' if ident else f' class="{cls}"'
    return (f'<text{i} x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'opacity="{opacity}">{s}</text>')


def korean_label(b):
    try:
        s = b.decode("utf-8")
        return f"{s} {b.hex(' ').upper()}" if len(b) == 3 else s
    except UnicodeDecodeError:
        return b.hex(" ").upper()


def load_merges(path):
    merges = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        a, b, new = map(int, line.split())
        merges[(a, b)] = new
    return merges


def main():
    counts = json.loads((HERE / "results" / "token-counts.json").read_text(encoding="utf-8"))
    example = counts["example"]
    word = example["word"]

    toy = train(TOY, 4)
    toy_table = vocab(toy)
    a_states = merge_states(b"lowest", toy)
    assert [toy_table[i] for i in a_states[-1][0]] == [b"low", b"est"]
    assert encode("lowest", toy) == a_states[-1][0]

    en = load_merges(HERE / "results" / "merges-en.txt")
    mixed = load_merges(HERE / "results" / "merges-mixed.txt")
    assert len(encode(word, en)) == len(example["tokens"]["ours-en"]) == 9
    k_states = merge_states(word.encode("utf-8"), mixed)
    assert k_states[-1][0] == encode(word, mixed)
    mixed_table = vocab(mixed)
    # Show the start and each moment a whole syllable becomes one token.
    k_show = [(k_states[0][0], 0)] + [(ids, new - 255) for ids, new in k_states[1:]
                                      if len(mixed_table[new]) == 3]
    assert [len(ids) for ids, _ in k_show] == [9, 7, 5, 3]
    en_table = vocab(en)

    css_anim = []
    body = []

    # Part 1: lowest, 0 to 6.5 s then held.
    body.append(text(X0, 24, "1 · Merges learned from a tiny English text", size=15, weight=600))
    steps = [0, 1.5, 3.0, 4.5, 6.0]
    captions = ["start: 6 bytes, 6 tokens"] + [
        f"merge {n}: {toy_table[pair[0]].decode()} + {toy_table[pair[1]].decode()} → "
        f"{toy_table[new].decode()}"
        for n, (pair, new) in enumerate(toy.items(), start=1)]
    for n, (ids, new_id) in enumerate(a_states):
        start, end = steps[n], (steps[n + 1] if n + 1 < len(steps) else DUR)
        last = n == len(a_states) - 1
        ident = f"a{n}"
        body.append(f'<g id="{ident}" class="a" opacity="{1 if last else 0}">'
                    + row(ids, toy_table, 36, 34, 44, new_id)
                    + text(X0, 94, captions[n], size=14) + "</g>")
        css_anim.append(window(ident, start, end))
    body.append(text(X0, 116, '"lowest" is not in the text. It still gets low + est.',
                      "mut", 13, ident="a_note", opacity=1))
    css_anim.append(window("a_note", 6.0, DUR))

    # Part 2: 한국어, appears at 7 s.
    sec = ['<g id="b" class="a">']
    sec.append(text(X0, 152, f"2 · {word} is 9 bytes in UTF-8", size=15, weight=600))
    sec.append(text(X0, 176, "Merges from English only: 9 tokens", "mut"))
    sec.append(row(encode(word, en), en_table, 184, 30, 46, label=korean_label))
    b_steps = [7.0, 9.0, 11.0, 13.0]
    b_caps = [f"after {m} merges: {len(ids)} tokens" if m else "start: 9 tokens" for ids, m in k_show]
    for n, (ids, _) in enumerate(k_show):
        start, end = b_steps[n], (b_steps[n + 1] if n + 1 < len(b_steps) else DUR)
        last = n == len(k_show) - 1
        new_ids = set(ids) - set(range(256))
        ident = f"k{n}"
        group = [f'<g id="{ident}" class="a" opacity="{1 if last else 0}">',
                 text(X0, 238, f"English + Korean, {b_caps[n]}", "mut")]
        x = X0
        for i in ids:
            b = mixed_table[i]
            w = len(b) * 46 - 4
            cls = "ko" if i in new_ids else "byt"
            group.append(f'<rect class="{cls}" x="{x}" y="246" width="{w}" height="30" rx="3"/>')
            group.append(f'<text class="{"tw" if i in new_ids else "ink"}" x="{x + w / 2:.1f}" '
                         f'y="266" text-anchor="middle" font-size="14">{korean_label(b)}</text>')
            x += len(b) * 46
        group.append("</g>")
        sec.append("".join(group))
        css_anim.append(window(ident, start, end))
    cl = len(example["tokens"]["cl100k_base"])
    o2 = len(example["tokens"]["o200k_base"])
    sec.append(text(X0, 302, f"Same word: cl100k_base {cl} tokens, o200k_base {o2}.", "mut"))
    sec.append(text(X0, 320, "Ours: 8,000 merges each, ~540 KB of Wikipedia.", "mut"))
    sec.append("</g>")
    body.append("".join(sec))
    css_anim.append(window("b", 7.0, DUR))

    # Legend, always visible: colour = which training text a merged token came from.
    body.append(f'<rect class="en" x="{X0}" y="331" width="14" height="14" rx="2"/>'
                + text(X0 + 20, 343, "from English", "mut")
                + f'<rect class="ko" x="{X0 + 140}" y="331" width="14" height="14" rx="2"/>'
                + text(X0 + 160, 343, "from Korean", "mut")
                + f'<rect class="byt" x="{X0 + 272}" y="331.5" width="13" height="13" rx="2"/>'
                + text(X0 + 292, 343, "raw byte", "mut"))

    css = f"""
  .bg{{fill:#fcfcfb}} .ink{{fill:#0b0b0b}} .mut{{fill:#52514e}} .tw{{fill:#ffffff;font-weight:600}}
  .byt{{fill:#fcfcfb;stroke:#898781;stroke-width:1}}
  .en{{fill:#2a78d6}} .ko{{fill:#eb6834}}
  .cur{{stroke:#0b0b0b;stroke-width:2}}
  text{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
  .a{{animation-duration:{DUR}s;animation-iteration-count:infinite;animation-timing-function:linear}}
  {' '.join(css_anim)}
  @media (prefers-color-scheme: dark) {{
    .bg{{fill:#1a1a19}} .ink{{fill:#f0efec}} .mut{{fill:#c3c2b7}}
    .byt{{fill:#1a1a19;stroke:#898781}} .en{{fill:#3987e5}} .ko{{fill:#d95926}}
    .cur{{stroke:#f0efec}}
  }}
  @media (prefers-reduced-motion: reduce) {{ .a{{animation:none}} }}
"""
    desc = ("Part 1: the word lowest starts as 6 bytes. Four merges learned from a tiny English text "
            "run one by one: e+s, es+t, l+o, lo+w, ending as low + est, although lowest never "
            "appears in that text. Part 2: the Korean word 한국어 is 9 UTF-8 bytes. With merges "
            "learned from English only it stays 9 raw-byte tokens. With merges learned from "
            f"English and Korean, the syllables become single tokens after merges {k_show[1][1]}, "
            f"{k_show[2][1]} and {k_show[3][1]}, ending as 3 tokens. Same word: cl100k_base {cl} tokens, o200k_base {o2}. "
            "Tokens merged from English text are blue, tokens merged from Korean text are orange, "
            "raw single bytes are outlined boxes.")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'role="img"><title>How BPE merges build tokens</title><desc>{desc}</desc>'
           f'<style>{css}</style><rect class="bg" width="{W}" height="{H}"/>' + "".join(body) + "</svg>\n")
    out = HERE / "figures" / "bpe-merges-v2.svg"
    out.write_text(svg, encoding="utf-8")
    print(out.name, len(svg.encode("utf-8")), "bytes")


if __name__ == "__main__":
    main()
