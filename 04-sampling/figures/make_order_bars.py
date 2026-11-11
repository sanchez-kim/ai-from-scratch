"""Write figures/order-bars-v2.svg: one real position, two sampler orders.

    python figures/make_order_bars.py results/probs-<model>-<stamp>.json \
        --passage 1 --prefix 20 --t 0.7 --p 0.9 --k 15

The 20 most likely next tokens at one position of the saved data, drawn
twice. Top panel: temperature -> top-k -> top-p (HF order). Bottom panel:
top-k -> top-p -> temperature (llama.cpp order). Bars are probabilities on
one 0-1 scale; the coloured line is the running total that top-p cuts at p.
Colour = identity: blue (series slot 1) is the HF order, orange (slot 2) the
llama.cpp order. Survivors take their panel's colour; cut tokens stay grey.
The survivor sets come from sampling.run_chain, so they match analyze.py.

CSS animation only (no script, no SMIL), so it plays inside an <img> tag.
Every element's static attributes are the final frame, so
prefers-reduced-motion switches the animations off and shows that frame.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampling import run_chain, softmax  # noqa: E402

W, H = 460, 396
N = 20
X0, STEP, BW = 46, 20, 14
PANELS = {"A": 78, "B": 232}     # top of each plot area
PH = 84                          # plot height for probability 1.0
DUR = 12


def kf(name, frames):
    return f"@keyframes {name} {{ " + " ".join(f"{p}% {{ {c} }}" for p, c in frames) + " }"


def fade_in(name, at):
    return kf(name, [(0, "opacity:0"), (at - 2, "opacity:0"), (at, "opacity:1"), (100, "opacity:1")])


def probs_over(cands, keep_ids, t):
    # Probabilities top-p and the final draw see: softmax at temperature t,
    # renormalized over the tokens still in play.
    scaled = [(tok, lp / t) for tok, lp in cands if tok in keep_ids]
    return dict(softmax(scaled))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probs")
    ap.add_argument("--passage", type=int, required=True)
    ap.add_argument("--prefix", type=int, required=True)
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--k", type=int, required=True)
    args = ap.parse_args()
    T, P, K = args.t, args.p, args.k

    data = json.loads(Path(args.probs).read_text())
    pos = next(x for x in data["positions"] if x["passage"] == args.passage and x["prefix_len"] == args.prefix)
    cands = [(tok, lp) for tok, lp in pos["top"]]
    shown = [tok for tok, _ in cands[:N]]
    topk_ids = {tok for tok, _ in cands[:K]}

    surv_a = {tok for tok, _ in run_chain(cands, [("temperature", T), ("top_k", K), ("top_p", P)])}
    surv_b = {tok for tok, _ in run_chain(cands, [("top_k", K), ("top_p", P), ("temperature", T)])}
    assert len(surv_a) < N and len(surv_b) < N, "pick a position whose survivors fit in 20 bars"

    s0 = probs_over(cands, topk_ids, 1.0)          # top-k only, T = 1
    sT = probs_over(cands, topk_ids, T)            # top-k only, T applied
    fa = probs_over(cands, surv_a, T)              # final draw, order A
    fb = probs_over(cands, surv_b, T)              # final draw, order B

    states = {
        # bar heights per phase, and which bars survive
        "A": {"phases": [s0, sT, sT, fa], "cum": sT, "surv": surv_a},
        "B": {"phases": [s0, s0, s0, fb], "cum": s0, "surv": surv_b},
    }
    # Phase times (% of the loop): raw, temperature (A), cut, final heights.
    P_T, P_K, P_P, P_F = 12, 28, 42, 62

    css_kf, body = [], []
    for name, top in PANELS.items():
        st = states[name]
        base = top + PH

        def y(prob):
            return base - PH * prob

        # grid: 0, 0.5, 1 and axis labels
        for v in (0, 0.5, 1.0):
            body.append(f'<line class="grid" x1="{X0 - 4}" x2="{X0 + STEP * N}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
            body.append(f'<text class="mut" x="{X0 - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')

        for i, tok in enumerate(shown):
            x = X0 + STEP * i + 3
            in_k = tok in st["surv"] or tok in topk_ids
            survivor = tok in st["surv"]
            heights = [ph.get(tok, 0.0) for ph in st["phases"]]
            if not survivor:
                # A cut bar keeps the height it had when it was cut.
                heights[-1] = heights[-2] if tok in topk_ids else heights[0]
            final_h = max(heights[-1], 1e-6)
            ratios = [h / final_h for h in heights]
            bid = f"{name}{i}"
            frames = [(0, f"transform:scaleY({ratios[0]:.4f})")]
            if name == "A":
                frames += [(P_T, f"transform:scaleY({ratios[0]:.4f})"), (P_T + 8, f"transform:scaleY({ratios[1]:.4f})")]
            frames += [(P_F, f"transform:scaleY({ratios[2]:.4f})"), (P_F + 8, f"transform:scaleY({ratios[3]:.4f})"),
                       (100, f"transform:scaleY({ratios[3]:.4f})")]
            css_kf.append(kf(f"h{bid}", frames))
            anims = [f"h{bid}"]
            cls = "bar"
            if not survivor:
                cls += " cut"
                anims.append("cutP" if in_k else "cutK")
            css_kf.append(f"#g{bid}{{animation-name:{','.join(anims)}}}")
            h = PH * heights[-1]
            body.append(f'<g id="g{bid}" class="a sy"><rect class="blk" x="{x}" y="{base - h:.2f}" width="{BW}" height="{h:.2f}"/>')
            if survivor:
                body.append(f'<rect class="c{name} a vio{name}" x="{x}" y="{base - h:.2f}" width="{BW}" height="{h:.2f}"/>')
            body.append("</g>")
            # Survivor strip: most survivors are too small to see as bars, so
            # each token also gets a square under its bar.
            body.append(f'<rect class="sq" x="{x}" y="{base + 6}" width="{BW}" height="9"/>')
            if survivor:
                body.append(f'<rect class="c{name} a vio{name}" x="{x}" y="{base + 6}" width="{BW}" height="9"/>')
            if tok == pos["actual_next"]:
                body.append(f'<text class="ink" x="{x + BW / 2}" y="{base + 32}" text-anchor="middle">↑ "{pos["actual_next_text"].strip()}"</text>')

        # top-k cut: a vertical line after bar K
        kx = X0 + STEP * K + 1
        body.append(f'<g class="a kline"><line class="kl" x1="{kx}" x2="{kx}" y1="{top - 4}" y2="{base + 18}"/>'
                    f'<text class="sec" x="{kx + 4}" y="{base + 32}">top-k {K}</text></g>')
        # top-p: the running total over the top-k tokens, and the line at p
        total, pts = 0.0, []
        for i, tok in enumerate(shown[:K]):
            total += st["cum"][tok]
            pts.append(f"{X0 + STEP * i + 3 + BW / 2:.1f},{y(total):.1f}")
        body.append(f'<g class="a pline{name}"><line class="pl" x1="{X0}" x2="{X0 + STEP * N}" y1="{y(P):.1f}" y2="{y(P):.1f}"/>'
                    f'<polyline class="cum l{name} a draw{name}" pathLength="1" points="{" ".join(pts)}"/>'
                    f'<text class="sec" x="{X0 + STEP * N}" y="{y(P) - 5:.1f}" text-anchor="end">top-p {P:g}</text></g>')
        n = len(st["surv"])
        body.append(f'<g class="a cnt{name}"><rect class="c{name}" x="{X0 + STEP * N - 96}" y="{base - 40}" width="10" height="10"/>'
                    f'<text class="ink b" x="{X0 + STEP * N}" y="{base - 30}" text-anchor="end">{n} survive</text></g>')

    css_kf += [
        fade_in("kin", P_K), fade_in("pin", P_P), fade_in("vioin", P_P + 8), fade_in("cntin", P_P + 10),
        kf("drawA", [(0, "stroke-dashoffset:1"), (P_P, "stroke-dashoffset:1"), (P_P + 8, "stroke-dashoffset:0"), (100, "stroke-dashoffset:0")]),
        kf("cutP", [(0, "opacity:1"), (P_P + 6, "opacity:1"), (P_P + 9, "opacity:.28"), (100, "opacity:.28")]),
        kf("cutK", [(0, "opacity:1"), (P_K, "opacity:1"), (P_K + 3, "opacity:.28"), (100, "opacity:.28")]),
        ".kline{animation-name:kin} .plineA,.plineB{animation-name:pin}",
        ".drawA,.drawB{animation-name:drawA} .vioA,.vioB{animation-name:vioin} .cntA,.cntB{animation-name:cntin}",
    ]

    text_of = dict(zip(shown, pos["top_text"]))
    top5 = [(text_of[t], math.exp(lp)) for t, lp in cands[:5]]
    ctx = pos["context_tail"].replace("\n", " ").strip()
    if len(ctx) > 30:
        # Cut at a word boundary so the subtitle never starts mid-word.
        tail = ctx[-31:]
        ctx = "…" + tail[tail.index(" ") + 1:]
    ctx = ctx.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")

    css = """
  .bg{fill:#fcfcfb} .ink{fill:#0b0b0b} .sec{fill:#52514e} .mut{fill:#898781} .blk{fill:#898781}
  .cA{fill:#2a78d6} .cB{fill:#eb6834} .lA{stroke:#2a78d6} .lB{stroke:#eb6834} .b{font-weight:600}
  .grid{stroke:#e1e0d9;stroke-width:1} .kl{stroke:#0b0b0b;stroke-width:1.5;stroke-dasharray:4 3}
  .pl{stroke:#52514e;stroke-width:1;stroke-dasharray:3 3} .cum{fill:none;stroke-width:2;stroke-dasharray:1}
  .cut{opacity:.35} .sq{fill:#c3c2b7}
  text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
  .h{font-size:14px;font-weight:600}
  .a{animation-duration:%ds;animation-iteration-count:infinite;animation-timing-function:ease-in-out}
  .sy{transform-box:fill-box;transform-origin:center bottom}
  @media (prefers-color-scheme: dark) {
    .bg{fill:#1a1a19} .ink{fill:#f0efec} .sec{fill:#c3c2b7} .mut{fill:#898781} .blk{fill:#898781}
    .cA{fill:#3987e5} .cB{fill:#d95926} .lA{stroke:#3987e5} .lB{stroke:#d95926}
    .grid{stroke:#2c2c2a} .kl{stroke:#f0efec} .sq{fill:#383835} .pl{stroke:#c3c2b7}
  }
  @media (prefers-reduced-motion: reduce) { .a{animation:none !important} }
""" % DUR
    lines_a = f"temperature {T:g} → top-k → top-p · HF"
    lines_b = f"top-k → top-p → temperature {T:g} · llama.cpp"
    desc = (f"Next-token probabilities from qwen3.5:4b-q4_K_M after the text "
            f"'{ctx}'. Top five at temperature 1, over the full vocabulary: "
            + ", ".join(f"{w!r} {p:.3f}" for w, p in top5)
            + f". Panel A (blue, the Hugging Face order) applies temperature {T:g} first, then top-k {K} and top-p {P:g}: {len(surv_a)} tokens survive. "
            f"Panel B (orange, the llama.cpp order) applies top-k {K} and top-p {P:g} first and temperature {T:g} last: {len(surv_b)} tokens survive. "
            f"The word the passage actually continues with, {pos['actual_next_text'].strip()!r}, is rank "
            f"{shown.index(pos['actual_next']) + 1}: {'kept' if pos['actual_next'] in surv_a else 'cut'} in A, "
            f"{'kept' if pos['actual_next'] in surv_b else 'cut'} in B. "
            "Survivors are coloured in their panel's colour, cut tokens are grey. "
            "Same probabilities, same settings, different survivor sets.")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Same settings, two sampler orders</title>
<desc>{desc.replace('&', '&amp;').replace('<', '&lt;')}</desc>
<style>{css}{" ".join(css_kf)}</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="16" y="24">Same settings, two orders</text>
<text class="sec" x="16" y="42">next token after "{ctx}"</text>
<rect class="cA" x="16" y="{PANELS['A'] - 22}" width="12" height="12"/>
<text class="ink" x="34" y="{PANELS['A'] - 12}">{lines_a}</text>
<rect class="cB" x="16" y="{PANELS['B'] - 22}" width="12" height="12"/>
<text class="ink" x="34" y="{PANELS['B'] - 12}">{lines_b}</text>
{"".join(body)}
<text class="sec" x="16" y="{H - 24}">coloured: survivors, grey: cut · ↑ my word</text>
<text class="sec" x="16" y="{H - 8}">line: running total · top-k {K} in this figure</text>
</svg>
"""
    out = Path(__file__).resolve().parent / "order-bars-v2.svg"
    out.write_text(svg)
    print("wrote", out, len(svg.encode()), "bytes;", f"A {len(surv_a)} survive, B {len(surv_b)} survive")
    print("top5:", top5)


if __name__ == "__main__":
    main()
