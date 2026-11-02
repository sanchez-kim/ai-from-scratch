"""Write figures/prefix-reuse-v1.svg: the animated prefix-reuse diagram.

    python figures/make_prefix_reuse.py

CSS animation only (no script, no SMIL), so it plays inside an <img> tag.
Every element's static attributes are the final frame, so
prefers-reduced-motion simply switches the animations off.
"""

from pathlib import Path

N = 30            # tokens in the toy prompt
X0, STEP, W = 20, 14, 12
ROW_Y, ROW_H = 70, 30
SPAN = N * STEP - 2   # 418: left edge of the first block to right edge of the last
BAR_Y, BAR_H = 164, 16
DUR = 8               # seconds per loop

# Phase boundaries as % of the loop: 0-1 s, 1-2.5 s, 2.5-4 s, 4-6.5 s, 6.5-8 s.
P1, P2, P3, P4 = 12.5, 31.25, 50.0, 81.25


def recompute_scale(k):
    # Overlay covering the last k blocks, anchored at the right edge.
    return (k * STEP - 2) / SPAN


def marker_x(i):
    return i * STEP  # translateX offset of the changed-token marker from block 0


def kf(name, frames):
    body = " ".join(f"{pct}% {{ {css} }}" for pct, css in frames)
    return f"@keyframes {name} {{ {body} }}"


def window(name, start, end, fade=1.5):
    # Visible only between start% and end% of the loop.
    a, b = max(start - fade, 0), min(end + fade, 100)
    frames = [(0, "opacity:0" if start > 0 else "opacity:1")]
    if start > 0:
        frames += [(a, "opacity:0"), (start, "opacity:1")]
    frames += [(end, "opacity:1")]
    if end < 100:
        frames += [(b, "opacity:0"), (100, "opacity:0")]
    else:
        frames += [(100, "opacity:1")]
    return kf(name, frames)


def main():
    s29, s1, s30 = recompute_scale(29), recompute_scale(1), recompute_scale(30)
    keyframes = [
        # Request 1 wipes in left to right: the cover shrinks toward the right.
        kf("wipe", [(0, "transform:scaleX(1)"), (P1 - 1, "transform:scaleX(0)"),
                    (100, "transform:scaleX(0)")]),
        kf("over", [
            (0, "transform:scaleX(1)"), (P1, "transform:scaleX(1)"),
            (P1 + 2, f"transform:scaleX({s29:.4f})"), (P2, f"transform:scaleX({s29:.4f})"),
            (P2 + 2, f"transform:scaleX({s1:.4f})"), (P3, f"transform:scaleX({s1:.4f})"),
            (P3 + 2, f"transform:scaleX({s30:.4f})"), (P4, f"transform:scaleX({s1:.4f})"),
            (100, f"transform:scaleX({s1:.4f})"),
        ]),
        kf("mark", [
            (0, "opacity:0;transform:translateX(0px)"), (P1, "opacity:0;transform:translateX(14px)"),
            (P1 + 2, "opacity:1;transform:translateX(14px)"), (P2, "opacity:1;transform:translateX(14px)"),
            (P2 + 2, f"opacity:1;transform:translateX({marker_x(29)}px)"),
            (P3, f"opacity:1;transform:translateX({marker_x(29)}px)"),
            (P3 + 2, "opacity:1;transform:translateX(0px)"),
            (P4, f"opacity:1;transform:translateX({marker_x(29)}px)"),
            (100, f"opacity:1;transform:translateX({marker_x(29)}px)"),
        ]),
        kf("ttft", [
            (0, "transform:scaleX(0)"), (P1 - 1, "transform:scaleX(1)"), (P1, "transform:scaleX(1)"),
            (P1 + 2, f"transform:scaleX({29 / 30:.4f})"), (P2, f"transform:scaleX({29 / 30:.4f})"),
            (P2 + 2, f"transform:scaleX({1 / 30:.4f})"), (P3, f"transform:scaleX({1 / 30:.4f})"),
            (P3 + 2, "transform:scaleX(1)"), (P4, f"transform:scaleX({1 / 30:.4f})"),
            (100, f"transform:scaleX({1 / 30:.4f})"),
        ]),
        window("t1", 0, P1), window("t2", P1, P2), window("t3", P2, P3),
        window("t4", P3, P4), window("t5", P4, 100),
        window("c1", 0, P1), window("c2", P1, P2), window("c3", P2, P3),
        window("c4", P3, P4), window("c5", P4, 100),
    ]

    blocks = "".join(
        f'<rect class="blk" x="{X0 + i * STEP}" y="{ROW_Y}" width="{W}" height="{ROW_H}" rx="2"/>'
        for i in range(N)
    )
    css = f"""
  .bg{{fill:#FAFAFA}} .ink{{fill:#17171A}} .mut{{fill:#71717A}}
  .blk{{fill:#8A8A93}} .acc{{fill:#7C3AED}} .accs{{fill:#7C3AED;fill-opacity:.42}}
  .trk{{fill:#E4E4E7}} .cov{{fill:#FAFAFA}}
  text{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
  .a{{animation-duration:{DUR}s;animation-iteration-count:infinite;animation-timing-function:ease-in-out}}
  .r{{transform-box:fill-box;transform-origin:right center}}
  .l{{transform-box:fill-box;transform-origin:left center}}
  #cover{{animation-name:wipe}} #over{{animation-name:over}} #mark{{animation-name:mark}}
  #ttft{{animation-name:ttft}}
  {' '.join(f'#{n}{{animation-name:{n}}}' for n in ['t1','t2','t3','t4','t5','c1','c2','c3','c4','c5'])}
  {' '.join(keyframes)}
  @media (prefers-color-scheme: dark) {{
    .bg,.cov{{fill:#18181B}} .ink{{fill:#F4F4F5}} .mut{{fill:#A1A1AA}}
    .blk{{fill:#71717A}} .acc{{fill:#9061F9}} .accs{{fill:#9061F9;fill-opacity:.38}} .trk{{fill:#3F3F46}}
  }}
  @media (prefers-reduced-motion: reduce) {{ .a{{animation:none}} }}
"""
    # Text is 13px or larger so it stays legible when the 460-wide figure is shown at 370 px.
    # Grey marks are #8A8A93 (3.3:1 on #FAFAFA) and #71717A (3.7:1 on #18181B).
    # Static attributes = final frame: token 30 changed, 29 reused, 1 recomputed.
    s1_static = recompute_scale(1)
    texts = [
        ("t1", 28, "ink", 15, 600, "Request 1: empty cache", 0),
        ("t2", 28, "ink", 15, 600, "Request 2: token 2 changed", 0),
        ("t3", 28, "ink", 15, 600, "Request 3: token 30 changed", 0),
        ("t4", 28, "ink", 15, 600, "Moving the changed token right", 0),
        ("t5", 28, "ink", 15, 600, "Put what changes last", 1),
        ("c1", 214, "ink", 14, 400, "computed 30", 0),
        ("c2", 214, "ink", 14, 400, "reused 1 · recomputed 29", 0),
        ("c3", 214, "ink", 14, 400, "reused 29 · recomputed 1", 0),
        ("c4", 214, "ink", 14, 400, "recomputed = tokens after the change", 0),
        ("c5", 214, "ink", 14, 400, "reused 29 · recomputed 1", 1),
    ]
    text_svg = "".join(
        f'<text id="{i}" class="a {cls}" x="{X0}" y="{y}" font-size="{size}" '
        f'font-weight="{wt}" opacity="{op}">{t}</text>'
        for i, y, cls, size, wt, t, op in texts
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 260" width="460" height="260" role="img">
<title>Prefix reuse in a KV cache</title>
<desc>A prompt of 30 tokens. Request 1 computes all 30. Request 2 changes token 2, so only token 1 is reused and 29 are recomputed, and the time-to-first-token bar is long. Request 3 changes token 30, so 29 are reused, 1 is recomputed, and the bar is short. The changed token then sweeps from left to right and the bar shrinks with it. Conclusion: put what changes last. Toy model with one attention head.</desc>
<style>{css}</style>
<rect class="bg" width="460" height="260"/>
{text_svg}
<text class="mut" x="{X0}" y="50" font-size="13">30 prompt tokens, one attention head</text>
{blocks}
<rect id="over" class="a r accs" x="{X0}" y="{ROW_Y}" width="{SPAN}" height="{ROW_H}" rx="2" style="transform:scaleX({s1_static:.4f})"/>
<rect id="mark" class="a acc" x="{X0}" y="{ROW_Y}" width="{W}" height="{ROW_H}" rx="2" style="transform:translateX({marker_x(29)}px)"/>
<rect id="cover" class="a r cov" x="{X0 - 1}" y="{ROW_Y - 1}" width="{SPAN + 2}" height="{ROW_H + 2}" style="transform:scaleX(0)"/>
<rect class="blk" x="{X0}" y="116" width="10" height="10" rx="2"/><text class="mut" x="{X0 + 16}" y="125" font-size="13">reused from cache</text>
<rect class="blk" x="{X0 + 170}" y="116" width="10" height="10" rx="2"/><rect class="accs" x="{X0 + 170}" y="116" width="10" height="10" rx="2"/><text class="mut" x="{X0 + 186}" y="125" font-size="13">recomputed</text>
<rect class="acc" x="{X0 + 290}" y="116" width="10" height="10" rx="2"/><text class="mut" x="{X0 + 306}" y="125" font-size="13">changed</text>
<text class="mut" x="{X0}" y="156" font-size="13">time to first token</text>
<rect class="trk" x="{X0}" y="{BAR_Y}" width="{SPAN}" height="{BAR_H}" rx="3"/>
<rect id="ttft" class="a l acc" x="{X0}" y="{BAR_Y}" width="{SPAN}" height="{BAR_H}" rx="3" style="transform:scaleX({1 / 30:.4f})"/>
<text class="mut" x="{X0}" y="244" font-size="13">Reuse stops at the first changed token.</text>
</svg>
"""
    out = Path(__file__).resolve().parent / "prefix-reuse-v1.svg"
    out.write_text(svg)
    print("wrote", out, len(svg.encode()), "bytes")


if __name__ == "__main__":
    main()
