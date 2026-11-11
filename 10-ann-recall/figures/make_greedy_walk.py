"""Animated SVG: a beam search walks a 2-D neighbour graph toward a query.

Phase 1 keeps a candidate list of 1 (ef_search = 1) and stops at a point that
is not the nearest. Phase 2 keeps 4 and reaches the nearest. The walk is a real
run of walk.beam_search, picked by a fixed rule (the first seed whose graph is connected, where
ef 1 stops early and ef 4 succeeds), not drawn by hand.

    uv run python figures/make_greedy_walk.py
"""
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from walk import beam_search, make_points, neighbour_graph  # noqa: E402

OUT = pathlib.Path(__file__).parent / "greedy-walk-v2.svg"
W, H = 460, 432
X0, Y0, PW, PH = 16, 58, 428, 262
EF_SMALL, EF_BIG = 1, 4


def connected(graph):
    seen, todo = {0}, [0]
    while todo:
        for nb in graph[todo.pop()]:
            if nb not in seen:
                seen.add(nb)
                todo.append(nb)
    return len(seen) == len(graph)


def pick():
    for seed in range(1000):
        pts = make_points(seed, w=PW, h=PH)
        graph = neighbour_graph(pts)
        if not connected(graph):
            continue  # every point must be reachable, as in a real HNSW layer
        rng = random.Random(seed + 1000)
        query = (rng.uniform(0.2, 0.8) * PW, rng.uniform(0.2, 0.8) * PH)
        order = sorted(range(len(pts)), key=lambda i: math.dist(pts[i], query))
        entry = order[-1]  # start at the point farthest from the query
        small, big = (beam_search(graph, pts, query, entry, ef) for ef in (EF_SMALL, EF_BIG))
        got_small = min(small[-1][2], key=lambda i: math.dist(pts[i], query))
        got_big = min(big[-1][2], key=lambda i: math.dist(pts[i], query))
        if got_small != order[0] and order.index(got_small) >= 2 and got_big == order[0] \
                and 3 <= len(small) <= 6 and len(big) <= 12:
            return seed, pts, graph, query, entry, order, small, big, order.index(got_small) + 1


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def xy(p):
    return f'cx="{X0 + p[0]:.1f}" cy="{Y0 + p[1]:.1f}"'


def frame_svg(pts, step, path, cls, lines):
    node, visited, cands = step
    out = [f'<g class="{cls}">']
    pl = " ".join(f"{X0 + pts[i][0]:.1f},{Y0 + pts[i][1]:.1f}" for i in path)
    out.append(f'<polyline class="pth" points="{pl}"/>')
    out += [f'<circle class="v" {xy(pts[i])} r="4.5"/>' for i in sorted(visited) if i not in cands]
    out += [f'<circle class="c" {xy(pts[i])} r="5.5"/>' for i in cands]
    out.append(f'<circle class="cur" {xy(pts[node])} r="10"/>')
    for k, t in enumerate(lines):
        out.append(f'<text class="{"ink" if k == 0 else "mut"}" x="16" y="{344 + 19 * k}">{t}</text>')
    out.append("</g>")
    return "".join(out)


def main():
    seed, pts, graph, query, entry, order, small, big, small_rank = pick()
    n = len(pts)
    frames = []  # (svg, seconds on screen)
    for k, st in enumerate(small):
        path = [s[0] for s in small[: k + 1]]
        last = k == len(small) - 1
        lines = [f"ef_search {EF_SMALL} · step {k + 1}: list keeps 1 point",
                 f"stuck: no neighbour is closer. This is the {ordinal(small_rank)} nearest." if last
                 else "move to the closest neighbour, if it is closer"]
        frames.append((path, st, lines, 2.6 if last else 1.1))
    for k, st in enumerate(big):
        path = [s[0] for s in big[: k + 1]]
        last = k == len(big) - 1
        lines = [f"ef_search {EF_BIG} · step {k + 1}: list keeps {len(st[2])} points",
                 f"found the true nearest. Checked {len(st[1])} of {n} points." if last
                 else "expand the closest unexpanded point in the list"]
        frames.append((path, st, lines, 5.0 if last else 1.0))
    total = sum(f[3] for f in frames)

    css, groups, t = [], [], 0.0
    for idx, (path, st, lines, dur) in enumerate(frames):
        a, b = 100 * t / total, 100 * (t + dur) / total
        final = idx == len(frames) - 1
        name = f"f{idx}"
        if final:
            css.append(f"@keyframes {name}{{0%,{a:.2f}%{{opacity:0}}{a + .01:.2f}%,100%{{opacity:1}}}}")
        else:
            css.append(f"@keyframes {name}{{0%,{a:.2f}%{{opacity:0}}{a + .01:.2f}%,{b:.2f}%{{opacity:1}}"
                       f"{b + .01:.2f}%,100%{{opacity:0}}}}")
        css.append(f".{name}{{animation-name:{name}}}")
        groups.append(frame_svg(pts, st, path, f"a {name} {'fin' if final else 'fr'}", lines))
        t += dur
    # Where ef 1 stopped: shown from phase 2 to the end, and in the static frame.
    stop = small[-1][0]
    a = 100 * sum(f[3] for f in frames[: len(small)]) / total
    css.append(f"@keyframes stp{{0%,{a:.2f}%{{opacity:0}}{a + .01:.2f}%,100%{{opacity:1}}}} .stp{{animation-name:stp}}")

    edges = sorted({(min(i, j), max(i, j)) for i in graph for j in graph[i]})
    nearest = order[0]
    qx, qy = X0 + query[0], Y0 + query[1]
    sx, sy = X0 + pts[stop][0], Y0 + pts[stop][1]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>Greedy walk with a candidate list of 1 and of 4</title>
<desc>{n} points, each linked to its 3 nearest (both ways). The walk starts at the point farthest from the query. With ef_search {EF_SMALL} it moves to a closer neighbour while one exists and stops after {len(small)} steps at the {ordinal(small_rank)} nearest point. With ef_search {EF_BIG} it keeps the {EF_BIG} closest points seen so far, expands {len(big)} of them, checks {len(big[-1][1])} of {n} points and finds the true nearest.</desc>
<style>
  .bg{{fill:#fcfcfb}} .ink{{fill:#0b0b0b}} .mut{{fill:#52514e}}
  .e{{stroke:#c3c2b7;stroke-width:1}} .p{{fill:#fcfcfb;stroke:#898781;stroke-width:1.2}}
  .v{{fill:#eda100}} .c{{fill:#eb6834}} .cur{{fill:none;stroke:#eb6834;stroke-width:2}}
  .pth{{fill:none;stroke:#eb6834;stroke-width:2;stroke-opacity:.6}}
  .q{{stroke:#0b0b0b;stroke-width:2.5}} .tn{{fill:none;stroke:#2a78d6;stroke-width:2.5}}
  .sm{{fill:none;stroke:#52514e;stroke-width:1.5;stroke-dasharray:3 2}}
  text{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}}
  .h{{font-size:14px;font-weight:600}}
  .fr{{opacity:0}} .fin,.stp{{opacity:1}}
  .a{{animation-duration:{total:.1f}s;animation-iteration-count:infinite;animation-timing-function:step-end}}
  {" ".join(css)}
  @media (prefers-color-scheme: dark) {{
    .bg{{fill:#1a1a19}} .ink{{fill:#f0efec}} .mut{{fill:#c3c2b7}}
    .e{{stroke:#383835}} .p{{fill:#1a1a19;stroke:#898781}} .v{{fill:#c98500}} .c{{fill:#d95926}}
    .cur{{stroke:#d95926}} .pth{{stroke:#d95926}} .q{{stroke:#f0efec}} .tn{{stroke:#3987e5}} .sm{{stroke:#c3c2b7}}
  }}
  @media (prefers-reduced-motion: reduce) {{ .a{{animation:none !important}} }}
</style>
<rect class="bg" width="{W}" height="{H}"/>
<text class="ink h" x="16" y="24">A greedy walk through a neighbour graph</text>
<text class="mut" x="16" y="42">ef_search = size of the candidate list</text>
{"".join(f'<line class="e" x1="{X0 + pts[i][0]:.1f}" y1="{Y0 + pts[i][1]:.1f}" x2="{X0 + pts[j][0]:.1f}" y2="{Y0 + pts[j][1]:.1f}"/>' for i, j in edges)}
{"".join(f'<circle class="p" {xy(p)} r="4"/>' for p in pts)}
<circle class="tn" {xy(pts[nearest])} r="9"/>
{"".join(groups)}
<g class="a stp"><circle class="sm" cx="{sx:.1f}" cy="{sy:.1f}" r="13"/></g>
<line class="q" x1="{qx - 7:.1f}" y1="{qy:.1f}" x2="{qx + 7:.1f}" y2="{qy:.1f}"/><line class="q" x1="{qx:.1f}" y1="{qy - 7:.1f}" x2="{qx:.1f}" y2="{qy + 7:.1f}"/>
<circle class="c" cx="22" cy="396" r="5.5"/><text class="mut" x="32" y="400">in the list</text>
<circle class="v" cx="138" cy="396" r="4.5"/><text class="mut" x="148" y="400">checked</text>
<circle class="tn" cx="230" cy="396" r="7"/><text class="mut" x="242" y="400">true nearest</text>
<line class="q" x1="16" y1="419" x2="28" y2="419"/><line class="q" x1="22" y1="413" x2="22" y2="425"/><text class="mut" x="32" y="423">query</text>
<circle class="sm" cx="104" cy="419" r="7"/><text class="mut" x="116" y="423">where ef_search 1 stopped</text>
</svg>
'''
    OUT.write_text(svg)
    print(f"seed {seed}: ef1 {len(small)} steps -> rank {small_rank}; ef4 {len(big)} steps, "
          f"{len(big[-1][1])}/{n} checked; {OUT.name} {len(svg.encode()) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
