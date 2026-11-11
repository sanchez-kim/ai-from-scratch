"""The 2-D search used by make_greedy_walk.py: a neighbour graph and a beam search.

This is the bottom-layer search of HNSW, written out in plain Python so the
figure shows a real run, not a drawing. ef is the size of the candidate list.
"""
import heapq
import math
import random


def make_points(seed, n=46, w=1.0, h=1.0):
    """Five loose clusters in a w x h box (distances are measured in that box)."""
    rng = random.Random(seed)
    centres = [(rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)) for _ in range(5)]
    pts = []
    for i in range(n):
        cx, cy = centres[i % len(centres)]
        x = min(max(rng.gauss(cx, 0.11), 0.03), 0.97)
        y = min(max(rng.gauss(cy, 0.11), 0.03), 0.97)
        pts.append((x * w, y * h))
    return pts


def neighbour_graph(pts, m=3):
    """Link each point to its m nearest, both ways (a small, sparse graph)."""
    graph = {i: set() for i in range(len(pts))}
    for i, p in enumerate(pts):
        near = sorted((math.dist(p, q), j) for j, q in enumerate(pts) if j != i)[:m]
        for _, j in near:
            graph[i].add(j)
            graph[j].add(i)
    return {i: sorted(v) for i, v in graph.items()}


def beam_search(graph, pts, query, entry, ef):
    """Returns the steps: (node expanded, visited set, candidate list) after each expansion."""
    dist = lambda i: math.dist(pts[i], query)
    visited = {entry}
    frontier = [(dist(entry), entry)]           # nodes still to expand, closest first
    best = [(-dist(entry), entry)]              # the candidate list: the ef closest seen so far
    steps = []
    while frontier:
        d, node = heapq.heappop(frontier)
        if len(best) == ef and d > -best[0][0]:
            break                               # nothing left to expand is closer than the list
        for nb in graph[node]:
            if nb in visited:
                continue
            visited.add(nb)
            dn = dist(nb)
            if len(best) < ef or dn < -best[0][0]:
                heapq.heappush(frontier, (dn, nb))
                heapq.heappush(best, (-dn, nb))
                if len(best) > ef:
                    heapq.heappop(best)
        steps.append((node, set(visited), sorted(i for _, i in best)))
    return steps
