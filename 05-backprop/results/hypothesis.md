# Hypotheses, written before running the measurement scripts
Written: 2026-11-02 21:41:41 KST

Already known from writing the code (smoke checks, not measurements): the
diamond a*3 + a*5 gives da = 8 with the correct code; with bug 1 the result
depends on which product's _backward runs last.

## H1 finite-difference error vs epsilon (measure_fd.py)
Error = |finite difference - gradient from backward()|, float64, for
x*x at 3, tanh at 0.5, exp at 1, and a one-neuron loss.
- Forward difference: error about eps * |f''| / 2 for large eps, smallest
  near eps = 1e-8 with an error around 1e-8, then rising again as rounding
  (about 1e-16 / eps) takes over. Around 1e-4 at eps = 1e-12.
- Central difference: error about eps^2 * |f'''| / 6 for large eps, smallest
  near eps = 1e-5 to 1e-6 with an error around 1e-10 to 1e-11, then the same
  rounding rise. Central beats forward by several orders of magnitude at its
  best eps.
- x*x: central difference has no truncation error, so its error is rounding
  only and is smallest at the largest eps. Forward difference error is
  exactly eps in exact arithmetic.
- Both methods are worse at eps = 1e-12 than at 1e-6.

## H2 random small graphs (measure_bugs.py)
3 leaves, n ops from {+, *, tanh}, operands drawn from all earlier nodes
(so values get reused), root = sum of nodes nobody uses. Control: trees,
where every node is used once.
- Trees: bug 1 (assign) changes 0% of gradients, bug 2b (recursive, no
  topological sort) 0%, bug 2a (forgot reversed) close to 100% for n >= 2.
- Reuse graphs: bug 1 share rises with n, about 20% at n = 1, above 95% at
  n = 16. Bug 2b share is lower than bug 1 at every n (it needs a shared
  node that is not a leaf), 0% at n = 1, maybe 70% at n = 16.
- Bug 1 changes a gradient exactly when a node is used more than once, and
  bug 2b exactly when an inner (non-leaf) node is used more than once,
  apart from rare cases where the lost part happens to be 0.

## H3 pure Python vs numpy (bench_numpy.py, timing pending)
Gradient of sum(tanh(W x)) for an n x n W. numpy 100x to 1000x faster at
n = 64. Not measured in this session: the machine is loaded by other
processes, so only a smoke test runs now.

## Addendum after the first runs (2026-11-02 21:52:59 KST)
After the first measure_bugs.py run I added two worked examples to its
"examples" section (backward() called twice; grad_check on the nested graph
a = x*x, a*3 + a*5) and ran it again. Then I ran measure_bugs.py a third time and
measure_fd.py a second time; the sweep, examples and finite-difference rows
matched the earlier files exactly. No measured number was changed by adding statistics.
Outcome notes: H1 held for the location of the best eps; the best errors
were mostly lower than guessed. H2 held for bug 1 and bug 2a; bug 2b was
higher than guessed (90.9% at n = 8, 100% at n = 16).

## Addendum after fact-check (2026-11-02 23:52:06 KST)
Changed only the "leaves" description string in measure_bugs.py setup and re-ran it (fourth run). Sweep and examples identical.
