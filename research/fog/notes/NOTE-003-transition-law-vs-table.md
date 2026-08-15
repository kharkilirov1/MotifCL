# NOTE-003 — Transition law vs transition table

A recurrent module can still memorize a finite graph of states.

For a dense PC matrix `W`, observing states `e_0..e_3` only constrains the rows /
subspace touched by those inputs.  Reusing the same matrix at the next clock
cycle does not magically define the intended action on unseen states.

By contrast, a translation-shared parameterization imposes

`T(e_j) = shift_delta(e_j)`

with the same low-dimensional `delta` at every position.  Learning the law from
a prefix then constrains its action outside the prefix.

Research implication: FOG should distinguish

- **weight sharing in time**, from
- **structural sharing across state identities**.

The first gives recurrence.  The second is what can give algorithmic depth
extrapolation.
