# RESULT-015 — A constraint becomes reliable when parameterization makes it global

Free codebooks are usually successful but can learn a nearly-shared transition
whose small phase variation causes binary and recurrent failure.

Generator-orbit sharing **alone** performed badly from the tested random
initializations: the reduced periodic parameterization introduced hard local
minima.

Adding the matching global closure `T^30=I` and identity law changes the result:
for seeds 40–44, **5/5** achieve

- successor 100%;
- complete binary law 100%;
- recurrent depth 64: 100%.

This is important because a similar root regularizer did not fix EXP-010 when
all identity phases were independent.  Here the regularizer constrains the
single object that generates the entire chart.

Conclusion:

> parameterization and algebraic constraint must refer to the same global
> structure; otherwise the loss may regularize only one disconnected degree of
> freedom.

Evidence: `artifacts/research/exp_016/orbit_closure.json` and
`metrics_combined.json`.
