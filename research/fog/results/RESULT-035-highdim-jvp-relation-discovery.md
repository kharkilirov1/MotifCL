# RESULT-035 — Full Jacobians are not required for the first production structural gate

EXP-036 shows that randomized Jacobian-vector products are sufficient to verify
candidate finite-order/conjugacy laws in nonlinear hidden systems at widths up
to the current FOG `d_model=320` setting.

This changes the production instrumentation requirement from an infeasible
`O(d^2)` Jacobian materialization to a small number of black-box directional
probes per candidate law.  The probe is an **evidence gate**, not a proof that a
law is exact: perturbed systems retain the correct best candidate while their
state/JVP residuals correctly remain nonzero.

The current PyTorch CPU flash-SDPA kernel lacks forward AD; `fog_lmw.structural`
therefore switches only the structural probe to math-SDPA. Normal model
execution is unchanged.
