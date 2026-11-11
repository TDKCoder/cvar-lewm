## YYYY-MM-DD: <experiment name>

**Question / hypothesis:** <which of H1-H3, stated as a testable claim>

**Setup:** env=<>, p=<>, heads=<>, costs=<>, S=<>, alpha/beta=<>, seeds=<>, episodes per cell=<>.
Code at commit `<sha>`. Jobs: `<jobs/submitted/...>`. GPU: <class, model from env.txt>, <h> h, <SU> SU.

**Commands:**
```
<exact aggregate / verify commands>
```

**Results** (from `results/<exp-name>/summary.csv`; verifier: PASS):

| cell | n | success [95% CI] | worst-10% cost [95% CI] | plan time (s) |
|---|---|---|---|---|

Figures: `results/<exp-name>/<fig>.png`

**Verdict:** supported / not supported / inconclusive, because <numbers, CI overlap>.

**Caveats:** <hardware variation, small n, anything excluded and why>

**Next:** <one concrete next step>
