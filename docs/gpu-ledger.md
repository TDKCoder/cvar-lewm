# GPU ledger

`scripts/submit_job.sh` appends a row when a job is submitted. Fill in `actual_h` and `actual_SU`
from the reservation app (reserve.dsmlp.ucsd.edu, Dashboard -> SU Budget) after the job ends.

Weekly budget (default, confirm on the Dashboard): 10 SU per member per workspace, Mon 00:00 to Mon 00:00 Pacific.

| submitted (PT) | job | class | peak? | est_h | est_SU | pod | actual_h | actual_SU | notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-13 18:46 | smoke-eval-pusht | medium | peak | 1.0 | 1.000 | cvl-smoke-eval-pusht |  |  | Phase 0 GPU smoke: get_cost/encode/collect timings, 10-ep upstream eval, closed-loop kick check (250 ep) |
