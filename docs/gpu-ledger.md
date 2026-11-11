# GPU ledger

`scripts/submit_job.sh` appends a row when a job is submitted. Fill in `actual_h` and `actual_SU`
from the reservation app (reserve.dsmlp.ucsd.edu, Dashboard -> SU Budget) after the job ends.

Weekly budget (default, confirm on the Dashboard): 10 SU per member per workspace, Mon 00:00 to Mon 00:00 Pacific.

| submitted (PT) | job | class | peak? | est_h | est_SU | pod | actual_h | actual_SU | notes |
|---|---|---|---|---|---|---|---|---|---|
