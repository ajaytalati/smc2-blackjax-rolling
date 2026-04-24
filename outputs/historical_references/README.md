# Historical reference scripts

Reference / historical scripts not part of the active codebase. Kept for
provenance but not imported by anything.

- `Rolling_Batch_Orchestrator.py` — early architectural sketch.
- `Simulation_with_5_Year_Macrocycles_and_Missing_Data.py` — early
  5-year simulation experiment.

Both were written by an external LLM (Gemini) as reference explorations
during v4 development. The patterns they describe are superseded by the
`smc2bj/` + `models/` + `drivers/` structure of this repo.

**Do not run these scripts** — imports like `from simulator.sde_model`
no longer resolve (the framework is now `smc2bj.simulator`). Preserved
as-is for historical traceability only.
