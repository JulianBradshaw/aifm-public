# Plan A covert-project model

This directory models a vulnerability in Plan A's international AI slowdown deal:
could a covert project develop advanced AI while the legal consortium follows an
agreed scaling and alignment program? It compares capability trajectories,
milestone arrival times, detection risk, and outcomes if the deal breaks down.

The simulation extends the shared [AI Futures Model](../progress_model/README.md).
It has two consumers:

- **Offline analysis:** `pipeline/` samples assumptions and produces datasets,
  figures, tables, and a browser for comparing runs.
- **Interactive explorer:** the website's `/plan-a` page calls
  `POST /api/plan-a` in [api/index.py](../api/index.py). It uses the simulation
  and detection code here to compute an individual race.

## What is simulated

Each Monte Carlo draw starts with an unrestricted reference trajectory. Its
Automated Coder (AC) arrival anchors the deal's start, and its capability
thresholds provide a common scale for comparing projects. TED-AI means
Top-Expert-Dominating AI; ASI means Artificial Superintelligence.

The legal consortium pauses, then scales toward TED-AI while paying an alignment
tax. The model distinguishes software it discovers from software it deploys;
the covert project can steal a share of discovered progress. In the default
`capped` mode, the model limits legal research effort, so the consortium can
miss its target date. The alternative `backsolve` mode solves for the effort
needed to hit the target even when it exceeds that ceiling.

The covert project combines its own research with stolen software. Assumptions
cover compute acquisition, facility size, dormancy, construction time, and
espionage reach. Detection depends on facility size and time, with a separate
AI-assisted detection mechanism. Configurations with a `deal_breakdown` block
also sample a collapse time and simulate the leading company's continuation
after dissolution.

The baseline series in [inputs/input_data.csv](../inputs/input_data.csv) drives
the reference trajectory. When `plan_a_input_data` is configured, the legal and
covert paths use the compute ramp from [inputs/plan-a.csv](../inputs/plan-a.csv),
shifted and scaled to each rollout's pause. Results depend on the selected
assumptions and simulation horizon. See [potential-issues.md](potential-issues.md)
for modeling questions to consider when interpreting them.

## Run locally

Run all commands below from the **repository root**, with the project's Python
environment active. See the [repository setup instructions](../README.md#quick-start)
to create it. The offline pipeline also needs Matplotlib, which is currently
absent from `requirements-dev.txt`:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt matplotlib

python -m plan_a_covert_proj.pipeline.run \
  --local --num 50 --workers 2 --name local-trial
```

This builds a small dataset and all applicable figures locally. The detection
fit runs once per dataset build and can take several minutes even with a small
sample count. Fifty draws are useful for trying the workflow, not for precise
probability estimates.

Browse outputs and launch further runs with:

```bash
python -m plan_a_covert_proj.pipeline.serve
```

Open `http://127.0.0.1:8765/` for the run browser or
`http://127.0.0.1:8765/launch` for the launcher. The launcher includes config
previews, inheritance information, and job logs. Select local execution when
launching without a remote box. Use `--port` to change the server port.

## Choose assumptions

Every run takes a **model config** and a **covert config**. Model configs set
capability-model parameters; covert configs set the deal, legal strategy,
covert project, and detection assumptions. YAML files can inherit from another
file using `parent:`, resolved relative to the child file.

| Config in `plan_a_covert_proj/config/` | Purpose |
| --- | --- |
| [model_uncertain_thomas.yaml](config/model_uncertain_thomas.yaml) | Default model config; inherits sampled parameters from `config/sampling_config_daniel.yaml` |
| [model_scenario_branch.yaml](config/model_scenario_branch.yaml) | Fixed central model parameters, stored as a standalone snapshot |
| [covert_scenario_branch.yaml](config/covert_scenario_branch.yaml) | Default covert config; pins facility size and theft fraction to scenario values |
| [covert_uncertain.yaml](config/covert_uncertain.yaml) | Base covert config, including uncertain facility size and theft fraction |
| [covert_compute_bottom_up.yaml](config/covert_compute_bottom_up.yaml) | Separate compute-acquisition distributions referenced by the base covert config |
| [covert_scenario_branch_deal_breakdown.yaml](config/covert_scenario_branch_deal_breakdown.yaml) | Adds a deal-collapse hazard and continuation assumptions; explicitly uses frontier-relative compute sizing |

**The default scenario branch still contains uncertainty.** It inherits
`bottom_up_estimate` compute sizing, whose acquisition parameters are sampled
per draw, as well as stochastic detection. Pinning the model config does not
make the entire experiment deterministic.

The compute modes are `bottom_up_estimate` (diverted world compute),
`frontier_relative` (a fraction of frontier compute at the pause), and `absolute`
(a specified H100-equivalent count). Only sizing parameters for the selected
mode are used. `covert_bottom_up.yaml` is an alias of the base uncertain config,
retained for existing variants.

To vary covert assumptions over the fixed central takeoff trajectory:

```bash
python -m plan_a_covert_proj.pipeline.run --local --num 50 --workers 2 \
  --model-config plan_a_covert_proj/config/model_scenario_branch.yaml \
  --covert-config plan_a_covert_proj/config/covert_uncertain.yaml
```

Inspect resolved settings before running:

```bash
python -m plan_a_covert_proj.pipeline.show_config
python -m plan_a_covert_proj.pipeline.show_config covert_uncertain.yaml --sampled-only
python -m plan_a_covert_proj.pipeline.show_config covert_compute_bottom_up.yaml
python -m plan_a_covert_proj.pipeline.show_config --all
```

The compact view resolves parent inheritance and shows distributions and their
medians. Bare filenames resolve within `plan_a_covert_proj/config/`. Inspect the
bottom-up config separately: it is referenced by path, rather than inherited.

## Reuse and inspect results

Outputs are gitignored and organized by a content-derived run ID:

```text
plan_a_covert_proj/outputs/
  runs/<runid>/
    dataset.json                      # trajectories, milestones, draws, summary
    manifest.json                     # configs, inputs, git SHA, output inventory
    figures/                          # plots and other figure outputs
    tables_compute_acquisition.md     # bottom-up runs only
    tables_compute_acquisition.html
  latest -> runs/<runid>
  viewer.html                         # browser regenerated after each run
  _jobs/                              # logs from the web launcher
```

Repeating a command reuses its dataset and rebuilds figures. To rebuild selected
figures for the local trial above, keep the same sample count and configs:

```bash
python -m plan_a_covert_proj.pipeline.run \
  --local --num 50 --workers 2 --only spaghetti_pct,lead_hist
```

`--only` selects figures; it still builds the dataset if that run has no cached
one. Tables, the manifest, and the browser are refreshed as well. Use `--force`
to rebuild an existing dataset. Figure failures are printed and do not stop
the whole run, so check the log and `manifest.json` for the outputs you need.

The run ID hashes both config inheritance chains, the referenced input CSVs and
bottom-up config, selected model and simulation code, and run parameters.
[run.py](pipeline/run.py)'s `_input_files()` and `_code_inputs()` define the
exact set. Figure-code changes reuse the dataset; changes to hashed inputs
produce a new run directory. The manifest records resolved config snapshots
and the input-file list. Run IDs identify inputs, not a complete snapshot of
the Python environment.

| Flag | Effect |
| --- | --- |
| `--num` / `--workers` | Sample count / worker count; defaults are 500 / 30 |
| `--seed` | Override the seed from the configs |
| `--years-before-ac` | Pin the deal's start relative to reference AC arrival |
| `--legal-mode capped` or `backsolve` | Override legal research-effort treatment |
| `--min-survival-years` | Condition a configured deal-collapse hazard on survival for at least this long |
| `--covert-exists-prior` | Set the covert-existence prior used by the capability-distribution figures |
| `--solve-legal-compute` | Also solve implied legal compute for each rollout; the median scenario always includes it |
| `--name` | Human-readable run label; does not change the run ID |

Run `python -m plan_a_covert_proj.pipeline.run --help` for the full interface.

## Run on a remote box

Omitting `--local` builds the dataset remotely, then downloads it and generates
figures locally. Remote execution requires SSH access, `rsync`, a prepared repo
working copy, and a Python environment with the model dependencies installed.
All build inputs must be inside the repository tree.

| Environment variable | Default |
| --- | --- |
| `COVERT_REMOTE_HOST` | `aifm-runpod` |
| `COVERT_REMOTE_ROOT` | `~/covert-pause-exp` |
| `COVERT_REMOTE_PY` | `python3.13` |

Authentication comes from your SSH configuration and keys. The pipeline syncs
build inputs and checks their digest on the remote machine before running.

```bash
python -m plan_a_covert_proj.pipeline.run --num 500
```

## Code map and maintenance

| File | Responsibility |
| --- | --- |
| [sim.py](sim.py) | Reference, legal, covert, compute-acquisition, and post-collapse trajectories |
| [detection.py](detection.py) | Fit and sample the detection model |
| [detection_baked.py](detection_baked.py) | Load or write the precomputed detection fit |
| [deal_hazard.py](deal_hazard.py) | Deal survival curves and collapse-time sampling |
| [plan_a_inputs.py](plan_a_inputs.py) | Validate and transform the Plan A input series |
| [pipeline/mc.py](pipeline/mc.py) | Sample assumptions and build the dataset |
| [pipeline/run.py](pipeline/run.py) | Cache, build, render, and record a run |
| [pipeline/figures.py](pipeline/figures.py), [pipeline/tables.py](pipeline/tables.py) | Generate analyses from saved datasets |
| [pipeline/serve.py](pipeline/serve.py), [pipeline/make_viewer.py](pipeline/make_viewer.py) | Launcher and run browser |
| [pipeline/ac_reweight.py](pipeline/ac_reweight.py), [pipeline/proposal.py](pipeline/proposal.py) | AC-time conditioning and importance-sampling tools |
| [pipeline/deal_survival_sweep.py](pipeline/deal_survival_sweep.py) | Compare different minimum deal-survival assumptions |

To add a figure, register a `(name, function)` pair in `FIGURES` in
`pipeline/figures.py`. The function receives `(dataset, output_directory)` and
returns its output filename, or `None` when inapplicable. Use saved trajectories
and milestones; if an analysis needs another simulated quantity, store it in
`mc.py` during dataset construction.

Detection assumptions are documented in
[detection_elicitation.md](config/detection_elicitation.md). The explorer loads
[detection_fit_baked.json](config/detection_fit_baked.json) when its fingerprint
matches the configured grids, falling back to a live fit otherwise. After
changing those grids, regenerate the fit so API requests do not need to perform
the expensive optimization:

```bash
python -m plan_a_covert_proj.pipeline.bake_detection_fit
```

The website explorer has its own frontend and default assets. To run it locally,
follow the [website setup](../README.md#quick-start), then open
`http://localhost:3000/plan-a` (`?embed=1` enables the embedded view).
