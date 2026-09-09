/**
 * Explanatory text for chart elements and concepts
 */

export const HORIZON_LENGTH_EXPLANATION =
  "The coding time horizon is the maximum length of coding tasks frontier AI systems can complete with a success rate of 80%, with the length defined as the time taken by typical AI company employees who do similar tasks."

export const COMPUTE_CHART_EXPLANATION =
  "Effective compute is the amount of compute that would be needed, using early 2025 training algorithms, to achieve the frontier AI capability level at a given time. " +
  "In our model, most progress comes from software/algorithm improvements, so in practice we think our model is useful even if we need a breakthrough to get to AGI (we could then think of effective compute as a stand-in for general capabilities).";

export const AI_R_D_PROGRESS_MULTIPLIER_EXPLANATION =
  "The AI Software R&D Uplift is the speedup in software progress " +
  "that would be achieved if the frontier AI systems at a given time were deployed within today's leading AI company. (In previous work we called this the AI R&D progress multiplier.)";

export const MILESTONE_FULL_NAMES: Record<string, string> = {
  'AC': 'Automated Coder',
  'SC': 'Superhuman Coder',
  'SAR': 'Superhuman AI Researcher',
  'SIAR': 'Superintelligent AI Researcher',
  'STRAT-AI': 'Strategic AI',
  'TED-AI': 'Top-Expert-Dominating AI',
  'ASI': 'Artificial Superintelligence',
};

export const MILESTONE_EXPLANATIONS: Record<string, string> = {
  'AC': 'An AC can fully automate an AGI project\'s coding work, autonomously replacing the project\'s entire software engineering staff.',
  'AI2027-SC': 'A Superhuman Coder milestone from AI 2027 projections.',
  'SAR': 'A SAR can fully automate AI R&D.',
  'SIAR': ' The gap between a SIAR and the top AGI project human researcher is 2x greater than the gap between the top AGI project human researcher and the median researcher.', // todo improve explanatino
  // 'STRAT-AI': 'AI directs large-scale R&D piortfolios, making strategic decisions that compound progress across teams.',
  'TED-AI': 'A TED-AI is at least as good as top human experts at virtually all cognitive tasks.',
  'ASI': 'The gap between an ASI and the best humans is 2x greater than the gap between the best humans and the median professional, at virtually all cognitive tasks.',
};

export const SMALL_CHART_EXPLANATIONS = {
  // Coding Automation Fractions
  automationFraction: 'The fraction of coding work involved in frontier AI research that can be efficiently automated.',
  // AI Parallel Coding Labor Multiplier
  aiCodingLaborMultiplier: 'A multiplier of 2x would mean that AIs were increasing coding productivity by as much as if the AGI project had an extra copy of all of their employees.',
  // AI Serial Coding Labor Multiplier
  serialCodingLaborMultiplier: 'The speedup in coding work from AI assistance. A multiplier of 2x would mean that AIs were increasing coding productivity by as much as if the AGI project sped up their employees by 2x.',
  // AI Research Taste
  aiResearchTaste: 'If AIs were to select experiments, this would be the value per experiment, where 1 is the mean value per experiment for human researchers.',
  // AI Software R&D Uplift
  aiSwProgressMultRefPresentDay: 'The speedup in software progress that would be achieved if the frontier AI systems at a given time were deployed within today\'s leading AI company.',
  // Research Effort
  researchEffort: 'Measures the amount of quality-adjusted research happening at a given time.',
  // Cumulative Research Effort
  researchStock: 'The accumluation of software research effort over time. Cumulative research effort is to research effort what distance traveled is to speed. (We call this research stock in the model explanation.)',
  // ETSP Rate
  etspRate: 'By how many orders of magnitude ETSP (effective training system performance) grows per year. Equal to the sum of the software efficiency growth rate and the training system performance growth rate.',
  predictedRevenue: 'The leading AGI developer\'s annualized revenue is predicted by extrapolating from May 2026 revenues and growth rates as an exponential trend in effective compute. (Annualized revenue is the most recent month\'s revenue multiplied by 12.) The red diamonds are Anthropic’s reported annualized revenue; they are shown for comparison only and do not influence the trend.',
  capabilityEci: 'The Epoch Capabilities Index (ECI) score predicted by assuming that ECI is a linear function of log(effective compute).',
  capabilityEciRate: 'How fast the Epoch Capabilities Index is rising, in index points per year. This is the slope of the ECI trend, computed analytically as points-per-OOM \u00d7 the effective-compute progress rate rather than by differencing the plotted curve, so the two always agree exactly. At the ECI reference date it equals the present-day ECI growth-rate setting by construction.',
  excessSerialCodingLaborUplift: 'The most capable model\u2019s serial coding-labor uplift minus 1, measured with human labor and inference compute held at present-day levels. Unlike the revenue and time horizon trends, this is not an independent extrapolation; it is what the model produces.',
  // Software Progress Rate
  softwareProgressRate: 'By how many orders of magnitude software efficiency grows per year.',
  // Training System Performance Growth Rate
  trainingSystemPerformanceGrowthRate: 'By how many orders of magnitude training system performance grows per year.',
  // Training Compute (H100e input)
  trainingComputeH100e: 'The number of H100-equivalents running the leading AI company\'s frontier training run at a given time — the instantaneous hardware training rate. Multiplied by FLOP_PER_H100E_YEAR it is the training rate in FLOP/yr (the hardware part of Training System Performance); cumulative training compute is its integral over time.',
  // Total R&D Compute (H100e)
  totalRdComputeH100e: 'Total R&D compute at the leading AI company, in H100-equivalents: the sum of training compute, internal inference compute (for coding automation), and experiment compute — i.e. everything except commercial serving. All amortized over the year.',
  // World Compute (H100e)
  worldComputeH100e: 'Total AI-relevant compute in the world, in H100-equivalents.',
  // Frontier AI company share of world compute
  companyShareOfWorldPct: 'The percentage of the world\'s AI-relevant compute operated by the leading AI company (its R&D compute plus commercial serving compute).',
  // AI company compute spend shares
  spendPctCumCommercial: 'How the leading AI company splits its total compute, as percentages: inference for coding automation, training, experiments, and commercial (serving customers — the total minus everything else).',
  // Software Efficiency
  softwareEfficiency: 'Software efficiency measures how efficiently the training process at a given time can convert training compute into performance. It is multiplied by training compute to get effective compute.',
  etsp: 'Effective Training System Performance (ETSP) is the instantaneous effective-compute rate, in OOMs. It is the sum of training system performance (the hardware rate) and software efficiency (the algorithmic contribution); the stacked bands show how each contributes.',
  // Training System Performance (rate)
  trainingSystemPerformance: 'Training System Performance (TSP) is the instantaneous hardware training rate, in OOMs of FLOP/yr — i.e. the frontier run\'s H100e converted to FLOP/yr. Effective compute accumulates as the integral of ETSP = TSP + software efficiency.',
  // Training compute (cumulative FLOP stock)
  trainingCompute: 'Cumulative training compute (a stock), in OOMs of FLOP: the running integral of the training system performance. Effective compute is the same integral but including software efficiency.',
  // Experiment Throughput
  experimentCapacity: 'The number of experiments the leading AI company can run per unit time, where experiments are weighted by how much compute they use and how labor-intensive they are to code.',
  // Inference Compute for Coding Automation
  inferenceCompute: 'The amount of compute used by the leading AI company for automating coding, in H100-equivalents.',
  // Experiment Compute
  experimentCompute: 'The amount of compute used by the leading AI company for running experiments, in H100-equivalents.',
  // Human Coding Labor
  humanLabor: 'The number of human coders at the leading AI company.',
};

// Explanations for the Plan A explorer (/plan-a) panels.
export const COVERT_CHART_EXPLANATIONS = {
  capability:
    'Effective compute (OOMs) over time for three projects racing toward TED-AI after a ' +
    'pause. The covert project pays no alignment tax and reaches bare TED-AI. The legal ' +
    "consortium runs two glidepaths: DISCOVERED (its full research output, reaching TED-AI + " +
    'tax) and USED (its deployed capability, reaching bare TED-AI — the rest of the research ' +
    'is spent on the alignment tax). Horizontal markers show TED-AI and TED-AI + tax; vertical ' +
    'markers show when the covert and the legal used capability reach TED-AI.',
  etsp:
    'ETSP (effective training system performance) decomposed into its two additive ' +
    'contributions, in OOMs/yr: TSP (the hardware training rate) and SW (software efficiency). ' +
    'Effective compute is the time-integral of ETSP = TSP + SW. Shown for the covert project ' +
    'and the legal discovered/used glidepaths.',
  detection:
    'P(detected by t) for the covert facility, t = years after the pause. The combined ' +
    'detection probability is one minus the product of two independent survival terms: the ' +
    "facility-frailty survival (the covert datacenter's power signature) and the silver-bullet " +
    'survival (1 − P a program-level detection breakthrough has emerged, a ceiling-logistic ' +
    "anchored on the legal project's used→TED-AI time).",
};

export const COVERT_KNOB_EXPLANATIONS: Record<string, string> = {
  years_before_ac: 'How many years before the (un-paused) AC date the deal/pause lands. Anchors the whole scenario timing.',
  setup_period_years: 'Training-R&D pause length after the deal (the "six-month pause"). Training is frozen; only the software windfall accrues.',
  setup_sw_gain_progress_months: 'Radical-transparency windfall: algorithmic progress (in 2025-progress-months) pooled publicly during the pause. The covert steals a frac_stolen share.',
  alignment_tax_progress_years: 'Research the legal consortium must spend on alignment instead of capability, in 2025-progress-years. The discovered glidepath reaches TED-AI + tax; the used glidepath reaches bare TED-AI.',
  legal_scaling_duration: 'Years from the END of the setup pause to the legal target. target_year = pause + setup_period + this.',
  legal_strategy: 'How the legal consortium scales after the pause. actual_hw follows the real training-compute series; target_year tunes HW+SW; capped_sw caps the tuned software rate.',
  legal_mode: 'AIFM effort physics: capped clamps legal research effort at the physics ceiling; backsolve lets it exceed the ceiling to hit the target.',
  frac_stolen: "Fraction of the legal project's algorithmic progress the covert steals (the windfall plus ongoing software gains).",
  dormancy_years: 'Years the covert lies low after the pause BEFORE starting construction (effective compute frozen; only stolen software accrues). No facility exists yet, so it cannot be detected — the detection clock starts when construction begins.',
  construction_years: 'Years to build the covert datacenter, after the dormancy period (effective compute still frozen; only stolen software accrues). The facility is detectable while under construction.',
  compute_mode: 'How the covert datacenter is sized: frontier_relative (a fraction of frontier compute at the pause) or absolute (a fixed H100e count).',
  total_h100e_frac: "Covert R&D compute as a fraction of the frontier's total compute at the pause (frontier_relative mode).",
  total_h100e: 'Covert datacenter size in total H100e (absolute mode).',
  facility_w_per_h100e: 'Facility power draw per H100e (W). Converts the covert compute into facility power (MW), which the detection model keys on.',
  optimal_site_mw: 'Per-site power (MW) the covert splits its compute across (n = total_MW / this, floored at 1). Must lie within the detection-grid size range.',
  max_hw_rate: 'Max hardware (training) scaling rate after the pause (OOM/yr). Unused by the actual_hw strategy.',
  max_sw_rate: 'Flat ceiling (OOM/yr) on the legal tuned software rate (capped_sw strategy only); does not cap the setup windfall.',
  post_scaling_externalities_rate: 'After the legal project reaches its target, it keeps DISCOVERING software efficiency at this flat rate (OOM/yr) as an unavoidable side-effect of ongoing alignment research — none deployed (TSP frozen), but it inflates the discovered research stock the covert steals at frac_stolen. 0 ⇒ the legal simply sits after its target. (actual_hw strategy only.)',
  min_sw_frac_of_tsp: 'Mandatory minimum software during the scaling phase: software efficiency must grow at AT LEAST this fraction of the TSP (hardware) growth rate, on BOTH the used and discovered tracks, regardless of the AIFM effort ceiling (a guaranteed floor like the transparency windfall). 0 ⇒ no floor. (actual_hw strategy only.)',
  enable_deal_breakdown: 'Gate for the deal-dissolution scenario: the deal dissolves some years after the pause ends. The consortium disbands (its trajectory freezes and covert stealing stops) and the leading company continues alone as its own project with no leakage.',
  deal_breakdown_years_after_pause_end: 'When the deal dissolves, in years AFTER the end of the setup pause. The consortium freezes at that moment; the leading company takes over from the used (deployed) capability with the full discovered research stock.',
  breakdown_compute_mode: "What the leading company's post-dissolution compute share is measured against. frontier_at_pause: frontier R&D compute at the pause — compute built or kept during the deal is effectively destroyed at dissolution. world_at_breakdown: the world's compute at the moment the deal dissolves (the input series carried through the deal) — nothing is destroyed; the share is the % of current-world compute the largest post-dissolution project runs on.",
  breakdown_compute_reduction: "The leading company's post-dissolution compute as a fraction of the basis chosen above. Under frontier_at_pause, 30% ⇒ it restarts with 0.3× the compute the frontier had when the deal landed; under world_at_breakdown, 30% ⇒ it runs on 0.3× the compute the world has when the deal dissolves.",
  post_breakdown_compute_growth_rate: "Constant growth rate (OOM/yr) of the leading company's compute after the dissolution.",
};
