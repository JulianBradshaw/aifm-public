import { benchmarkData } from './benchmarkData';

interface BenchmarkPoint {
  year: number;
  horizonLength: number;
  label: string;
  company: string;
  releaseDate: string;
  alwaysShow?: boolean;
}

// Convert date string to decimal year
function dateToDecimalYear(dateString: string): number {
  const date = new Date(dateString);
  const year = date.getFullYear();
  const startOfYear = new Date(year, 0, 1);
  const dayOfYear = Math.floor((date.getTime() - startOfYear.getTime()) / (24 * 60 * 60 * 1000)) + 1;
  const daysInYear = new Date(year, 11, 31).getDate() === 31 && new Date(year, 1, 29).getDate() === 29 ? 366 : 365;
  return year + (dayOfYear - 1) / daysInYear;
}

// Convert model key to display name
function modelKeyToDisplayName(key: string): string {
  const displayNames: { [key: string]: string } = {
    // GPT-2 era
    'gpt2': 'GPT-2',
    'davinci_002': 'Davinci-002',
    'gpt_3_5_turbo_instruct': 'GPT-3.5 Turbo Instruct',
    // GPT-4 era
    'gpt_4': 'GPT-4',
    'gpt_4_1106_inspect': 'GPT-4 Turbo',
    'gpt_4_turbo_inspect': 'GPT-4 Turbo',
    'gpt_4o_inspect': 'GPT-4o',
    // GPT-5 era
    'gpt_5_2025_08_07_inspect': 'GPT-5',
    'gpt_5_1_codex_max_inspect': 'GPT-5.1-Codex-Max',
    'gpt_5_2': 'GPT-5.2',
    'gpt_5_3_codex': 'GPT-5.3 Codex',
    'gpt_5_4': 'GPT-5.4',
    // Claude
    'claude_3_opus_inspect': 'Claude 3 Opus',
    'claude_3_5_sonnet_20240620_inspect': 'Claude 3.5 Sonnet',
    'claude_3_5_sonnet_20241022_inspect': 'Claude 3.5 Sonnet (Oct)',
    'claude_3_7_sonnet_inspect': 'Claude 3.7 Sonnet',
    'claude_4_opus_inspect': 'Claude 4 Opus',
    'claude_4_1_opus_inspect': 'Claude Opus 4.1',
    'claude_opus_4_5_inspect': 'Claude Opus 4.5',
    'claude_opus_4_6_inspect': 'Claude Opus 4.6',
    // METR's only Mythos-line measurement: an EARLY Mythos Preview run from a
    // limited access window, not Mythos 5 / Fable 5 (which METR has not measured).
    'claude_mythos_preview_early_inspect': 'Claude Mythos Preview (early)',
    // OpenAI reasoning
    'o1_preview': 'o1-preview',
    'o1_inspect': 'o1',
    'o3_inspect': 'o3',
    // Google
    'gemini_3_pro': 'Gemini 3 Pro',
    'gemini_3_1_pro': 'Gemini 3.1 Pro',
  };
  return displayNames[key] || key;
}

// Determine model company from key
function getModelCompany(key: string): string {
  if (key.startsWith('gpt') || key.startsWith('davinci') || key.startsWith('o1') || key.startsWith('o3') || key.startsWith('o4')) {
    return 'openai';
  } else if (key.startsWith('claude')) {
    return 'claude';
  } else if (key.startsWith('gemini')) {
    return 'google';
  } else if (key.startsWith('deepseek')) {
    return 'deepseek';
  } else if (key.startsWith('qwen')) {
    return 'alibaba';
  } else if (key.startsWith('grok')) {
    return 'xai';
  }
  return 'other';
}

// If true, only display frontier models (those with the highest time horizon at the time of release)
const SHOW_ONLY_FRONTIER_MODELS = true;

export function loadBenchmarkData(percentile: 'p50' | 'p80' = 'p80'): BenchmarkPoint[] {
  const benchmarkPoints: BenchmarkPoint[] = [];
  const horizonKey = percentile === 'p50' ? 'p50_horizon_length' : 'p80_horizon_length';

  Object.entries(benchmarkData.results).forEach(([modelKey, modelData]) => {
    // Get the horizon length from the first available agent
    const firstAgent = Object.values(modelData.agents)[0];
    if (firstAgent && firstAgent[horizonKey]) {
      const horizonLength = firstAgent[horizonKey]!.estimate;
      const year = dateToDecimalYear(modelData.release_date);
      const label = modelKeyToDisplayName(modelKey);
      const company = getModelCompany(modelKey);

      benchmarkPoints.push({
        year,
        horizonLength,
        label,
        company,
        releaseDate: modelData.release_date,
        alwaysShow: modelData.always_show
      });
    }
  });

  // Sort by year for consistency
  benchmarkPoints.sort((a, b) => a.year - b.year);

  if (SHOW_ONLY_FRONTIER_MODELS) {
    // Filter to only include frontier models (highest horizon at time of release)
    // and models marked with always_show
    const result: BenchmarkPoint[] = [];
    let maxHorizonSoFar = -Infinity;

    for (const point of benchmarkPoints) {
      if (point.alwaysShow || point.horizonLength > maxHorizonSoFar) {
        if (point.horizonLength > maxHorizonSoFar) {
          maxHorizonSoFar = point.horizonLength;
        }
        result.push(point);
      }
    }

    return result;
  }

  return benchmarkPoints;
}