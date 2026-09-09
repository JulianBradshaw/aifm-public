'use client';

import { useMemo } from 'react';
import type { ParameterDistribution } from '@/types/samplingConfig';

interface DistributionPreviewChartProps {
  distribution: ParameterDistribution;
  width?: number;
  height?: number;
}

// Compute median for different distribution types
function computeMedian(dist: ParameterDistribution): number | null {
  switch (dist.dist) {
    case 'fixed':
      return typeof dist.value === 'number' ? dist.value : null;

    case 'uniform': {
      const min = dist.min ?? 0;
      const max = dist.max ?? 1;
      return (min + max) / 2;
    }

    case 'normal': {
      if (!dist.ci80) return null;
      const [low, high] = dist.ci80;
      return (low + high) / 2; // Mean = median for normal
    }

    case 'lognormal': {
      if (!dist.ci80) return null;
      const [low, high] = dist.ci80;
      const logLow = Math.log(low);
      const logHigh = Math.log(high);
      const mu = (logLow + logHigh) / 2;
      return Math.exp(mu); // Median of lognormal is exp(mu)
    }

    case 'shifted_lognormal': {
      if (!dist.ci80 || dist.shift === undefined) return null;
      const [low, high] = dist.ci80;
      const logLow = Math.log(low);
      const logHigh = Math.log(high);
      const mu = (logLow + logHigh) / 2;
      return Math.exp(mu) + dist.shift; // Median + shift
    }

    case 'beta': {
      if (dist.alpha === undefined || dist.beta === undefined) return null;
      const a = dist.alpha;
      const b = dist.beta;
      const min = dist.min ?? 0;
      const max = dist.max ?? 1;
      // Approximation for beta median: (a - 1/3) / (a + b - 2/3) for a,b > 1
      // For other cases, use mean as approximation
      let medianT: number;
      if (a > 1 && b > 1) {
        medianT = (a - 1 / 3) / (a + b - 2 / 3);
      } else {
        medianT = a / (a + b); // Use mean as fallback
      }
      return min + medianT * (max - min);
    }

    default:
      return null;
  }
}

// Compute PDF values for different distribution types
function computePdfPoints(
  dist: ParameterDistribution,
  numPoints: number = 100
): { x: number; y: number }[] {
  const points: { x: number; y: number }[] = [];

  switch (dist.dist) {
    case 'fixed':
      // Show a spike at the fixed value
      return [];

    case 'uniform': {
      const min = dist.min ?? 0;
      const max = dist.max ?? 1;
      const height = 1 / (max - min);
      // Uniform distribution is a rectangle
      return [
        { x: min, y: 0 },
        { x: min, y: height },
        { x: max, y: height },
        { x: max, y: 0 },
      ];
    }

    case 'normal': {
      if (!dist.ci80) return [];
      const [low, high] = dist.ci80;
      // CI80 corresponds to ~1.28 standard deviations on each side
      const mean = (low + high) / 2;
      const sigma = (high - low) / (2 * 1.28);

      const xMin = mean - 3.5 * sigma;
      const xMax = mean + 3.5 * sigma;
      const step = (xMax - xMin) / numPoints;

      for (let i = 0; i <= numPoints; i++) {
        const x = xMin + i * step;
        const z = (x - mean) / sigma;
        const y = Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
        points.push({ x, y });
      }
      return points;
    }

    case 'lognormal': {
      if (!dist.ci80) return [];
      const [low, high] = dist.ci80;
      // For lognormal, CI80 in log space
      const logLow = Math.log(low);
      const logHigh = Math.log(high);
      const mu = (logLow + logHigh) / 2;
      const sigma = (logHigh - logLow) / (2 * 1.28);

      // Base range on CI80 with padding (extend by 50% on each side in log space)
      const logRange = logHigh - logLow;
      const logXMin = logLow - 0.5 * logRange;
      const logXMax = logHigh + 0.5 * logRange;
      const logStep = (logXMax - logXMin) / numPoints;

      for (let i = 0; i <= numPoints; i++) {
        const logX = logXMin + i * logStep;
        const x = Math.exp(logX);
        const z = (logX - mu) / sigma;
        const y = Math.exp(-0.5 * z * z) / (x * sigma * Math.sqrt(2 * Math.PI));
        points.push({ x, y });
      }
      return points;
    }

    case 'shifted_lognormal': {
      if (!dist.ci80 || dist.shift === undefined) return [];
      const [low, high] = dist.ci80;
      const shift = dist.shift;
      // Shifted lognormal: X = Y + shift where Y is lognormal
      const logLow = Math.log(low);
      const logHigh = Math.log(high);
      const mu = (logLow + logHigh) / 2;
      const sigma = (logHigh - logLow) / (2 * 1.28);

      // Base range on CI80 with padding (extend by 50% on each side in log space)
      const logRange = logHigh - logLow;
      const logYMin = logLow - 0.5 * logRange;
      const logYMax = logHigh + 0.5 * logRange;
      const logStep = (logYMax - logYMin) / numPoints;

      for (let i = 0; i <= numPoints; i++) {
        const logY = logYMin + i * logStep;
        const y_unshifted = Math.exp(logY);
        const x = y_unshifted + shift;
        const z = (logY - mu) / sigma;
        const pdf = Math.exp(-0.5 * z * z) / (y_unshifted * sigma * Math.sqrt(2 * Math.PI));
        points.push({ x, y: pdf });
      }
      return points;
    }

    case 'beta': {
      if (dist.alpha === undefined || dist.beta === undefined) return [];
      const a = dist.alpha;
      const b = dist.beta;
      const min = dist.min ?? 0;
      const max = dist.max ?? 1;
      const range = max - min;

      // Beta function approximation using Stirling
      const logGamma = (z: number): number => {
        if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
        const zAdj = z - 1;
        return 0.5 * Math.log(2 * Math.PI) + (zAdj + 0.5) * Math.log(zAdj + 1) - (zAdj + 1) +
               (1 / 12) / (zAdj + 1);
      };
      const logBeta = (a: number, b: number): number => {
        // log(B(a,b)) = log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a+b))
        return logGamma(a) + logGamma(b) - logGamma(a + b);
      };

      const logB = logBeta(a, b);
      const step = 1 / numPoints;

      for (let i = 1; i < numPoints; i++) {
        const t = i * step; // t in (0, 1)
        const x = min + t * range;
        // Beta PDF: x^(a-1) * (1-x)^(b-1) / B(a,b)
        const logPdf = (a - 1) * Math.log(t) + (b - 1) * Math.log(1 - t) - logB;
        const y = Math.exp(logPdf) / range; // Scale for the range
        if (Number.isFinite(y)) {
          points.push({ x, y });
        }
      }
      return points;
    }

    case 'choice': {
      // For choice distributions, we can't really show a PDF
      // Return empty - the component will show a bar chart instead
      return [];
    }

    default:
      return [];
  }
}

export function DistributionPreviewChart({
  distribution,
  width = 180,
  height = 80,
}: DistributionPreviewChartProps) {
  const { points, isChoice, choiceData, useLogScale, median } = useMemo(() => {
    if (distribution.dist === 'choice') {
      const values = distribution.values ?? [];
      const probs = distribution.p ?? values.map(() => 1 / values.length);
      return {
        points: [],
        isChoice: true,
        choiceData: values.map((v, i) => ({
          label: String(v),
          prob: probs[i] ?? 0,
        })),
        useLogScale: false,
        median: null as number | null,
      };
    }

    if (distribution.dist === 'fixed') {
      return {
        points: [],
        isChoice: false,
        choiceData: [],
        useLogScale: false,
        median: null as number | null,
      };
    }

    // Use log scale for lognormal distributions
    const isLogDist = distribution.dist === 'lognormal' || distribution.dist === 'shifted_lognormal';

    return {
      points: computePdfPoints(distribution),
      isChoice: false,
      choiceData: [],
      useLogScale: isLogDist,
      median: computeMedian(distribution),
    };
  }, [distribution]);

  const padding = { top: 10, right: 10, bottom: 20, left: 10 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Handle fixed value - show text
  if (distribution.dist === 'fixed') {
    return (
      <div
        style={{ width, height }}
        className="flex items-center justify-center bg-gray-50 rounded text-xs text-gray-600"
      >
        Fixed: {distribution.value}
      </div>
    );
  }

  // Handle choice distribution - show bar chart
  if (isChoice && choiceData.length > 0) {
    const maxProb = Math.max(...choiceData.map(d => d.prob));
    const barWidth = chartWidth / choiceData.length - 4;

    return (
      <svg width={width} height={height} className="bg-gray-50 rounded">
        <g transform={`translate(${padding.left}, ${padding.top})`}>
          {choiceData.map((item, i) => {
            const barHeight = (item.prob / maxProb) * chartHeight;
            const x = i * (chartWidth / choiceData.length) + 2;
            const y = chartHeight - barHeight;

            return (
              <g key={i}>
                <rect
                  x={x}
                  y={y}
                  width={barWidth}
                  height={barHeight}
                  fill="#6366f1"
                  opacity={0.7}
                />
                <text
                  x={x + barWidth / 2}
                  y={chartHeight + 12}
                  textAnchor="middle"
                  fontSize={8}
                  fill="#666"
                >
                  {item.label.length > 6 ? item.label.slice(0, 5) + '…' : item.label}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    );
  }

  // Handle continuous distributions - show PDF curve
  if (points.length === 0) {
    return (
      <div
        style={{ width, height }}
        className="flex items-center justify-center bg-gray-50 rounded text-xs text-gray-500"
      >
        No preview
      </div>
    );
  }

  // Calculate scales
  const xMin = Math.min(...points.map(p => p.x));
  const xMax = Math.max(...points.map(p => p.x));
  const yMax = Math.max(...points.map(p => p.y));

  // Use log scale for x-axis if appropriate
  const scaleX = useLogScale
    ? (x: number) => {
        const logMin = Math.log(Math.max(xMin, 1e-10));
        const logMax = Math.log(xMax);
        const logX = Math.log(Math.max(x, 1e-10));
        return ((logX - logMin) / (logMax - logMin)) * chartWidth;
      }
    : (x: number) => ((x - xMin) / (xMax - xMin)) * chartWidth;

  const scaleY = (y: number) => chartHeight - (y / yMax) * chartHeight;

  // Create path
  const pathD = points
    .map((p, i) => {
      const x = scaleX(p.x);
      const y = scaleY(p.y);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(' ');

  // Create filled area path
  const areaD =
    pathD +
    ` L ${scaleX(points[points.length - 1].x)} ${chartHeight} L ${scaleX(points[0].x)} ${chartHeight} Z`;

  return (
    <svg width={width} height={height} className="bg-gray-50 rounded">
      <g transform={`translate(${padding.left}, ${padding.top})`}>
        {/* Filled area */}
        <path d={areaD} fill="#6366f1" opacity={0.2} />
        {/* Line */}
        <path d={pathD} fill="none" stroke="#6366f1" strokeWidth={1.5} />
        {/* X-axis labels */}
        <text x={0} y={chartHeight + 12} fontSize={8} fill="#666">
          {formatAxisLabel(xMin)}
        </text>
        {median !== null && (
          <text x={scaleX(median)} y={chartHeight + 12} fontSize={8} fill="#666" textAnchor="middle">
            {formatAxisLabel(median)}
          </text>
        )}
        <text x={chartWidth} y={chartHeight + 12} fontSize={8} fill="#666" textAnchor="end">
          {formatAxisLabel(xMax)}
        </text>
        {/* Beta distribution parameter annotation */}
        {distribution.dist === 'beta' && distribution.alpha !== undefined && distribution.beta !== undefined && (
          <text x={chartWidth / 2} y={-2} fontSize={8} fill="#666" textAnchor="middle">
            α={distribution.alpha}, β={distribution.beta}
          </text>
        )}
      </g>
    </svg>
  );
}

function formatAxisLabel(n: number): string {
  if (n === 0) return '0';
  const absN = Math.abs(n);
  if (absN >= 1e6) return n.toExponential(0);
  if (absN >= 1000) return (n / 1000).toFixed(0) + 'k';
  if (absN >= 1) return n.toFixed(1);
  if (absN >= 0.01) return n.toFixed(2);
  return n.toExponential(0);
}

export default DistributionPreviewChart;
