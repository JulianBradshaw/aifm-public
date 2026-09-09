/**
 * Forecast chart PNG export
 * ---------------------------------------------------------------------------
 * Builds a self-contained, presentation-grade PNG from a live forecast chart.
 *
 * The forecast charts (MilestoneDistributionChart, ConditionalMilestoneTimingChart)
 * render their plot as an SVG but their *legend* as plain HTML next to it. Simply
 * serialising the <svg> would therefore silently drop the legend, so this module
 * composes a brand new SVG document containing:
 *
 *   title (et-book) → subtitle (Geist) → redrawn legend → the cloned plot →
 *   footer rule + watermark
 *
 * and rasterises it through an <img> onto a canvas at 3x pixel density.
 *
 * Two pitfalls drive most of the code here:
 *
 *  1. FONTS. The page's text resolves to next/font's generated family name
 *     (e.g. `__Geist_e5a1cd`), which does not exist inside a standalone SVG —
 *     text would silently fall back to a system serif. So every text node is
 *     re-pointed at our own family names, whose @font-face rules embed the
 *     self-hosted woff files as base64 data URIs inside the SVG itself.
 *  2. DETACHED CLONES. `getComputedStyle` returns nothing useful for an element
 *     that is not in the document, so styles are read from the *live* tree and
 *     copied onto the clone by walking both trees in lockstep.
 */

import { EXPORT_COLORS, downloadBlob } from './chartExport';

/** Family names declared inside the exported SVG (must not collide with page fonts). */
const F = {
  serif: 'AIFMExportSerif',
  sans: 'AIFMExportSans',
  mono: 'AIFMExportMono',
} as const;

const COLORS = {
  background: EXPORT_COLORS.background,
  foreground: EXPORT_COLORS.foreground,
  muted: '#666666',
  hairline: 'rgba(13, 13, 13, 0.18)',
  watermark: 'rgba(13, 13, 13, 0.30)',
} as const;

const LAYOUT = {
  pad: 40,
  minContentWidth: 820,
  titleSize: 27,
  subtitleSize: 13.5,
  legendSize: 12.5,
  legendLineHeight: 19,
  legendSwatchWidth: 30,
  legendSwatchGap: 9,
  legendRowGap: 4,
  watermarkSize: 11,
} as const;

const SVG_NS = 'http://www.w3.org/2000/svg';

// ---------------------------------------------------------------------------
// Font embedding
// ---------------------------------------------------------------------------

async function fetchAsBase64(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch font ${url}: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

let fontCssPromise: Promise<string> | null = null;

/**
 * Build @font-face rules with the woff payloads inlined as data URIs.
 * External URLs are never fetched when an SVG is rendered through an <img>,
 * so embedding is the only way to get the real typefaces into the raster.
 */
function getEmbeddedFontCss(): Promise<string> {
  if (fontCssPromise) return fontCssPromise;

  fontCssPromise = (async () => {
    const [etRoman, etBold, geistSans, geistMono] = await Promise.all([
      fetchAsBase64('/fonts/et-book/et-book-roman-line-figures.woff'),
      fetchAsBase64('/fonts/et-book/et-book-bold-line-figures.woff'),
      fetchAsBase64('/fonts/GeistVF.woff'),
      fetchAsBase64('/fonts/GeistMonoVF.woff'),
    ]);

    return `
      @font-face { font-family: '${F.serif}'; font-weight: normal; font-style: normal;
        src: url(data:font/woff;base64,${etRoman}) format('woff'); }
      @font-face { font-family: '${F.serif}'; font-weight: bold; font-style: normal;
        src: url(data:font/woff;base64,${etBold}) format('woff'); }
      @font-face { font-family: '${F.sans}'; font-weight: 100 900; font-style: normal;
        src: url(data:font/woff;base64,${geistSans}) format('woff'); }
      @font-face { font-family: '${F.mono}'; font-weight: 100 900; font-style: normal;
        src: url(data:font/woff;base64,${geistMono}) format('woff'); }
    `;
  })();

  // Don't cache a rejected fetch.
  fontCssPromise.catch(() => { fontCssPromise = null; });
  return fontCssPromise;
}

/** Map a resolved CSS font-family stack onto one of our embedded families. */
function mapFontFamily(resolved: string): string {
  const lower = resolved.toLowerCase();
  if (lower.includes('et-book') || lower.includes('georgia') || lower.includes('serif')) {
    // "sans-serif" also contains "serif" — check it isn't that.
    if (!lower.includes('sans-serif') || lower.includes('et-book')) return F.serif;
  }
  if (lower.includes('mono')) return F.mono;
  return F.sans;
}

// ---------------------------------------------------------------------------
// Legend extraction
// ---------------------------------------------------------------------------

export interface LegendRun {
  text: string;
  color: string;
}

export interface LegendItem {
  runs: LegendRun[];
  swatchColor: string;
  swatchDash: string | null;
  opacity: number;
}

/** Collect text runs with their resolved colour, so the PNG legend keeps the
 *  site's convention of a coloured label plus a grey statistics tail. */
function collectRuns(el: Element, runs: LegendRun[]): void {
  const color = window.getComputedStyle(el).color || COLORS.foreground;
  el.childNodes.forEach((node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent ?? '';
      if (text.trim().length > 0) runs.push({ text, color });
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      collectRuns(node as Element, runs);
    }
  });
}

/**
 * Read the live HTML legend into a structured model.
 *
 * Depends on the legend markup shape used by both forecast charts:
 *   [data-chart-legend] > div (per milestone column) > div (one row per series)
 * with the row's first child being the swatch and its last child the label.
 */
export function extractLegendItems(container: HTMLElement | null): LegendItem[] {
  if (!container) return [];

  const rows = Array.from(container.querySelectorAll<HTMLElement>(':scope > div > div'));

  return rows.map((row): LegendItem | null => {
    const swatch = row.firstElementChild;
    const label = row.lastElementChild;
    if (!swatch || !label || swatch === label) return null;

    let swatchColor: string = COLORS.muted;
    let swatchDash: string | null = null;
    let opacity = 1;

    const swatchStyle = window.getComputedStyle(swatch);
    opacity = parseFloat(swatchStyle.opacity || '1') || 1;

    const innerLine = swatch.querySelector('line');
    if (innerLine) {
      // Breakout rows draw their swatch as a tiny inline <svg><line/></svg>.
      const lineStyle = window.getComputedStyle(innerLine);
      swatchColor = lineStyle.stroke || swatchColor;
      swatchDash = innerLine.getAttribute('stroke-dasharray');
      opacity *= parseFloat(innerLine.getAttribute('opacity') ?? '1') || 1;
    } else if (
      swatchStyle.backgroundColor &&
      swatchStyle.backgroundColor !== 'rgba(0, 0, 0, 0)' &&
      swatchStyle.backgroundColor !== 'transparent'
    ) {
      swatchColor = swatchStyle.backgroundColor;
    } else if (parseFloat(swatchStyle.borderTopWidth || '0') > 0) {
      swatchColor = swatchStyle.borderTopColor || swatchColor;
      if (swatchStyle.borderTopStyle === 'dashed') swatchDash = '5 4';
    }

    const runs: LegendRun[] = [];
    collectRuns(label, runs);
    if (runs.length === 0) return null;

    const labelOpacity = parseFloat(window.getComputedStyle(label).opacity || '1') || 1;
    return { runs, swatchColor, swatchDash, opacity: Math.min(opacity, labelOpacity) };
  }).filter((item): item is LegendItem => item !== null);
}

// ---------------------------------------------------------------------------
// Text measurement & wrapping
// ---------------------------------------------------------------------------

let measureCtx: CanvasRenderingContext2D | null = null;

function getMeasureContext(): CanvasRenderingContext2D {
  if (!measureCtx) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to acquire a 2D context for text measurement');
    measureCtx = ctx;
  }
  return measureCtx;
}

/** Measure with the page's own webfonts, whose metrics match the embedded copies. */
function measureText(text: string, font: string): number {
  const ctx = getMeasureContext();
  ctx.font = font;
  return ctx.measureText(text).width;
}

interface WrappedSegment {
  text: string;
  color: string;
}

/** Greedy word wrap that preserves per-run colours across line breaks. */
function wrapRuns(runs: LegendRun[], maxWidth: number, font: string): WrappedSegment[][] {
  const lines: WrappedSegment[][] = [];
  let current: WrappedSegment[] = [];
  let currentWidth = 0;

  for (const run of runs) {
    // Keep the leading space of a run (e.g. " (p10: ...)") as a word separator.
    const words = run.text.split(/(\s+)/).filter((w) => w.length > 0);
    for (const word of words) {
      const isSpace = /^\s+$/.test(word);
      const width = measureText(word, font);

      if (!isSpace && currentWidth + width > maxWidth && current.length > 0) {
        lines.push(current);
        current = [];
        currentWidth = 0;
      }
      // Don't start a wrapped line with whitespace.
      if (isSpace && current.length === 0) continue;

      const last = current[current.length - 1];
      if (last && last.color === run.color) {
        last.text += word;
      } else {
        current.push({ text: word, color: run.color });
      }
      currentWidth += width;
    }
  }

  if (current.length > 0) lines.push(current);
  return lines;
}

// ---------------------------------------------------------------------------
// Chart SVG sanitising
// ---------------------------------------------------------------------------

/** Presentation properties that may come from CSS rather than SVG attributes. */
const TEXT_PROPS = ['font-size', 'font-weight', 'font-style', 'letter-spacing'] as const;

/**
 * Deep-clone the live plot and make it standalone:
 *  - copy resolved styles from the live tree (a detached clone has none)
 *  - re-point every font-family at an embedded family
 *  - convert foreignObject tick labels into real SVG text
 *  - drop hover-only chrome (crosshair, hover shading)
 */
function sanitizeChartSvg(liveSvg: SVGSVGElement): SVGSVGElement {
  const clone = liveSvg.cloneNode(true) as SVGSVGElement;

  // Walk both trees in lockstep — indices align because the clone is exact.
  const liveEls = [liveSvg, ...Array.from(liveSvg.querySelectorAll('*'))];
  const cloneEls = [clone, ...Array.from(clone.querySelectorAll('*'))];

  for (let i = 0; i < liveEls.length && i < cloneEls.length; i++) {
    const live = liveEls[i];
    const target = cloneEls[i] as Element & { style?: CSSStyleDeclaration };
    if (!target.style) continue;

    const tag = live.tagName.toLowerCase();
    const isTextual = tag === 'text' || tag === 'tspan' || tag === 'foreignobject' || tag === 'div';
    if (!isTextual) continue;

    const computed = window.getComputedStyle(live);
    target.style.setProperty('font-family', mapFontFamily(computed.fontFamily));
    for (const prop of TEXT_PROPS) {
      const value = computed.getPropertyValue(prop);
      if (value) target.style.setProperty(prop, value);
    }
    // Text colour usually arrives as a fill attribute; only fill the gap.
    if (tag === 'text' && !live.getAttribute('fill') && computed.fill) {
      target.style.setProperty('fill', computed.fill);
    }
  }

  // Replace any foreignObject tick label with an equivalent <text>.
  clone.querySelectorAll('foreignObject').forEach((fo) => {
    const text = (fo.textContent ?? '').trim();
    const parent = fo.parentNode;
    if (!parent) return;
    if (!text) {
      parent.removeChild(fo);
      return;
    }
    const x = parseFloat(fo.getAttribute('x') ?? '0');
    const width = parseFloat(fo.getAttribute('width') ?? '0');
    const y = parseFloat(fo.getAttribute('y') ?? '0');
    const height = parseFloat(fo.getAttribute('height') ?? '0');

    const textEl = document.createElementNS(SVG_NS, 'text');
    textEl.setAttribute('x', String(x + width));
    textEl.setAttribute('y', String(y + height / 2));
    textEl.setAttribute('text-anchor', 'end');
    textEl.setAttribute('dominant-baseline', 'middle');
    textEl.setAttribute('font-size', '11');
    textEl.setAttribute('fill', COLORS.muted);
    textEl.style.setProperty('font-family', F.sans);
    textEl.textContent = text;
    parent.replaceChild(textEl, fo);
  });

  // Hover-only chrome must never appear in a static export.
  clone.querySelectorAll('[data-export-hide]').forEach((el) => el.remove());

  return clone;
}

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------

/**
 * The chart primitives size themselves from a ResizeObserver, starting at a
 * 250px placeholder. Toggling PDF/CDF remounts the chart, so an export fired
 * immediately after the toggle can otherwise capture the placeholder width.
 * Wait until the SVG matches its container and has stopped changing.
 */
async function waitForStableChartSize(svg: SVGSVGElement): Promise<void> {
  const deadline = performance.now() + 1200;
  let previous = -1;

  while (performance.now() < deadline) {
    const width = svg.getBoundingClientRect().width;
    const container = svg.parentElement;
    const containerWidth = container ? container.getBoundingClientRect().width : width;

    if (width > 0 && width === previous && Math.abs(width - containerWidth) <= 2) return;

    previous = width;
    // rAF alone would never fire in a hidden tab, so race it against a timer:
    // the export must still finish if the user switches away mid-click.
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => resolve());
      setTimeout(resolve, 32);
    });
  }
}

function createText(
  content: string,
  x: number,
  y: number,
  opts: { size: number; family: string; fill: string; weight?: string; anchor?: string; opacity?: number }
): SVGTextElement {
  const el = document.createElementNS(SVG_NS, 'text');
  el.setAttribute('x', String(x));
  el.setAttribute('y', String(y));
  el.setAttribute('font-size', String(opts.size));
  el.setAttribute('fill', opts.fill);
  el.style.setProperty('font-family', opts.family);
  if (opts.weight) el.setAttribute('font-weight', opts.weight);
  if (opts.anchor) el.setAttribute('text-anchor', opts.anchor);
  if (opts.opacity != null) el.setAttribute('opacity', String(opts.opacity));
  el.textContent = content;
  return el;
}

export interface ComposeOptions {
  /** The live plot SVG. */
  chartSvg: SVGSVGElement;
  /** The live HTML legend container, or null when the chart has no legend. */
  legendEl: HTMLElement | null;
  title: string;
  subtitle?: string;
  watermark?: string;
}

/** Compose the full export document. Returns the SVG plus its intrinsic size. */
async function composeExportSvg(
  options: ComposeOptions
): Promise<{ svg: SVGSVGElement; width: number; height: number }> {
  const { chartSvg, legendEl, title, subtitle, watermark = 'aifuturesmodel.com' } = options;
  const { pad } = LAYOUT;

  await waitForStableChartSize(chartSvg);

  const chartRect = chartSvg.getBoundingClientRect();
  const chartWidth = Math.round(
    chartRect.width || chartSvg.width?.baseVal?.value || LAYOUT.minContentWidth
  );
  const chartHeight = Math.round(chartRect.height || chartSvg.height?.baseVal?.value || 400);

  const contentWidth = Math.max(chartWidth, LAYOUT.minContentWidth);
  const width = contentWidth + pad * 2;

  // Measure using the page's live fonts so the wrap matches the embedded ones.
  const bodyFamily = window.getComputedStyle(document.body).fontFamily || 'sans-serif';
  const legendFont = `500 ${LAYOUT.legendSize}px ${bodyFamily}`;

  const legendItems = extractLegendItems(legendEl);
  const legendTextWidth = contentWidth - LAYOUT.legendSwatchWidth - LAYOUT.legendSwatchGap;
  const wrappedLegend = legendItems.map((item) => ({
    item,
    lines: wrapRuns(item.runs, legendTextWidth, legendFont),
  }));

  // ---- vertical layout ----
  let y = pad;

  const titleBaseline = y + LAYOUT.titleSize;
  y = titleBaseline + 8;

  let subtitleBaseline = 0;
  if (subtitle) {
    subtitleBaseline = y + LAYOUT.subtitleSize;
    y = subtitleBaseline + 6;
  }

  y += 16;
  const legendTop = y;
  for (const { lines } of wrappedLegend) {
    y += lines.length * LAYOUT.legendLineHeight + LAYOUT.legendRowGap;
  }
  if (wrappedLegend.length > 0) y += 10;

  const chartTop = y;
  y = chartTop + chartHeight;

  const footerRuleY = y + 14;
  const watermarkBaseline = footerRuleY + 6 + LAYOUT.watermarkSize;
  const height = Math.round(watermarkBaseline + pad * 0.5);

  // ---- build ----
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('xmlns', SVG_NS);
  svg.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
  svg.setAttribute('width', String(width));
  svg.setAttribute('height', String(height));
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

  const defs = document.createElementNS(SVG_NS, 'defs');
  const style = document.createElementNS(SVG_NS, 'style');
  style.textContent = await getEmbeddedFontCss();
  defs.appendChild(style);
  svg.appendChild(defs);

  const background = document.createElementNS(SVG_NS, 'rect');
  background.setAttribute('x', '0');
  background.setAttribute('y', '0');
  background.setAttribute('width', String(width));
  background.setAttribute('height', String(height));
  background.setAttribute('fill', COLORS.background);
  svg.appendChild(background);

  svg.appendChild(
    createText(title, pad, titleBaseline, {
      size: LAYOUT.titleSize,
      family: F.serif,
      fill: COLORS.foreground,
    })
  );

  if (subtitle) {
    svg.appendChild(
      createText(subtitle, pad, subtitleBaseline, {
        size: LAYOUT.subtitleSize,
        family: F.sans,
        fill: COLORS.muted,
      })
    );
  }

  // ---- legend ----
  let legendY = legendTop;
  for (const { item, lines } of wrappedLegend) {
    const firstBaseline = legendY + LAYOUT.legendSize;

    const swatch = document.createElementNS(SVG_NS, 'line');
    swatch.setAttribute('x1', String(pad));
    swatch.setAttribute('x2', String(pad + LAYOUT.legendSwatchWidth));
    swatch.setAttribute('y1', String(firstBaseline - LAYOUT.legendSize / 2 + 1));
    swatch.setAttribute('y2', String(firstBaseline - LAYOUT.legendSize / 2 + 1));
    swatch.setAttribute('stroke', item.swatchColor);
    swatch.setAttribute('stroke-width', '2');
    swatch.setAttribute('opacity', String(item.opacity));
    if (item.swatchDash) swatch.setAttribute('stroke-dasharray', item.swatchDash);
    svg.appendChild(swatch);

    const textX = pad + LAYOUT.legendSwatchWidth + LAYOUT.legendSwatchGap;
    lines.forEach((segments, lineIndex) => {
      const textEl = document.createElementNS(SVG_NS, 'text');
      textEl.setAttribute('x', String(textX));
      textEl.setAttribute('y', String(firstBaseline + lineIndex * LAYOUT.legendLineHeight));
      textEl.setAttribute('font-size', String(LAYOUT.legendSize));
      textEl.setAttribute('font-weight', '500');
      textEl.setAttribute('opacity', String(item.opacity));
      textEl.style.setProperty('font-family', F.sans);
      for (const segment of segments) {
        const tspan = document.createElementNS(SVG_NS, 'tspan');
        tspan.setAttribute('fill', segment.color);
        tspan.textContent = segment.text;
        textEl.appendChild(tspan);
      }
      svg.appendChild(textEl);
    });

    legendY += lines.length * LAYOUT.legendLineHeight + LAYOUT.legendRowGap;
  }

  // ---- plot ----
  const chartClone = sanitizeChartSvg(chartSvg);
  // Left-aligned with the title and legend; centring a narrow plot inside a
  // wider legend block reads as a layout mistake.
  chartClone.setAttribute('x', String(pad));
  chartClone.setAttribute('y', String(chartTop));
  chartClone.setAttribute('width', String(chartWidth));
  chartClone.setAttribute('height', String(chartHeight));
  chartClone.removeAttribute('style');
  svg.appendChild(chartClone);

  // ---- footer ----
  const rule = document.createElementNS(SVG_NS, 'line');
  rule.setAttribute('x1', String(pad));
  rule.setAttribute('x2', String(width - pad));
  rule.setAttribute('y1', String(footerRuleY));
  rule.setAttribute('y2', String(footerRuleY));
  rule.setAttribute('stroke', COLORS.hairline);
  rule.setAttribute('stroke-width', '1');
  svg.appendChild(rule);

  svg.appendChild(
    createText(watermark, width - pad, watermarkBaseline, {
      size: LAYOUT.watermarkSize,
      family: F.sans,
      fill: COLORS.watermark,
      anchor: 'end',
    })
  );

  return { svg, width, height };
}

// ---------------------------------------------------------------------------
// Rasterising
// ---------------------------------------------------------------------------

/** UTF-8 safe base64 data URL (labels contain apostrophes and middots). */
function svgToDataUrl(svgString: string): string {
  const withHeader = `<?xml version="1.0" encoding="UTF-8"?>\n${svgString}`;
  const bytes = new TextEncoder().encode(withHeader);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return `data:image/svg+xml;base64,${btoa(binary)}`;
}

async function rasterize(
  svg: SVGSVGElement,
  width: number,
  height: number,
  scale: number
): Promise<Blob> {
  const svgString = new XMLSerializer().serializeToString(svg);
  const dataUrl = svgToDataUrl(svgString);

  const img = new Image();
  img.width = width;
  img.height = height;

  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('Failed to rasterise the exported SVG'));
    img.src = dataUrl;
  });
  // decode() resolves only once embedded fonts are applied.
  if (typeof img.decode === 'function') {
    try {
      await img.decode();
    } catch {
      /* Safari can reject an already-loaded image; the onload above is enough. */
    }
  }

  const canvas = document.createElement('canvas');
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);

  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Failed to get canvas 2D context');
  ctx.fillStyle = COLORS.background;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.drawImage(img, 0, 0, width, height);

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Failed to encode PNG'))),
      'image/png',
      1.0
    );
  });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface ForecastChartExportOptions extends ComposeOptions {
  filename: string;
  /** Pixel density multiplier; 3x by default for presentation use. */
  scale?: number;
}

/** Compose, rasterise and download a forecast chart as a PNG. */
export async function exportForecastChartPng(
  options: ForecastChartExportOptions
): Promise<void> {
  const { filename, scale = 3, ...composeOptions } = options;

  // Make sure the page's own webfonts are ready before measuring text.
  if (document.fonts?.ready) await document.fonts.ready;

  const { svg, width, height } = await composeExportSvg(composeOptions);
  const blob = await rasterize(svg, width, height, scale);
  downloadBlob(blob, filename.endsWith('.png') ? filename : `${filename}.png`);
}

const MONTH_NAMES = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

/**
 * Forecast dates are stored in the site's compact `MM-DD-YY` form (e.g.
 * "08-10-26"). That reads as ambiguous in a standalone image, so spell it out
 * for the export subtitle. Anything that isn't that shape is passed through.
 */
export function formatForecastDateLabel(date: string | undefined): string | undefined {
  if (!date) return date;
  const match = /^(\d{2})-(\d{2})-(\d{2})$/.exec(date.trim());
  if (!match) return date;

  const month = Number(match[1]);
  const year = Number(match[3]);
  if (month < 1 || month > 12) return date;

  return `${MONTH_NAMES[month - 1]} 20${String(year).padStart(2, '0')}`;
}

/** `aifm_<simulationId>_<chart>_<yyyy-mm-dd>.png` */
export function buildExportFilename(simulationId: string | undefined, chartSlug: string): string {
  const now = new Date();
  const date = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
  ].join('-');
  const sim = (simulationId || 'forecast').replace(/[^a-zA-Z0-9._-]+/g, '-');
  return `aifm_${sim}_${chartSlug}_${date}.png`;
}
