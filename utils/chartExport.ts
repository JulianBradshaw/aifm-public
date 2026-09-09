/**
 * Chart Export Utility
 * Converts SVG charts and DOM elements to PNG images for sharing
 */

import { toBlob } from 'html-to-image';

// Colors from the codebase (globals.css)
export const EXPORT_COLORS = {
  background: '#fffff8', // --vivid-background
  foreground: '#0D0D0D', // --vivid-foreground / primary
  graphGreen: '#2A623D', // --slowdown-background (used for graph lines)
} as const;

export interface ExportOptions {
  width: number;
  height: number;
  scale?: number; // For retina displays, default 2x
  backgroundColor?: string;
  padding?: number;
}

export const EXPORT_PRESETS = {
  combined: { width: 1200, height: 675, scale: 2 },  // Twitter/LinkedIn optimal (16:9)
  individual: { width: 1200, height: 675, scale: 2 }, // Single chart - also Twitter optimized
} as const;

// Font embedding cache
let fontCache: { roman: string | null; bold: string | null } = { roman: null, bold: null };

/**
 * Fetch a font file and convert to base64
 */
async function fetchFontAsBase64(url: string): Promise<string> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      console.warn(`Failed to fetch font: ${url}`);
      return '';
    }
    const buffer = await response.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  } catch (error) {
    console.warn(`Error fetching font ${url}:`, error);
    return '';
  }
}

/**
 * Load and cache fonts for embedding
 */
async function loadFonts(): Promise<{ roman: string; bold: string }> {
  if (fontCache.roman && fontCache.bold) {
    return { roman: fontCache.roman, bold: fontCache.bold };
  }

  const [roman, bold] = await Promise.all([
    fetchFontAsBase64('/fonts/et-book/et-book-roman-line-figures.woff'),
    fetchFontAsBase64('/fonts/et-book/et-book-bold-line-figures.woff'),
  ]);

  fontCache = { roman, bold };
  return { roman, bold };
}

/**
 * Create a style element with embedded fonts
 */
async function createFontStyleElement(): Promise<string> {
  const { roman, bold } = await loadFonts();

  let fontFaces = '';

  if (roman) {
    fontFaces += `
      @font-face {
        font-family: 'et-book';
        src: url(data:font/woff;base64,${roman}) format('woff');
        font-weight: normal;
        font-style: normal;
      }
    `;
  }

  if (bold) {
    fontFaces += `
      @font-face {
        font-family: 'et-book';
        src: url(data:font/woff;base64,${bold}) format('woff');
        font-weight: bold;
        font-style: normal;
      }
    `;
  }

  return fontFaces;
}

/**
 * Inline all computed styles into SVG elements
 * This is necessary because canvas cannot access external stylesheets
 */
function inlineStyles(svg: SVGSVGElement): void {
  const elements = svg.querySelectorAll('*');

  elements.forEach((el) => {
    if (!(el instanceof SVGElement || el instanceof HTMLElement)) return;

    const computedStyle = window.getComputedStyle(el);

    // Key properties to inline for SVG rendering
    const propertiesToInline = [
      'fill',
      'stroke',
      'stroke-width',
      'stroke-dasharray',
      'stroke-opacity',
      'fill-opacity',
      'opacity',
      'font-family',
      'font-size',
      'font-weight',
      'text-anchor',
      'dominant-baseline',
      'alignment-baseline',
      'letter-spacing',
    ];

    propertiesToInline.forEach((prop) => {
      const value = computedStyle.getPropertyValue(prop);
      if (value && value !== 'none' && value !== '') {
        (el as SVGElement).style.setProperty(prop, value);
      }
    });
  });
}

/**
 * Embed fonts into an SVG element
 */
async function embedFontsInSvg(svg: SVGSVGElement): Promise<void> {
  const fontStyles = await createFontStyleElement();
  if (!fontStyles) return;

  // Check if defs element exists, create if not
  let defs = svg.querySelector('defs');
  if (!defs) {
    defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    svg.insertBefore(defs, svg.firstChild);
  }

  // Create and insert style element
  const styleEl = document.createElementNS('http://www.w3.org/2000/svg', 'style');
  styleEl.textContent = fontStyles;
  defs.appendChild(styleEl);
}

/**
 * Serialize SVG element to a data URL with embedded fonts
 */
export async function svgToDataURL(svgElement: SVGSVGElement): Promise<string> {
  // Clone the SVG to avoid modifying the original
  const clone = svgElement.cloneNode(true) as SVGSVGElement;

  // Embed fonts
  await embedFontsInSvg(clone);

  // Inline styles
  inlineStyles(clone);

  // Add XML namespace if not present
  if (!clone.getAttribute('xmlns')) {
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  }

  // Serialize to string
  const serializer = new XMLSerializer();
  const svgString = serializer.serializeToString(clone);

  // Encode as data URL
  const encoded = encodeURIComponent(svgString)
    .replace(/'/g, '%27')
    .replace(/"/g, '%22');

  return `data:image/svg+xml,${encoded}`;
}

/**
 * Convert SVG element to PNG Blob via canvas
 */
export async function svgToPngBlob(
  svgElement: SVGSVGElement,
  options: ExportOptions
): Promise<Blob> {
  const { width, height, scale = 2, backgroundColor = EXPORT_COLORS.background } = options;

  // Get the SVG data URL
  const svgDataUrl = await svgToDataURL(svgElement);

  // Create canvas
  const canvas = document.createElement('canvas');
  canvas.width = width * scale;
  canvas.height = height * scale;

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to get canvas 2D context');
  }

  // Set background color
  ctx.fillStyle = backgroundColor;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Scale for retina
  ctx.scale(scale, scale);

  // Load and draw the SVG
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => {
      // Calculate scaling to fit the SVG into the canvas
      const svgWidth = svgElement.viewBox?.baseVal?.width || svgElement.clientWidth || width;
      const svgHeight = svgElement.viewBox?.baseVal?.height || svgElement.clientHeight || height;

      // Calculate scale to fit
      const scaleX = width / svgWidth;
      const scaleY = height / svgHeight;
      const fitScale = Math.min(scaleX, scaleY);

      // Center the image
      const drawWidth = svgWidth * fitScale;
      const drawHeight = svgHeight * fitScale;
      const offsetX = (width - drawWidth) / 2;
      const offsetY = (height - drawHeight) / 2;

      ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);

      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to create PNG blob'));
        }
      }, 'image/png', 1.0);
    };

    img.onerror = () => {
      reject(new Error('Failed to load SVG image'));
    };

    img.src = svgDataUrl;
  });
}

/**
 * Trigger a browser download for a Blob
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Main export function - export SVG chart as PNG
 */
export async function exportChartAsPng(
  svgElement: SVGSVGElement,
  filename: string,
  options: ExportOptions
): Promise<void> {
  const blob = await svgToPngBlob(svgElement, options);
  const filenameWithExt = filename.endsWith('.png') ? filename : `${filename}.png`;
  downloadBlob(blob, filenameWithExt);
}

/**
 * Export a DOM element (like a div containing charts) to PNG using html-to-image
 * Properly captures CSS layout, fonts, and SVG content
 */
export async function exportElementAsPng(
  element: HTMLElement,
  filename: string,
  options: ExportOptions
): Promise<void> {
  const { width, height, scale = 2, backgroundColor = EXPORT_COLORS.background } = options;

  // Use html-to-image to capture the element exactly as rendered
  const blob = await toBlob(element, {
    width,
    height,
    pixelRatio: scale,
    backgroundColor,
    cacheBust: true,
    // Remove any CSS transform on the element (used for preview scaling)
    style: {
      transform: 'none',
      transformOrigin: 'top left',
    },
  });

  if (!blob) {
    throw new Error('Failed to create PNG blob');
  }

  const filenameWithExt = filename.endsWith('.png') ? filename : `${filename}.png`;
  downloadBlob(blob, filenameWithExt);
}

/**
 * Copy text to clipboard with fallback
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }

    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    const success = document.execCommand('copy');
    document.body.removeChild(textArea);
    return success;
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return false;
  }
}

/**
 * Format a year as month + year string (e.g., "Mar 2027")
 * Returns ">2045" for dates beyond simulation end
 */
export function formatMilestoneDate(year: number | null | undefined, simulationEndYear = 2045): string {
  if (year == null || !Number.isFinite(year)) {
    return 'N/A';
  }

  if (year > simulationEndYear) {
    return `>${simulationEndYear}`;
  }

  const wholeYear = Math.floor(year);
  const monthFraction = year - wholeYear;
  const monthIndex = Math.round(monthFraction * 12);

  const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const month = monthNames[Math.min(monthIndex, 11)];

  return `${month} ${wholeYear}`;
}
