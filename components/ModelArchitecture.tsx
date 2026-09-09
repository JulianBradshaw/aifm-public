'use client';

import Link from 'next/link';
import ModelDiagram from '@/svgs/model-diagram-semantic.svg';
import { useCallback, useEffect, useRef, useState, type MouseEvent as ReactMouseEvent, type ReactNode } from 'react';
import LazyExplanationsHtml from './LazyExplanationsHtml';
import { useParameterHover, PARAMETER_TO_SVG_NODES } from './ParameterHoverContext';

// Override mapping: SVG node label -> heading text to scroll to
// Keys are normalized labels (whitespace collapsed to single spaces)
export const SVG_LABEL_TO_HEADING: Record<string, string> = {
  'Automation compute': 'Compute forecasts',
  'Experiment compute': 'Compute forecasts',
  'Training compute': 'Compute forecasts',
  'Human coding labor': 'Aggregate coding labor',
  'Software research effort': 'Research effort and AI R&D uplift',
  'Automated Coder Time Horizon': 'Time horizon and the Automated Coder milestone',
  'Effective Compute': 'Modeling effective compute, EL version',
  'Software efficiency': 'Software efficiency',
  'Experiment throughput': 'Experiment throughput',
  'Aggregate coding labor': 'Aggregate coding labor',
  'Coding automation fraction and efficiency': 'Coding automation fraction',
  'Human experiment selection skill': 'Human research taste',
  'Automated experiment selection skill': 'Automated research taste',
  'Aggregate experiment selection skill': 'Aggregate research taste',
};

// Reverse mapping: heading match key -> array of SVG data-label values that link to it
// Built from SVG_LABEL_TO_HEADING
const buildHeadingToSvgLabels = (): Record<string, string[]> => {
  const result: Record<string, string[]> = {};
  for (const [svgLabel, heading] of Object.entries(SVG_LABEL_TO_HEADING)) {
    const key = heading.toLowerCase().replace(/[^a-z0-9]+/g, '');
    if (!result[key]) {
      result[key] = [];
    }
    result[key].push(svgLabel);
  }
  return result;
};

const HEADING_TO_SVG_LABELS = buildHeadingToSvgLabels();

type SectionEntry = {
  label: string;
  x: number | null;
};

const normalizeLabel = (input: string | null | undefined) => {
  if (!input) {
    return null;
  }
  const normalized = input.replace(/\s+/g, ' ').trim();
  return normalized.length > 0 ? normalized : null;
};

const normalizeHeadingText = (input: string | null | undefined) => {
  if (!input) {
    return null;
  }
  const withoutPrefix = input.replace(/^[\d\s.:()-]+/, '');
  return normalizeLabel(withoutPrefix);
};

const buildMatchKey = (input: string | null | undefined) => {
  const normalized = normalizeLabel(input);
  if (!normalized) {
    return null;
  }
  return normalized.toLowerCase().replace(/[^a-z0-9]+/g, '');
};

const parseXCoordinate = (element: Element | null) => {
  if (!element) {
    return null;
  }
  const rawValue = element.getAttribute('x');
  if (!rawValue) {
    return null;
  }
  const parsed = Number.parseFloat(rawValue);
  return Number.isNaN(parsed) ? null : parsed;
};

const findNearestSectionLabel = (sections: SectionEntry[], nodeX: number | null) => {
  if (sections.length === 0) {
    return null;
  }
  if (nodeX == null) {
    return sections[0].label;
  }
  const nearest = sections.reduce<SectionEntry>((best, candidate) => {
    if (candidate.x == null) {
      return best;
    }
    if (best.x == null) {
      return candidate;
    }
    const bestDistance = Math.abs(best.x - nodeX);
    const candidateDistance = Math.abs(candidate.x - nodeX);
    return candidateDistance < bestDistance ? candidate : best;
  }, sections[0]);

  return nearest.label;
};

const findForeignObject = (element: Element | null) => {
  let current: Element | null = element;
  while (current && current.tagName.toLowerCase() !== 'foreignobject') {
    current = current.parentElement;
  }

  return current;
};

type DiagramLabelElement = HTMLElement | SVGTextElement;

const isSvgTextElement = (element: Element): element is SVGTextElement =>
  element.tagName.toLowerCase() === 'text';

const isTspanElement = (element: Element): element is SVGTSpanElement =>
  element.tagName.toLowerCase() === 'tspan';

const isParagraphSection = (element: HTMLElement) =>
  element.style.fontStyle === 'italic' && element.style.fontSize === '48px';

const TEXT_FONT_THRESHOLD = 12;

const shouldIncludeSvgTextElement = (element: SVGTextElement) => {
  const label = normalizeLabel(element.textContent);
  if (!label) {
    return false;
  }
  const fontSizeAttr = element.getAttribute('font-size');
  const fontSize = fontSizeAttr ? Number.parseFloat(fontSizeAttr) : Number.NaN;
  if (!Number.isNaN(fontSize) && fontSize < TEXT_FONT_THRESHOLD) {
    return false;
  }
  return true;
};

const getLabelElements = (svg: SVGElement): DiagramLabelElement[] => {
  const paragraphElements = Array.from(svg.querySelectorAll('foreignObject p')) as HTMLElement[];
  const textElements = Array.from(svg.querySelectorAll('text')).filter((text): text is SVGTextElement =>
    isSvgTextElement(text) && shouldIncludeSvgTextElement(text),
  );
  return [...paragraphElements, ...textElements];
};

const getElementXCoordinate = (element: DiagramLabelElement): number | null => {
  if (element instanceof SVGGraphicsElement) {
    try {
      const bbox = element.getBBox();
      return Number.isFinite(bbox.x) ? bbox.x : null;
    } catch {
      return null;
    }
  }
  const foreignObject = findForeignObject(element);
  return parseXCoordinate(foreignObject);
};

const ensureOriginalColor = (element: DiagramLabelElement) => {
  if (isSvgTextElement(element)) {
    if (!element.hasAttribute('data-original-fill')) {
      const fill = element.getAttribute('fill') ?? '#1E1E1E';
      element.setAttribute('data-original-fill', fill);
    }
  } else if (!element.hasAttribute('data-original-color')) {
    const computedStyle = window.getComputedStyle(element);
    element.setAttribute('data-original-color', computedStyle.color);
  }
};

const restoreOriginalColor = (element: DiagramLabelElement) => {
  if (isSvgTextElement(element)) {
    const originalFill = element.getAttribute('data-original-fill');
    if (originalFill) {
      element.setAttribute('fill', originalFill);
    }
  } else {
    const originalColor = element.getAttribute('data-original-color');
    if (originalColor) {
      element.style.color = originalColor;
    }
  }
};

const setTextHighlightColor = (element: DiagramLabelElement, color: string) => {
  if (isSvgTextElement(element)) {
    element.setAttribute('fill', color);
  } else {
    element.style.color = color;
  }
};

const findLabelElement = (target: EventTarget | null): DiagramLabelElement | null => {
  let current: Element | null = null;
  if (target instanceof Element) {
    current = target;
  } else if (target instanceof Node) {
    current = target.parentElement;
  }

  while (current) {
    const tagName = current.tagName.toLowerCase();
    if (tagName === 'p' || tagName === 'text') {
      return current as DiagramLabelElement;
    }
    if (tagName === 'tspan') {
      current = current.parentElement;
      continue;
    }
    current = current.parentElement;
  }

  return null;
};

type PointerPosition = {
  x: number;
  y: number;
};

const MAX_POINTER_DISTANCE = 140;

const getNearestLabelElement = (svg: SVGElement, pointer: PointerPosition): DiagramLabelElement | null => {
  const labelElements = getLabelElements(svg);
  let closestElement: DiagramLabelElement | null = null;
  let closestDistance = Number.POSITIVE_INFINITY;

  labelElements.forEach((element) => {
    const rect = element.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) {
      return;
    }
    const clampedX = Math.min(Math.max(pointer.x, rect.left), rect.right);
    const clampedY = Math.min(Math.max(pointer.y, rect.top), rect.bottom);
    const dx = pointer.x - clampedX;
    const dy = pointer.y - clampedY;
    const distance = Math.hypot(dx, dy);
    if (distance < closestDistance) {
      closestDistance = distance;
      closestElement = element;
    }
  });

  if (closestDistance <= MAX_POINTER_DISTANCE) {
    return closestElement;
  }
  return null;
};

const buildSectionEntries = (svg: SVGElement): SectionEntry[] =>
  Array.from(svg.querySelectorAll('foreignObject')).reduce<SectionEntry[]>(
    (acc, foreignObject) => {
      const paragraph = foreignObject.querySelector('p');
      if (!paragraph) {
        return acc;
      }
      const label = normalizeLabel(paragraph.textContent);
      if (!label) {
        return acc;
      }

      const paragraphElement = paragraph as HTMLElement;
      const isSection =
        paragraphElement.style.fontStyle === 'italic' &&
        paragraphElement.style.fontSize === '48px';

      if (!isSection) {
        return acc;
      }

      acc.push({
        label,
        x: parseXCoordinate(foreignObject),
      });
      return acc;
    },
    [],
  );

const hasValidTarget = (label: string | null, headingLookup: Map<string, HTMLElement>): boolean => {
  if (!label) {
    return false;
  }

  // Check override mapping
  const overrideHeading = SVG_LABEL_TO_HEADING[label];
  const key = buildMatchKey(overrideHeading ?? label);
  if (!key) {
    return false;
  }

  let target = headingLookup.get(key);
  if (!target) {
    for (const [storedKey, element] of headingLookup.entries()) {
      if (storedKey.includes(key) || key.includes(storedKey)) {
        target = element;
        break;
      }
    }
  }

  return target != null;
};

type ModelArchitectureProps = {
  modelDescriptionGDocPortionMarkdown?: React.ReactNode;
  introContent?: ReactNode;
  scrollContainerRef?: React.RefObject<HTMLDivElement | null>;
};

// ToC link component - uses proper <a> tags for shareability
interface TocLinkProps {
  label: string;
  children: React.ReactNode;
  className?: string;
  onClick: (label: string) => void;
  headingLookup: React.RefObject<Map<string, HTMLElement>>;
}

function TocLink({ label, children, className, onClick, headingLookup }: TocLinkProps) {
  const key = buildMatchKey(SVG_LABEL_TO_HEADING[label] ?? label);
  
  // Try to get the actual element ID from the heading lookup
  let elementId: string | null = null;
  if (key && headingLookup.current) {
    let target = headingLookup.current.get(key);
    if (!target) {
      // Partial matching like scrollToHeading
      for (const [storedKey, element] of headingLookup.current.entries()) {
        if (storedKey.includes(key) || key.includes(storedKey)) {
          target = element;
          break;
        }
      }
    }
    if (target) {
      elementId = target.id;
    }
  }
  
  // Fall back to computed ID if not found in lookup yet
  const href = elementId ? `#${elementId}` : (key ? `#section-${key}` : '#');
  
  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    onClick(label);
  };

  return (
    <a
      href={href}
      onClick={handleClick}
      className={className}
    >
      {children}
    </a>
  );
}

export default function ModelArchitecture({ modelDescriptionGDocPortionMarkdown, introContent, scrollContainerRef }: ModelArchitectureProps) {
  const explanationContainerRef = useRef<HTMLDivElement | null>(null);
  const introContentRef = useRef<HTMLDivElement | null>(null);
  const gDocContentRef = useRef<HTMLDivElement | null>(null);
  const headingLookupRef = useRef<Map<string, HTMLElement>>(new Map());
  const svgContainerRef = useRef<HTMLDivElement | null>(null);
  const mobileSvgContainerRef = useRef<HTMLDivElement | null>(null);
  const mobileContentRef = useRef<HTMLDivElement | null>(null);
  const [isHoveringSvg, setIsHoveringSvg] = useState(false);
  const [isMobileDiagramVisible, setIsMobileDiagramVisible] = useState(false);
  const [activeSectionKey, setActiveSectionKey] = useState<string | null>(null);
  const [isDiagramCollapsed, setIsDiagramCollapsed] = useState(false);
  const [isContentsCollapsed, setIsContentsCollapsed] = useState(false);
  const [isDiagramModalOpen, setIsDiagramModalOpen] = useState(false);
  const { hoveredParameter } = useParameterHover();

  // Track scroll position to show/hide mobile diagram
  useEffect(() => {
    const SCROLL_THRESHOLD = 500; // Show diagram after scrolling 500px into the essay

    const handleScroll = () => {
      const contentEl = mobileContentRef.current;
      const scrollContainer = scrollContainerRef?.current;
      if (!contentEl) return;

      // Get the scroll position relative to the content
      // Use the scroll container if provided, otherwise fall back to viewport
      if (scrollContainer) {
        const containerRect = scrollContainer.getBoundingClientRect();
        const contentRect = contentEl.getBoundingClientRect();
        // Calculate how far the content has scrolled up relative to the scroll container
        const scrolledIntoContent = Math.max(0, containerRect.top - contentRect.top);
        setIsMobileDiagramVisible(scrolledIntoContent >= SCROLL_THRESHOLD);
      } else {
        // Fallback to viewport-based calculation
        const rect = contentEl.getBoundingClientRect();
        const scrolledIntoContent = Math.max(0, -rect.top);
        setIsMobileDiagramVisible(scrolledIntoContent >= SCROLL_THRESHOLD);
      }
    };

    // Check initial scroll position
    handleScroll();

    // Listen on the scroll container if provided, otherwise window
    const scrollTarget = scrollContainerRef?.current ?? window;
    scrollTarget.addEventListener('scroll', handleScroll, { passive: true });
    return () => scrollTarget.removeEventListener('scroll', handleScroll);
  }, [scrollContainerRef]);

  // State to track when we need to rebuild the heading lookup
  const [headingLookupVersion, setHeadingLookupVersion] = useState(0);

  // Build the heading lookup
  useEffect(() => {
    const nextLookup = new Map<string, HTMLElement>();

    // Helper function to process headings from a container
    const processHeadings = (container: HTMLElement | null, selectors = 'h2, h3') => {
      if (!container) return;
      const headings = container.querySelectorAll(selectors);
      headings.forEach((heading) => {
        const normalizedHeading = normalizeHeadingText(heading.textContent);
        const key = buildMatchKey(normalizedHeading);
        if (!key || nextLookup.has(key)) {
          return;
        }
        const element = heading as HTMLElement;
        if (!element.id) {
          element.id = `section-${key}`;
        }
        if (!element.hasAttribute('tabindex')) {
          element.setAttribute('tabindex', '-1');
        }
        element.style.scrollMarginTop = '8vh';
        nextLookup.set(key, element);
      });
    };

    // Process headings from intro content (includes h1 for "Introducing..." and h2 for stages)
    processHeadings(introContentRef.current, 'h1, h2, h3');
    
    // Process headings from Google Doc content (includes "Introducing the AI Futures Model" and Stages)
    processHeadings(gDocContentRef.current, 'h1, h2, h3');
    
    // Process headings from explanations HTML
    processHeadings(explanationContainerRef.current, 'h2, h3');

    headingLookupRef.current = nextLookup;
  }, [headingLookupVersion]);

  // Watch for DOM changes and rebuild heading lookup when content is added
  useEffect(() => {
    // Use a MutationObserver to detect when content is added
    const observer = new MutationObserver(() => {
      // Trigger a rebuild of the heading lookup
      setHeadingLookupVersion(v => v + 1);
    });

    // Observe the containers for child changes
    const containers = [introContentRef.current, gDocContentRef.current, explanationContainerRef.current];
    containers.forEach(container => {
      if (container) {
        observer.observe(container, { childList: true, subtree: true });
      }
    });

    // No initial bump needed: the builder effect above is declared first, so it
    // already ran against whatever was in the DOM at mount, and anything added
    // later trips the observer. Bumping here just forced a redundant rebuild.

    return () => observer.disconnect();
  }, []);

  // Track which section heading is currently visible using IntersectionObserver
  useEffect(() => {
    const lookup = headingLookupRef.current;
    if (lookup.size === 0) return;

    const headings = Array.from(lookup.values());
    
    // Track visibility ratios for each heading
    const visibilityMap = new Map<HTMLElement, number>();
    
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          visibilityMap.set(entry.target as HTMLElement, entry.intersectionRatio);
        });
        
        // Find the heading with the highest visibility that's actually visible
        let bestHeading: HTMLElement | null = null;
        let bestRatio = 0;
        
        for (const [heading, ratio] of visibilityMap.entries()) {
          if (ratio > bestRatio) {
            bestRatio = ratio;
            bestHeading = heading;
          }
        }
        
        // If nothing is visible enough, find the heading closest to the top of viewport
        if (bestRatio < 0.1) {
          let closestHeading: HTMLElement | null = null;
          let closestDistance = Infinity;
          let anyHeadingScrolledPast = false;
          
          for (const heading of headings) {
            const rect = heading.getBoundingClientRect();
            // Check if any heading has been scrolled past (is above the viewport top + some margin)
            if (rect.top < 150) {
              anyHeadingScrolledPast = true;
            }
            // Look for headings that are above or at the top of the viewport
            // (negative top means it's scrolled past, positive means it's below)
            const distance = Math.abs(rect.top - 100); // 100px offset from top
            if (rect.top < window.innerHeight && distance < closestDistance) {
              closestDistance = distance;
              closestHeading = heading;
            }
          }
          
          // Only set a heading if we've scrolled past at least one heading
          // Otherwise we're at the top of the essay before any sections
          if (anyHeadingScrolledPast) {
            bestHeading = closestHeading;
          } else {
            bestHeading = null;
          }
        }
        
        if (bestHeading) {
          // Find the key for this heading
          for (const [key, element] of lookup.entries()) {
            if (element === bestHeading) {
              setActiveSectionKey(key);
              break;
            }
          }
        } else {
          // No heading is active - clear the highlight
          setActiveSectionKey(null);
        }
      },
      {
        root: scrollContainerRef?.current ?? null,
        rootMargin: '-10% 0px -60% 0px', // Focus on upper portion of viewport
        threshold: [0, 0.1, 0.25, 0.5, 0.75, 1],
      }
    );

    headings.forEach((heading) => observer.observe(heading));

    return () => observer.disconnect();
  }, [scrollContainerRef, headingLookupVersion]);

  // Get the section ID for a given label (used for href and hash)
  const getSectionId = useCallback((label: string): string | null => {
    const overrideHeading = SVG_LABEL_TO_HEADING[label];
    const key = buildMatchKey(overrideHeading ?? label);
    return key ? `section-${key}` : null;
  }, []);

  const scrollToHeading = useCallback((label: string | null, updateHash = true) => {
    if (!label) {
      return;
    }

    // Check for manual override first
    const overrideHeading = SVG_LABEL_TO_HEADING[label];
    const key = buildMatchKey(overrideHeading ?? label);
    if (!key) {
      return;
    }

    const sectionId = `section-${key}`;

    // Check if we're on mobile (lg breakpoint is 1024px)
    const isMobile = window.matchMedia('(max-width: 1023px)').matches;
    
    // On mobile, find the heading in the mobile content container (includes intro + essay)
    if (isMobile) {
      const mobileContent = mobileContentRef.current;
      if (mobileContent) {
        // Search for h1, h2, h3 to include intro content headings
        const headings = mobileContent.querySelectorAll('h1, h2, h3');
        for (const heading of headings) {
          const headingKey = buildMatchKey(normalizeHeadingText(heading.textContent));
          if (headingKey && (headingKey.includes(key) || key.includes(headingKey))) {
            const element = heading as HTMLElement;
            // Ensure the element has an ID for the hash
            if (!element.id) {
              element.id = `section-${headingKey}`;
            }
            element.scrollIntoView({
              behavior: 'smooth',
              block: 'start',
            });
            // Update URL hash
            if (updateHash && element.id) {
              history.pushState(null, '', `#${element.id}`);
            }
            return;
          }
        }
      }
      return;
    }

    // Desktop: use the lookup ref
    const lookup = headingLookupRef.current;
    let target = lookup.get(key);
    if (!target) {
      for (const [storedKey, element] of lookup.entries()) {
        if (storedKey.includes(key) || key.includes(storedKey)) {
          target = element;
          break;
        }
      }
    }

    if (!target) {
      return;
    }

    target.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });

    // Update URL hash
    if (updateHash && target.id) {
      history.pushState(null, '', `#${target.id}`);
    }
  }, []);

  useEffect(() => {
    const container = svgContainerRef.current;
    if (!container) {
      return;
    }

    const svg = container.querySelector('svg');
    if (!svg) {
      return;
    }

    const handleMouseEnter = () => {
      setIsHoveringSvg(true);
    };

    const handleMouseLeave = () => {
      setIsHoveringSvg(false);
    };

    svg.addEventListener('mouseenter', handleMouseEnter);
    svg.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      svg.removeEventListener('mouseenter', handleMouseEnter);
      svg.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);

  useEffect(() => {
    const container = svgContainerRef.current;
    if (!container) {
      return;
    }

    const svg = container.querySelector('svg');
    if (!svg) {
      return;
    }

    const headingLookup = headingLookupRef.current;
    const sectionEntries = buildSectionEntries(svg);
    const labelElements = getLabelElements(svg);

    labelElements.forEach((element) => {
      const label = normalizeLabel(element.textContent);

      if (!label) {
        element.style.textDecoration = '';
        element.style.textDecorationStyle = '';
        element.style.textDecorationColor = '';
        element.style.cursor = '';
        return;
      }

      ensureOriginalColor(element);

      const isSection =
        element instanceof HTMLElement ? isParagraphSection(element) : false;

      const nodeX = getElementXCoordinate(element);
      const sectionLabel = isSection ? label : findNearestSectionLabel(sectionEntries, nodeX);
      const nodeLabel = isSection ? null : label;
      const targetLabel = nodeLabel ?? sectionLabel;

      const isValidLink = hasValidTarget(targetLabel, headingLookup) && !isSection;

      if (isHoveringSvg && isValidLink) {
        element.style.textDecorationStyle = 'dotted';
        element.style.textDecorationColor = '#2A623D';
        element.style.textDecorationThickness = '3px';
        element.style.cursor = 'pointer';
      } else {
        element.style.textDecoration = '';
        element.style.textDecorationStyle = '';
        element.style.textDecorationColor = '';
        element.style.cursor = '';
        restoreOriginalColor(element);
      }
    });
  }, [isHoveringSvg]);

  useEffect(() => {
    const container = svgContainerRef.current;
    if (!container) {
      return;
    }

    const svg = container.querySelector('svg');
    if (!svg) {
      return;
    }

    const highlightLabels = hoveredParameter ? PARAMETER_TO_SVG_NODES[hoveredParameter] : null;

    const labelElements = getLabelElements(svg);
    labelElements.forEach((element) => {
      const label = normalizeLabel(element.textContent);

      if (!label) {
        return;
      }

      ensureOriginalColor(element);

      const shouldHighlight = highlightLabels?.includes(label) ?? false;

      if (shouldHighlight) {
        if (element instanceof HTMLElement) {
          element.style.color = '#8b0000';
          element.style.fontWeight = 'bold';
          element.style.padding = '2px 4px';
          element.style.borderRadius = '3px';
          element.style.transition = 'all 0.15s ease-in-out';
        } else {
          element.style.fontWeight = 'bold';
          setTextHighlightColor(element, '#8b0000');
        }
      } else {
        if (element instanceof HTMLElement) {
          element.style.backgroundColor = '';
          element.style.fontWeight = '';
          element.style.padding = '';
          element.style.borderRadius = '';
          restoreOriginalColor(element);
        } else {
          element.style.fontWeight = '';
          restoreOriginalColor(element);
        }
      }
    });
  }, [hoveredParameter]);

  // Highlight SVG nodes corresponding to the currently visible essay section
  useEffect(() => {
    const applyHighlighting = (container: HTMLDivElement | null) => {
      if (!container) return;
      
      const svg = container.querySelector('svg');
      if (!svg) return;
      
      const diagramBoxes = svg.querySelectorAll('[data-label]');
      
      // Find which SVG labels should be highlighted based on current section
      let labelsToHighlight: string[] = [];
      
      if (activeSectionKey) {
        // Check direct match first
        if (HEADING_TO_SVG_LABELS[activeSectionKey]) {
          labelsToHighlight = HEADING_TO_SVG_LABELS[activeSectionKey];
        } else {
          // Try partial matching (similar to how scrollToHeading works)
          for (const [headingKey, svgLabels] of Object.entries(HEADING_TO_SVG_LABELS)) {
            if (headingKey.includes(activeSectionKey) || activeSectionKey.includes(headingKey)) {
              labelsToHighlight = svgLabels;
              break;
            }
          }
        }
      }
      
      diagramBoxes.forEach((box) => {
        const dataLabel = box.getAttribute('data-label');
        const shouldHighlight = dataLabel ? labelsToHighlight.includes(dataLabel) : false;
        
        const bgRect = box.querySelector('[data-box-bg]') as SVGRectElement | null;
        const textElements = box.querySelectorAll('text, tspan');
        
        if (shouldHighlight) {
          // Apply "you are here" highlight style - green background
          if (bgRect) {
            bgRect.setAttribute('fill', '#2A623D');
            bgRect.setAttribute('fill-opacity', '0.15');
          }
          textElements.forEach((el) => {
            el.setAttribute('fill', '#1a4d2e');
            el.setAttribute('fill-opacity', '1');
          });
          // Add a subtle border effect by using box-shadow-like approach with stroke
          const strokeRect = box.querySelector('rect:not([data-box-bg])') as SVGRectElement | null;
          if (strokeRect) {
            strokeRect.setAttribute('stroke', '#2A623D');
            strokeRect.setAttribute('stroke-width', '2');
          }
        } else {
          // Reset to default - but don't override hover state
          // The hover handlers in setupSvg will handle mouseenter/mouseleave
          if (bgRect) {
            bgRect.setAttribute('fill', 'transparent');
            bgRect.removeAttribute('fill-opacity');
          }
          textElements.forEach((el) => {
            // Restore original colors
            el.setAttribute('fill', 'black');
            el.setAttribute('fill-opacity', '0.8');
          });
          const strokeRect = box.querySelector('rect:not([data-box-bg])') as SVGRectElement | null;
          if (strokeRect) {
            strokeRect.setAttribute('stroke', 'black');
            strokeRect.setAttribute('stroke-opacity', '0.8');
            strokeRect.setAttribute('stroke-width', '1');
          }
        }
      });
    };
    
    applyHighlighting(svgContainerRef.current);
    applyHighlighting(mobileSvgContainerRef.current);
  }, [activeSectionKey]);

  const navigateFromSvgTarget = useCallback((eventTarget: EventTarget | null, svg: SVGElement, pointer?: PointerPosition) => {
    // First, check if we clicked on a group with a data-label
    let element = eventTarget as Element | null;
    while (element && element !== svg) {
      const dataLabel = element.getAttribute('data-label');
      if (dataLabel) {
        scrollToHeading(dataLabel);
        console.log('model-architecture-click', { box: dataLabel });
        return;
      }
      element = element.parentElement;
    }

    // Fall back to original label element logic
    let labelElement = findLabelElement(eventTarget);
    if (!labelElement && pointer) {
      labelElement = getNearestLabelElement(svg, pointer);
    }
    if (!labelElement) {
      return;
    }

    const label = normalizeLabel(labelElement.textContent);
    if (!label) {
      return;
    }

    const nodeX = getElementXCoordinate(labelElement);

    const isSection =
      labelElement instanceof HTMLElement ? isParagraphSection(labelElement) : false;

    const sectionEntries = buildSectionEntries(svg);

    const sectionLabel = isSection ? label : findNearestSectionLabel(sectionEntries, nodeX);

    const nodeLabel = isSection ? null : label;
    scrollToHeading(nodeLabel ?? sectionLabel);

    console.log('model-architecture-click', {
      node: nodeLabel ?? undefined,
      section: sectionLabel ?? undefined,
    });
  }, [scrollToHeading]);

  const handleSvgClick = useCallback((event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();

    const svg = event.currentTarget as SVGElement | null;
    if (!svg) {
      return;
    }

    navigateFromSvgTarget(event.target, svg, { x: event.clientX, y: event.clientY });
  }, [navigateFromSvgTarget]);

  const handleContainerClick = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    // Find the SVG within the clicked container (works for both desktop and mobile)
    const container = event.currentTarget as HTMLDivElement;
    const svg = container.querySelector('svg');
    if (!svg) {
      return;
    }
    navigateFromSvgTarget(event.target, svg, { x: event.clientX, y: event.clientY });
  }, [navigateFromSvgTarget]);

  useEffect(() => {
    const svg = svgContainerRef.current?.querySelector('svg');
    if (!svg) {
      return;
    }

    svg.addEventListener('click', handleSvgClick);
    return () => {
      svg.removeEventListener('click', handleSvgClick);
    };
  }, [handleSvgClick]);

  // Click handler for mobile SVG - re-run when visibility changes
  useEffect(() => {
    if (!isMobileDiagramVisible) return;
    
    const svg = mobileSvgContainerRef.current?.querySelector('svg');
    if (!svg) {
      return;
    }

    // Handle both click and touch events for better mobile support
    const handleTouchEnd = (event: TouchEvent) => {
      // Use changedTouches for touchend event
      if (event.changedTouches.length !== 1) return;
      
      const touch = event.changedTouches[0];
      event.preventDefault();
      event.stopPropagation();
      
      navigateFromSvgTarget(event.target, svg, { x: touch.clientX, y: touch.clientY });
    };

    svg.addEventListener('click', handleSvgClick);
    svg.addEventListener('touchend', handleTouchEnd as EventListener);
    
    return () => {
      svg.removeEventListener('click', handleSvgClick);
      svg.removeEventListener('touchend', handleTouchEnd as EventListener);
    };
  }, [handleSvgClick, isMobileDiagramVisible, navigateFromSvgTarget]);

  // Make SVG responsive by removing fixed dimensions and setup hover effects
  useEffect(() => {
    const setupSvg = (container: HTMLDivElement | null, isMobile: boolean = false) => {
      const svg = container?.querySelector('svg');
      if (!svg) return;
      
      // Store original viewBox or create one from width/height
      const width = svg.getAttribute('width');
      const height = svg.getAttribute('height');
      if (!svg.getAttribute('viewBox') && width && height) {
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      }
      
      // Remove fixed dimensions so CSS can control size
      svg.removeAttribute('width');
      svg.removeAttribute('height');
      
      if (isMobile) {
        // For mobile: constrain by height, let width scale proportionally
        svg.style.maxHeight = '100%';
        svg.style.width = 'auto';
        svg.style.height = 'auto';
      } else {
        // For desktop: fill width, let height scale proportionally
        svg.style.width = '100%';
        svg.style.height = 'auto';
      }

      // Setup hover effects for diagram boxes using data attributes
      const diagramBoxes = svg.querySelectorAll('[data-label]');
      
      diagramBoxes.forEach((box) => {
        const bgRect = box.querySelector('[data-box-bg]') as SVGRectElement | null;
        const textElements = box.querySelectorAll('text, tspan');
        
        // Store original fill values
        const originalFills = new Map<Element, string | null>();
        const originalOpacities = new Map<Element, string | null>();
        textElements.forEach((el) => {
          originalFills.set(el, el.getAttribute('fill'));
          originalOpacities.set(el, el.getAttribute('fill-opacity'));
        });
        
        const handleMouseEnter = () => {
          if (bgRect) {
            bgRect.setAttribute('fill', 'black');
          }
          textElements.forEach((el) => {
            el.setAttribute('fill', 'white');
            el.setAttribute('fill-opacity', '1');
          });
        };
        
        const handleMouseLeave = () => {
          if (bgRect) {
            bgRect.setAttribute('fill', 'transparent');
          }
          textElements.forEach((el) => {
            const originalFill = originalFills.get(el);
            const originalOpacity = originalOpacities.get(el);
            if (originalFill !== null && originalFill !== undefined) {
              el.setAttribute('fill', originalFill);
            } else {
              el.removeAttribute('fill');
            }
            if (originalOpacity !== null && originalOpacity !== undefined) {
              el.setAttribute('fill-opacity', originalOpacity);
            } else {
              el.removeAttribute('fill-opacity');
            }
          });
        };
        
        box.addEventListener('mouseenter', handleMouseEnter);
        box.addEventListener('mouseleave', handleMouseLeave);
      });
    };

    setupSvg(svgContainerRef.current, false);
    setupSvg(mobileSvgContainerRef.current, true);
  }, []);

  return (
    <>
      {/* Desktop: 5-column CSS Grid layout */}
      {/* Columns: left-margin | text-content | gap | diagram | right-margin */}
      {/* At lg+ breakpoint (1024px+), diagram floats in right column */}
      {/* Below lg, falls back to mobile/tablet fixed-bottom layout */}
      <div className="hidden lg:grid mt-16 grid-cols-[minmax(16px,1fr)_minmax(300px,680px)_minmax(20px,0.5fr)_minmax(150px,540px)_minmax(16px,1fr)]">
        {/* Left margin - empty */}
        <div />
        
        {/* Text content column */}
        <div className="min-w-0">
          {introContent && (
            <div ref={introContentRef} className="mb-8 [&>h1:first-child]:text-[57px] [&>h1:first-child]:leading-[1.2] [&>h1:first-child]:text-balance [&>h1:first-child]:mt-0">
              {introContent}
            </div>
          )}
          <div id="model-architecture-essay">
            {modelDescriptionGDocPortionMarkdown && (
              <div ref={gDocContentRef} className="pt-6 pb-2 prose [&_h1:first-of-type]:font-et-book [&_h1:first-of-type]:text-[64px] [&_h1:first-of-type]:leading-[1.2] [&_h1:first-of-type]:text-balance [&_h1:first-of-type]:mt-0">
                <div className="w-full overflow-hidden">
                  {modelDescriptionGDocPortionMarkdown}
                </div>
              </div>
            )}
            <h2 className="font-et-book text-[50px] font-semibold leading-[1.2] text-balance mt-20">
              Model Architecture Details
            </h2>
            <p className="prose">
              Now we describe how our model works in more detail, including: written descriptions of how the model works, the equations forming the model, the reasoning behind our modeling choices, and how the most important parameters of the model are set. We&apos;ll start with how the coding time horizon progression works, then discuss how we model improvements in AI efficiency and how automation affects the model, then describe how we model intelligence explosions. You can use the model diagram or table of contents on the right to navigate. If you&apos;d instead like to see the forecasts produced by our model, go to the <Link href="/forecast">forecast page</Link>.
            </p>
            <p className="prose italic mt-4">
              The final section <TocLink label="Recap of the model" onClick={scrollToHeading} headingLookup={headingLookupRef} className="text-accent hover:text-accent/80">Recap of the Model</TocLink> is a succinct summary of the whole model, which some readers may prefer and which can serve as a useful reference. It contains little motivation or explanation, and mainly consists of the formulas.
            </p>
            <LazyExplanationsHtml
              ref={explanationContainerRef}
              className="prose scroll-p-20 overflow-hidden"
            />
          </div>
        </div>

        {/* Gap column - empty */}
        <div />
        
        {/* Sticky diagram column */}
        <div className="mt-8">
          <div 
            className="sticky top-[10vh] rounded-lg"
          >
            <div className="flex items-center justify-between pl-3 mb-2">
              <button
                type="button"
                onClick={() => setIsDiagramCollapsed(!isDiagramCollapsed)}
                className="text-[9px] uppercase tracking-widest font-light flex items-center gap-1.5 hover:text-neutral-600 transition-colors cursor-pointer"
              >
                Model Diagram
              </button>
              {!isDiagramCollapsed && (
                <button
                  type="button"
                  onClick={() => setIsDiagramModalOpen(true)}
                  className="text-neutral-400 hover:text-neutral-600 transition-colors"
                  title="View full screen"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
                  </svg>
                </button>
              )}
            </div>
            {!isDiagramCollapsed && (
              <div ref={svgContainerRef} className="pl-[2px] pt-0" onClick={handleContainerClick}>
                <ModelDiagram
                  id="model-architecture-diagram-svg"
                />
              </div>
            )}
            
            {/* Compact Table of Contents */}
            <nav className="px-3 pt-2 pb-4" aria-label="Table of contents">
              <button
                type="button"
                onClick={() => setIsContentsCollapsed(!isContentsCollapsed)}
                className="text-[9px] uppercase tracking-widest font-light mb-1  flex items-center gap-1.5 hover:text-neutral-600 transition-colors cursor-pointer w-full"
              >
                Contents
              </button>
              {!isContentsCollapsed && (
              <div className="grid grid-cols-3 gap-x-3 gap-y-0.5 text-[10px] leading-tight">
                {/* Introduction */}
                <div className="col-span-3 mt-1">
                  <TocLink
                    label="Introducing the AI Futures Model"
                    onClick={scrollToHeading}
                    headingLookup={headingLookupRef}
                    className={`text-left font-semibold transition-colors hover:text-[#2A623D] ${
                      activeSectionKey?.includes('introducingtheaifuturesmodel') 
                        ? 'text-[#2A623D]' 
                        : 'text-neutral-700'
                    }`}
                  >
                    Introduction
                  </TocLink>
                </div>
                <TocLink
                  label="Stage 1: Automating coding"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left pl-0 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('stage1')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  Stage 1: Automating Coding
                </TocLink>
                <TocLink
                  label="Stage 2: Automating research taste"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left pl-0 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('stage2')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  Stage 2: Research Taste
                </TocLink>
                <TocLink
                  label="Stage 3: The intelligence explosion"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left pl-0 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('stage3')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  Stage 3: Intelligence Explosion
                </TocLink>
                
                {/* Section 1 */}
                <div className="col-span-3 mt-2">
                  <TocLink
                    label="Time horizon and the Automated Coder milestone"
                    onClick={scrollToHeading}
                    headingLookup={headingLookupRef}
                    className={`text-left font-semibold transition-colors hover:text-[#2A623D] ${
                      activeSectionKey?.includes('timehorizonandtheautomatedcodermilestone') 
                        ? 'text-[#2A623D]' 
                        : 'text-neutral-700'
                    }`}
                  >
                    1. Time Horizon & AC Milestone
                  </TocLink>
                </div>
                <TocLink
                  label="Replacing calendar time with effective compute"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left pl-0 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('replacingcalendartimewitheffectivecompute')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  1.1 Using Effective Compute
                </TocLink>
                <TocLink
                  label="Time horizon progression"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left pl-0 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('timehorizonprogression')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  1.2 Horizon Progession
                </TocLink>
                <TocLink
                  label="Setting the effective compute corresponding to the Automated Coder"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left col-span-1 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('settingtheeffectivecomputecorrespondingtotheautomatedcoder')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  1.3 AC Effective Compute
                </TocLink>
                
                {/* Section 2 */}
                <div className="col-span-3 mt-2">
                  <TocLink
                    label="Modeling effective compute"
                    onClick={scrollToHeading}
                    headingLookup={headingLookupRef}
                    className={`text-left font-semibold transition-colors hover:text-[#2A623D] ${
                      activeSectionKey?.includes('modelingeffectivecompute') && !activeSectionKey?.includes('computeforecasts')
                        ? 'text-[#2A623D]' 
                        : 'text-neutral-700'
                    }`}
                  >
                    2. Modeling Effective Compute
                  </TocLink>
                </div>
                <TocLink
                  label="Compute forecasts"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left pl-0 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('computeforecasts')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  2.1 Compute Forecasts
                </TocLink>
                <TocLink
                  label="Experiment throughput"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left pl-0 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('experimentthroughput')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  2.2 Experiment Throughput
                </TocLink>
                <TocLink
                  label="Research effort and AI R&D uplift"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left pl-0 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('researcheffortandairduplift')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  2.3 Research Effort
                </TocLink>
                <TocLink
                  label="Software efficiency"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left col-span-1 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('softwareefficiency')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  2.4 Software Efficiency
                </TocLink>
                
                {/* Section 3 */}
                <div className="col-span-3 mt-2">
                  <TocLink
                    label="Coding automation"
                    onClick={scrollToHeading}
                    headingLookup={headingLookupRef}
                    className={`text-left font-semibold transition-colors hover:text-[#2A623D] ${
                      activeSectionKey?.includes('codingautomation') && !activeSectionKey?.includes('fraction') && !activeSectionKey?.includes('efficiency') && !activeSectionKey?.includes('aggregate')
                        ? 'text-[#2A623D]' 
                        : 'text-neutral-700'
                    }`}
                  >
                    3. Coding Automation
                  </TocLink>
                </div>
                <TocLink
                  label="Coding automation fraction"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left pl-0 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('codingautomationfraction')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  3.1 Automation Fraction
                </TocLink>
                <TocLink
                  label="Coding automation efficiency"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left pl-0 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('codingautomationefficiency')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  3.2 Automation Efficiency
                </TocLink>
                <TocLink
                  label="Aggregate coding labor"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left col-span-1 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('aggregatecodinglabor')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  3.3 Aggregate Coding Labor
                </TocLink>
                
                {/* Section 4 */}
                <div className="col-span-3 mt-2">
                  <TocLink
                    label="Research taste automation"
                    onClick={scrollToHeading}
                    headingLookup={headingLookupRef}
                    className={`text-left font-semibold transition-colors hover:text-[#2A623D] ${
                      activeSectionKey?.includes('researchtasteautomation')
                        ? 'text-[#2A623D]' 
                        : 'text-neutral-700'
                    }`}
                  >
                    4. Research Taste Automation
                  </TocLink>
                </div>
                <TocLink
                  label="Human research taste"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left pl-0 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('humanresearchtaste')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  4.1 Human Taste
                </TocLink>
                <TocLink
                  label="Automated research taste"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left pl-0 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('automatedresearchtaste')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  4.2 Automated Taste
                </TocLink>
                <TocLink
                  label="Aggregate research taste"
                  onClick={scrollToHeading}
                  headingLookup={headingLookupRef}
                  className={`text-left col-span-1 transition-colors hover:text-[#2A623D] ${
                    activeSectionKey?.includes('aggregateresearchtaste')
                      ? 'text-[#2A623D] font-medium bg-[#2A623D]/5  px-1 rounded'
                      : 'text-neutral-500'
                  }`}
                >
                  4.3 Aggregate Taste
                </TocLink>
                
                {/* Section 5 */}
                <div className="col-span-3 mt-2">
                  <TocLink
                    label="How does our model behave after full AI R&D automation"
                    onClick={scrollToHeading}
                    headingLookup={headingLookupRef}
                    className={`text-left font-semibold transition-colors hover:text-[#2A623D] ${
                      activeSectionKey?.includes('howdoesourmodelbehaveafterfullairda')
                        ? 'text-[#2A623D]'
                        : 'text-neutral-700'
                    }`}
                  >
                    5. After Full AI R&D Automation
                  </TocLink>
                </div>

                {/* Section 6 */}
                <div className="col-span-3 mt-2">
                  <TocLink
                    label="Recap of the model"
                    onClick={scrollToHeading}
                    headingLookup={headingLookupRef}
                    className={`text-left font-semibold transition-colors hover:text-[#2A623D] ${
                      activeSectionKey?.includes('recapofthemodel')
                        ? 'text-[#2A623D]'
                        : 'text-neutral-700'
                    }`}
                  >
                    6. Recap of the Model
                  </TocLink>
                </div>
              </div>
              )}
            </nav>
          </div>
        </div>
        
        {/* Right margin - empty */}
        <div />
      </div>

      {/* Mobile/Tablet: Content with fixed bottom diagram */}
      <div className="lg:hidden mt-12">
        {/* Text content - padding matches header (pl-6) */}
        {/* Bottom padding uses calc to account for 20vh drawer + some buffer when visible */}
        <div 
          ref={mobileContentRef}
          className={`pl-6 pr-6 ${isMobileDiagramVisible ? 'pb-[calc(20vh+40px)]' : 'pb-10'}`}
        >
          {introContent && (
            <div className="mb-8 [&>h1:first-child]:text-[57px] [&>h1:first-child]:leading-[1.2] [&>h1:first-child]:text-balance [&>h1:first-child]:mt-0">
              {introContent}
            </div>
          )}
          <div id="model-architecture-essay-mobile">
            {modelDescriptionGDocPortionMarkdown && (
              <div className="pt-6 pb-2 prose [&_h1:first-of-type]:text-[57px] [&_h1:first-of-type]:leading-[1.2] [&_h1:first-of-type]:text-balance [&_h1:first-of-type]:mt-0">
                <div className="w-full overflow-hidden">
                  {modelDescriptionGDocPortionMarkdown}
                </div>
              </div>
            )}
            <h2 className="font-et-book text-[50px] font-semibold leading-[1.2] text-balance mt-20">
              Model Architecture Details
            </h2>
            <p className="prose"> Now we will describe how our model works in more detail, including: written descriptions of how the model works, the equations forming the model, the reasoning behind our modeling choices, and how the most important parameters of the model are set. We&apos;ll start with how the coding time horizon progression works, then discuss how we model improvements in AI efficiency and how automation affects the model, then describe how we model intelligence explosions. If you&apos;d instead like to see the forecasts produced by our model, go to the forecast page.            </p>
            <p className="prose italic mt-4">
              The final section <TocLink label="Recap of the model" onClick={scrollToHeading} headingLookup={headingLookupRef} className="text-accent hover:text-accent/80">Recap of the Model</TocLink> is a succinct summary of the whole model, which some readers may prefer and which can serve as a useful reference. It contains little motivation or explanation, and mainly consists of the formulas.
            </p>
            <LazyExplanationsHtml
              className="prose scroll-p-20 overflow-hidden"
            />
          </div>
        </div>
        
        {/* Fixed bottom diagram - clamped to 20% of screen height, fades in after scrolling */}
        <div 
          className={`fixed bottom-0 left-0 right-0 bg-[#fffff8]/80 backdrop-blur-md z-40 transition-all duration-500 ease-out ${
            isMobileDiagramVisible 
              ? 'opacity-100 translate-y-0' 
              : 'opacity-0 translate-y-8 pointer-events-none'
          }`}
        >
          <div 
            ref={mobileSvgContainerRef} 
            className="p-2 h-[20vh] flex items-center justify-center overflow-hidden [&>svg]:max-h-full cursor-pointer"
            onClick={handleContainerClick}
          >
            <ModelDiagram
              id="model-architecture-diagram-svg-mobile"
            />
          </div>
        </div>
      </div>

      {/* Full-screen diagram modal */}
      {isDiagramModalOpen && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={() => setIsDiagramModalOpen(false)}
        >
          <div 
            className="relative bg-[#fffff8] rounded-lg shadow-2xl max-w-[90vw] max-h-[90vh] p-8 overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              onClick={() => setIsDiagramModalOpen(false)}
              className="absolute top-4 right-4 text-neutral-400 hover:text-neutral-600 transition-colors p-2"
              title="Close"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
            <div className="text-[11px] uppercase tracking-widest text-neutral-400 font-semibold mb-4">
              Model Diagram
            </div>
            <div className="w-[80vw] max-w-[1200px]">
              <ModelDiagram
                id="model-architecture-diagram-svg-modal"
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
