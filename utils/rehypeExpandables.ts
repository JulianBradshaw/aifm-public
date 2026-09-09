import { visit, SKIP } from 'unist-util-visit';
import type { Root, Element, ElementContent, Text } from 'hast';
import type { Parent } from 'unist';

// Use ElementContent for element children (Element, Text, Comment, Raw)
type Content = ElementContent;

/**
 * Slugify text for use as an ID
 */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

/**
 * Get text content from a node recursively
 */
function getTextContent(node: Content | Element | Root): string {
  if (node.type === 'text') {
    return (node as Text).value || '';
  }
  if ('children' in node && Array.isArray(node.children)) {
    return node.children.map(child => getTextContent(child as Content)).join('');
  }
  return '';
}

/**
 * Check if a node contains the "Expandable:" marker
 */
function containsExpandableMarker(node: Content): boolean {
  const text = getTextContent(node);
  return text.includes('Expandable:');
}

/**
 * Check if a node contains the "End of expandable" marker
 */
function containsEndMarker(node: Content): boolean {
  const text = getTextContent(node);
  return text.includes('End of expandable');
}

/**
 * Extract summary text after "Expandable:" marker (plain text for slugify)
 */
function extractSummaryText(node: Content): string {
  const text = getTextContent(node);
  const markerIndex = text.indexOf('Expandable:');
  if (markerIndex === -1) return '';
  return text.slice(markerIndex + 'Expandable:'.length).trim();
}

/**
 * Extract summary children after "Expandable:" marker, preserving formatting (bold, italic, etc.)
 * Returns the child nodes that come after the "Expandable:" text.
 */
function extractSummaryChildren(node: Content): Content[] {
  if (node.type !== 'element') {
    // For text nodes, just return the text after Expandable:
    const text = getTextContent(node);
    const markerIndex = text.indexOf('Expandable:');
    if (markerIndex === -1) return [];
    const summaryText = text.slice(markerIndex + 'Expandable:'.length).trim();
    if (!summaryText) return [];
    return [{ type: 'text', value: summaryText } as Text];
  }

  const el = node as Element;
  const result: Content[] = [];
  let foundMarker = false;
  let markerInChild = false;

  for (const child of el.children) {
    const childText = getTextContent(child as Content);

    if (!foundMarker) {
      if (childText.includes('Expandable:')) {
        foundMarker = true;
        // Check if the marker is in THIS child or a descendant
        if (child.type === 'text') {
          // Text node contains the marker - extract what's after it
          const textNode = child as Text;
          const markerIndex = (textNode.value || '').indexOf('Expandable:');
          const afterMarker = (textNode.value || '').slice(markerIndex + 'Expandable:'.length).trim();
          if (afterMarker) {
            result.push({ type: 'text', value: afterMarker } as Text);
          }
        } else if (child.type === 'element') {
          // The marker might be inside this element - recurse
          const innerChildren = extractSummaryChildren(child as Content);
          if (innerChildren.length > 0) {
            markerInChild = true;
            result.push(...innerChildren);
          }
        }
      }
    } else {
      // After finding the marker, include all subsequent siblings
      result.push(cloneNode(child as Content));
    }
  }

  // If marker was found inside a child element, we've already extracted what we need
  // If marker was in a text node, we need the siblings that come after
  return result;
}

/**
 * Remove the expandable marker from a node, returning a cleaned clone or null if empty
 */
function removeExpandableMarker(node: Content): Content | null {
  if (node.type === 'text') {
    const textNode = node as Text;
    const text = textNode.value || '';
    const markerIndex = text.indexOf('Expandable:');
    if (markerIndex === -1) return node;

    // Remove everything from "Expandable:" to the end of this text node
    const newText = text.slice(0, markerIndex).trim();
    if (!newText) return null;
    return { ...textNode, value: newText };
  }

  if (node.type === 'element') {
    const el = node as Element;
    const nodeText = getTextContent(node);

    // If this element contains the marker, we need to handle it specially
    if (nodeText.includes('Expandable:')) {
      // If the entire element is just the expandable marker, remove it
      if (nodeText.trim().startsWith('Expandable:')) {
        return null;
      }

      // Otherwise, try to clean children
      const newChildren = el.children
        .map(child => removeExpandableMarker(child as Content))
        .filter((child): child is Content => child !== null);

      if (newChildren.length === 0) return null;
      return { ...el, children: newChildren };
    }

    return node;
  }

  return node;
}

/**
 * Remove the end marker from a node, returning a cleaned clone or null if empty
 */
function removeEndMarker(node: Content): Content | null {
  if (node.type === 'text') {
    const textNode = node as Text;
    const text = textNode.value || '';
    const markerIndex = text.indexOf('End of expandable');
    if (markerIndex === -1) return node;

    // Remove "End of expandable" and everything after
    const newText = text.slice(0, markerIndex).trim();
    if (!newText) return null;
    return { ...textNode, value: newText };
  }

  if (node.type === 'element') {
    const el = node as Element;
    const nodeText = getTextContent(node);

    if (nodeText.includes('End of expandable')) {
      // If the entire element is just the end marker, remove it
      if (nodeText.trim().startsWith('End of expandable')) {
        return null;
      }

      // Otherwise, clean children
      const newChildren = el.children
        .map(child => removeEndMarker(child as Content))
        .filter((child): child is Content => child !== null);

      if (newChildren.length === 0) return null;
      return { ...el, children: newChildren };
    }

    return node;
  }

  return node;
}

/**
 * Check if a node has meaningful content (text or important elements)
 */
function hasMeaningfulContent(node: Content): boolean {
  const text = getTextContent(node).trim();
  if (text.length > 0) return true;

  if (node.type === 'element') {
    const el = node as Element;
    const meaningfulTags = ['img', 'figure', 'table', 'details', 'ol', 'ul', 'li', 'code', 'pre', 'blockquote'];
    if (meaningfulTags.includes(el.tagName)) return true;

    // Recursively check children
    if (el.children) {
      return el.children.some(child => hasMeaningfulContent(child as Content));
    }
  }

  return false;
}

/**
 * Deep clone a node
 */
function cloneNode(node: Content): Content {
  if (node.type === 'text') {
    return { ...(node as Text) };
  }
  if (node.type === 'element') {
    const el = node as Element;
    return {
      ...el,
      properties: { ...el.properties },
      children: el.children ? el.children.map(child => cloneNode(child as Content)) : []
    };
  }
  return { ...node };
}

/**
 * Rehype plugin to transform Expandable: ... End of expandable markers
 * into <details><summary>...</summary>...</details> elements
 */
export default function rehypeExpandables() {
  return (tree: Root) => {
    // We need to handle nested expandables, so we process from innermost to outermost
    // by making multiple passes until no more expandables are found
    let hasChanges = true;
    let iterations = 0;
    const maxIterations = 100; // Safety limit

    while (hasChanges && iterations < maxIterations) {
      hasChanges = false;
      iterations++;

      visit(tree, 'element', (node: Element, index: number | undefined, parent: Parent | undefined) => {
        if (!parent || typeof index !== 'number') return;

        const children = parent.children as Content[];

        // Look for a child that contains "Expandable:"
        let startIndex = -1;
        let startNode: Content | null = null;

        for (let i = 0; i < children.length; i++) {
          const child = children[i];
          if (child.type === 'element' || child.type === 'text') {
            if (containsExpandableMarker(child)) {
              startIndex = i;
              startNode = child;
              break;
            }
          }
        }

        if (startIndex === -1 || !startNode) return;

        // Find the matching "End of expandable"
        let endIndex = -1;
        let depth = 0;

        for (let i = startIndex; i < children.length; i++) {
          const child = children[i];
          if (child.type === 'element' || child.type === 'text') {
            const text = getTextContent(child);

            // Count nested expandables
            if (text.includes('Expandable:')) {
              if (i !== startIndex) {
                depth++;
              }
            }

            if (text.includes('End of expandable')) {
              if (depth === 0) {
                endIndex = i;
                break;
              }
              depth--;
            }
          }
        }

        if (endIndex === -1) return;

        // Extract summary
        const summaryText = extractSummaryText(startNode);
        if (!summaryText) return;
        const summaryChildren = extractSummaryChildren(startNode);

        // Build the details content
        const detailsChildren: Content[] = [];

        // Clone and clean the start node (remove the Expandable: marker)
        const cleanedStart = removeExpandableMarker(cloneNode(startNode));
        if (cleanedStart && hasMeaningfulContent(cleanedStart)) {
          detailsChildren.push(cleanedStart);
        }

        // Add all nodes between start and end
        for (let i = startIndex + 1; i < endIndex; i++) {
          const cloned = cloneNode(children[i]);
          if (hasMeaningfulContent(cloned)) {
            detailsChildren.push(cloned);
          }
        }

        // Clone and clean the end node (remove the End of expandable marker)
        if (endIndex !== startIndex) {
          const cleanedEnd = removeEndMarker(cloneNode(children[endIndex]));
          if (cleanedEnd && hasMeaningfulContent(cleanedEnd)) {
            detailsChildren.push(cleanedEnd);
          }
        }

        // Create the summary element
        const summaryId = slugify(summaryText);
        const summaryEl: Element = {
          type: 'element',
          tagName: 'summary',
          properties: summaryId ? { id: summaryId } : {},
          children: summaryChildren.length > 0
            ? summaryChildren
            : [{ type: 'text', value: summaryText } as Text]
        };

        // Create the details element
        const detailsEl: Element = {
          type: 'element',
          tagName: 'details',
          properties: {},
          children: [summaryEl, ...detailsChildren]
        };

        // Replace the range [startIndex, endIndex] with the details element
        children.splice(startIndex, endIndex - startIndex + 1, detailsEl);

        hasChanges = true;
        return SKIP;
      });
    }
  };
}
