import { visit } from 'unist-util-visit';
import type { Root, Element, ElementContent } from 'hast';
import type { Parent } from 'unist';

// Note: Comments (#cmnt) are now stripped at the raw HTML level in googleDocToMarkdown.ts
// This file only handles footnotes (#ftnt)

function hasFtntAnchor(node: Element): boolean {
  const stack: ElementContent[] = [node];
  while (stack.length) {
    const cur = stack.pop() as ElementContent;
    if (cur.type === 'element' && cur.tagName === 'a') {
      const href = cur.properties?.href;
      if (typeof href === 'string' && href.startsWith('#ftnt')) return true;
    }
    const kids = (cur.type === 'element' ? cur.children : []) as ElementContent[];
    for (const k of kids) stack.push(k);
  }
  return false;
}

function containsFtntIdOrName(node: Element): boolean {
  const stack: ElementContent[] = [node];
  while (stack.length) {
    const cur = stack.pop() as ElementContent;
    if (cur.type === 'element') {
      const id = cur.properties?.id;
      const name = cur.properties?.name;
      const href = cur.properties?.href;
      if ((typeof id === 'string' && /^ftnt/.test(id)) || (typeof name === 'string' && /^ftnt/.test(name))) return true;
      if (typeof href === 'string' && /^#ftnt/.test(href)) return true;
    }
    const kids = (cur.type === 'element' ? cur.children : []) as ElementContent[];
    for (const k of kids) stack.push(k);
  }
  return false;
}

export default function rehypeStripFootnotes() {
  return (tree: Root) => {
    // 0) Convert full URLs with anchors to just anchor links (same-page navigation)
    const sitePatterns = [
      'https://aifuturesmodel.com/',
      'https://www.aifuturesmodel.com/',
      'http://aifuturesmodel.com/',
      'http://www.aifuturesmodel.com/',
    ];
    visit(tree, 'element', (node: Element) => {
      if (node.tagName === 'a' && node.properties?.href) {
        const href = node.properties.href as string;
        const matchedPattern = sitePatterns.find(pattern => href.startsWith(pattern + '#'));
        if (matchedPattern) {
          node.properties.href = href.replace(matchedPattern, '');
        }
      }
    });

    // 1) Remove inline footnote references
    visit(tree, 'element', (node: Element, index: number | undefined, parent: Parent | undefined) => {
      if (!parent || typeof index !== 'number') return;

      // Remove <sup> that wraps a footnote link
      if (node.tagName === 'sup' && hasFtntAnchor(node)) {
        parent.children.splice(index, 1);
        return;
      }

      // Remove bare <a href="#ftntN">
      if (node.tagName === 'a') {
        const href = (node.properties?.href as string) || '';
        if (href.startsWith('#ftnt')) {
          parent.children.splice(index, 1);
          return;
        }
      }
    });

    // 2) Remove footnote body blocks
    const blockTags = new Set(['p', 'li', 'div', 'section', 'ol', 'ul']);
    visit(tree, 'element', (node: Element, index: number | undefined, parent: Parent | undefined) => {
      if (!parent || typeof index !== 'number') return;
      if (!blockTags.has(node.tagName)) return;
      if (containsFtntIdOrName(node)) {
        parent.children.splice(index, 1);
        return;
      }
    });
  };
}
