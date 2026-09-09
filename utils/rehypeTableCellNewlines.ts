import { visit } from 'unist-util-visit';
import type { Root, Element, Content, RootContent } from 'hast';

/**
 * Check if an element is empty or contains only whitespace
 */
function isEmptyElement(node: Content): boolean {
  if (node.type === 'text') {
    return (node.value || '').trim().length === 0;
  }
  if (node.type === 'element') {
    const children = node.children || [];
    if (children.length === 0) return true;
    return children.every(isEmptyElement);
  }
  return false;
}

/**
 * Recursively clean empty spans and whitespace-only text nodes from content
 */
function cleanContent(children: Content[]): Content[] {
  return children
    .filter(child => !isEmptyElement(child))
    .map(child => {
      if (child.type === 'element' && child.children) {
        return {
          ...child,
          children: cleanContent(child.children) as Element['children']
        };
      }
      return child;
    });
}

/**
 * Converts <p> tags within table cells to <br> tags to preserve line breaks
 * and removes empty spans/whitespace that Google Docs adds
 */
export default function rehypeTableCellNewlines() {
  return (tree: Root) => {
    visit(tree, 'element', (node: Element) => {
      if (node.tagName !== 'td' && node.tagName !== 'th') {
        return;
      }

      if (!Array.isArray(node.children) || node.children.length === 0) {
        return;
      }

      const newChildren: Content[] = [];
      let hasAddedContent = false;
      
      for (let i = 0; i < node.children.length; i++) {
        const child = node.children[i];
        
        // If it's a <p> tag, unwrap it and add its children with <br> separators
        if (child.type === 'element' && child.tagName === 'p') {
          const pElement = child;
          const pChildren = cleanContent(pElement.children || []);
          
          // Check if this paragraph has actual content after cleaning
          if (pChildren.length === 0) {
            continue;
          }
          
          // Add line break before this paragraph (except for the very first paragraph with content)
          if (hasAddedContent) {
            const brElement = {
              type: 'element',
              tagName: 'br',
              properties: {},
              children: []
            } satisfies RootContent;

            // Push two to achieve proper look; can come back to this and switch it to one if we fix it with styling instead.
            newChildren.push(brElement, brElement);
          }
          
          newChildren.push(...pChildren);
          hasAddedContent = true;
        } else if (!isEmptyElement(child)) {
          // Not a <p> tag and not empty, keep it
          newChildren.push(child);
          hasAddedContent = true;
        }
      }
      
      node.children = newChildren as Element['children'];
    });
  };
}

