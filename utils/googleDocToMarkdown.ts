import { unified } from 'unified';
import rehypeParse from 'rehype-parse';
import rehypeRemark from 'rehype-remark';
import remarkGfm from 'remark-gfm';
import remarkStringify, { Options as RemarkStringifyOptions } from 'remark-stringify';
import juice from 'juice';
import rehypeGoogleStylesToSemantic from '@/utils/rehypeGoogleStylesToSemantic';
import rehypeStripFootnotes from '@/utils/rehypeStripFootnotes';
import rehypeTableCellNewlines from '@/utils/rehypeTableCellNewlines';
import rehypeExpandables from '@/utils/rehypeExpandables';
import { uploadGoogleDocImagesToCloudinary } from './cloudinary';
import type { Element as HastElement, Content as HastContent, Root as HastRoot, Text as HastText, Properties } from 'hast';
import type { Handle, State } from 'hast-util-to-mdast';

const serializeHeadingWithId: Handle = (_state, element) => element.properties?.id ? { type: 'html', value: serializeMinimalHtml(element) } : undefined;

/**
 * Recursively extract all text content from an element
 */
function getTextContent(element: HastElement): string {
  let text = '';
  if (!element.children) return text;

  for (const child of element.children) {
    if (child.type === 'text') {
      text += child.value || '';
    } else if (child.type === 'element') {
      text += getTextContent(child as HastElement);
    }
  }
  return text;
}

/**
 * Check if an element contains any non-text elements (images, etc.)
 */
function hasNonTextElements(element: HastElement): boolean {
  if (!element.children) return false;

  for (const child of element.children) {
    if (child.type === 'element') {
      const el = child as HastElement;
      // Images are definitely meaningful
      if (el.tagName === 'img') {
        return true;
      }
      // Recursively check children
      if (hasNonTextElements(el)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Check if a paragraph element is truly empty (Google Docs empty lines are <p><span></span></p>)
 * Returns true only for paragraphs that contain no text and no meaningful elements like images
 */
function isEmptyParagraph(element: HastElement): boolean {
  // If it has images or other non-text elements, it's not empty
  if (hasNonTextElements(element)) {
    return false;
  }

  // Check if there's any actual text content
  const textContent = getTextContent(element).trim();
  return textContent.length === 0;
}

/**
 * Rehype plugin that converts consecutive empty paragraphs to <br> tags,
 * but skips the first empty paragraph in each sequence (since markdown
 * already adds spacing between paragraphs).
 * E.g., 1 empty paragraph = no <br>, 2 empty paragraphs = 1 <br>, etc.
 */
function rehypeCollapseEmptyParagraphs() {
  return (tree: HastRoot) => {
    if (!tree.children) return;

    const newChildren: HastContent[] = [];
    let consecutiveEmptyCount = 0;

    for (const child of tree.children) {
      const isEmptyP = child.type === 'element' &&
        (child as HastElement).tagName === 'p' &&
        isEmptyParagraph(child as HastElement);

      if (isEmptyP) {
        consecutiveEmptyCount++;
        // Skip the first empty paragraph, convert subsequent ones to <br>
        if (consecutiveEmptyCount > 1) {
          newChildren.push({
            type: 'element',
            tagName: 'br',
            properties: {},
            children: []
          } as HastElement);
        }
      } else {
        consecutiveEmptyCount = 0;
        newChildren.push(child as HastContent);
      }
    }

    tree.children = newChildren as HastRoot['children'];
  };
}

const handlers: Record<string, Handle> = {
  // Preserve minimal HTML for sup/sub since remark-math does not cover them
  sup: (_state, element) => ({ type: 'html', value: serializeMinimalHtml(element) }),
  sub: (_state, element) => ({ type: 'html', value: serializeMinimalHtml(element) }),
  // Preserve <br> tags as HTML (especially important for table cells)
  br: () => ({ type: 'html', value: '<br>' }),
  // Preserve details/summary for expandables
  details: (_state, element) => ({ type: 'html', value: serializeDetailsHtml(element) }),
  // For paragraphs, explicitly handle them using state.all() because
  // returning undefined doesn't correctly fall back to default handling for all elements.
  // Empty paragraphs are handled by rehypeCollapseEmptyParagraphs below.
  p: (state, element) => {
    const children = state.all(element);
    return { type: 'paragraph', children } as ReturnType<Handle>;
  },
  // Preserve headings as HTML if they have an id attribute to enable internal links
  h1: serializeHeadingWithId,
  h2: serializeHeadingWithId,
  h3: serializeHeadingWithId,
  h4: serializeHeadingWithId,
  h5: serializeHeadingWithId,
  h6: serializeHeadingWithId,
};

export async function googleDocToMarkdown(docId: string, options?: { noCache?: boolean }): Promise<string> {
  // Cache for 1 hour (3600 seconds) to avoid frequent revalidation
  // The 'use cache' directive at the component level will also help
  const fetchOptions = options?.noCache
    ? { cache: 'no-store' as const }
    : { cache: 'force-cache' as const, next: { revalidate: 3600 } };
  const res = await fetch(`https://docs.google.com/document/d/${docId}/export?format=htm&tab=t.0`, fetchOptions);
  if (!res.ok) {
    throw new Error(`Failed to fetch Google Doc: ${res.status}`);
  }

  let html = await res.text();

  // Strip Google Doc comments from raw HTML before processing
  // (a) Remove inline comment references: <sup><a href="#cmnt...">...</a></sup>
  html = html.replace(/<sup><a href="#cmnt[^"]*"[^>]*>\[[a-z]\]<\/a><\/sup>/gi, '');
  // (b) Remove comment body divs at the bottom: <div><p><a href="#cmnt_ref...">...</a>...comment text...</p></div>
  // Only match divs that don't contain images (to avoid accidentally deleting images)
  html = html.replace(/<div[^>]*><p[^>]*><a href="#cmnt_ref\d+"[^>]*>\[[a-z]\]<\/a>(?:(?!<img)[^<]|<(?!img)[^>]*>)*<\/p><\/div>/gi, '');

  const htmlWithCloudinaryImages = await uploadGoogleDocImagesToCloudinary(html);

  const inlined = juice(htmlWithCloudinaryImages);
  const stringifyOptions = { fences: true, bullet: '-' } as unknown as RemarkStringifyOptions & { allowDangerousHtml?: boolean };
  stringifyOptions.allowDangerousHtml = true;

  const file = await unified()
    .use(rehypeParse, { fragment: true })
    .use(rehypeGoogleStylesToSemantic)
    .use(rehypeStripFootnotes)
    .use(rehypeTableCellNewlines)
    .use(rehypeExpandables)
    .use(rehypeCollapseEmptyParagraphs)
    .use(rehypeRemark, { handlers })
    .use(remarkGfm)
    .use(remarkStringify, stringifyOptions)
    .process(inlined);
  
  let markdown = String(file);
  
  // Fix: remarkStringify escapes underscores, but inside $$...$$ they should be literal
  // This is a brittle post-process to unescape them
  markdown = markdown.replace(/(\$\$[\s\S]+?\$\$)/g, (match) => match.replace(/\\_/g, '_'));
  markdown = markdown.replace(/(\$[^\$\n]+?\$)/g, (match) => match.replace(/\\_/g, '_'));

  // Fix: Empty spans in the HTML (from stripped comments) create **** artifacts
  markdown = markdown.replace(/\*\*\*\*/g, '');

  return markdown;
}


function isText(node: HastContent | HastRoot | null | undefined): node is HastText {
  return !!node && (node as HastText).type === 'text';
}

function isElement(node: HastContent | HastRoot | null | undefined): node is HastElement {
  return !!node && (node as HastElement).type === 'element';
}

function serializeMinimalHtml(node: HastContent | HastRoot | null | undefined): string {
  if (!node) return '';
  if (isText(node)) {
    return String(node.value || '');
  }
  if (isElement(node)) {
    const tag = (node.tagName as string) || 'span';
    const id = node.properties?.id;
    const idAttr = id ? ` id="${String(id)}"` : '';
    const children = Array.isArray(node.children) ? node.children : [];
    const inner = children.map(serializeMinimalHtml).join('');
    return `<${tag}${idAttr}>${inner}</${tag}>`;
  }
  const children = ((node as unknown as { children?: HastContent[] }).children) || [];
  return children.map(serializeMinimalHtml).join('');
}

/**
 * Serialize details/summary elements to HTML, preserving the expandable structure.
 * This recursively serializes all children, wrapping content in appropriate tags.
 */
function serializeDetailsHtml(node: HastContent | HastRoot | null | undefined): string {
  if (!node) return '';
  if (isText(node)) {
    return String(node.value || '');
  }
  if (isElement(node)) {
    const tag = (node.tagName as string) || 'span';
    const props = node.properties || {};
    const attrs: string[] = [];

    // Preserve id attribute
    if (props.id) {
      attrs.push(`id="${String(props.id)}"`);
    }

    // Preserve href attribute for anchor tags
    if (tag === 'a' && props.href) {
      attrs.push(`href="${String(props.href)}"`);
    }

    const attrStr = attrs.length > 0 ? ' ' + attrs.join(' ') : '';
    const children = Array.isArray(node.children) ? node.children : [];
    const inner = children.map(serializeDetailsHtml).join('');
    return `<${tag}${attrStr}>${inner}</${tag}>`;
  }
  const children = ((node as unknown as { children?: HastContent[] }).children) || [];
  return children.map(serializeDetailsHtml).join('');
}
