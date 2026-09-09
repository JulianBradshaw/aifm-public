import { visit } from 'unist-util-visit';
import type { Root, Element, Content, Text } from 'hast';
import type { Parent } from 'unist';

function hasBold(style: string | undefined): boolean {
  if (!style) return false;
  const s = style.toLowerCase();
  return s.includes('font-weight:bold') || /font-weight:\s*(6\d\d|7\d\d|8\d\d|9\d\d)\b/.test(s);
}

function hasItalic(style: string | undefined): boolean {
  if (!style) return false;
  const s = style.toLowerCase();
  return s.includes('font-style:italic');
}

function hasSup(style: string | undefined): boolean {
  if (!style) return false;
  const s = style.toLowerCase();
  return /vertical-align\s*:\s*super\b/.test(s) || /baseline-shift\s*:\s*super\b/.test(s);
}

function hasSub(style: string | undefined): boolean {
  if (!style) return false;
  const s = style.toLowerCase();
  return /vertical-align\s*:\s*sub\b/.test(s) || /baseline-shift\s*:\s*sub\b/.test(s);
}

function parseFontSize(style: string | undefined): number | undefined {
  if (!style) return undefined;
  const m = /font-size\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(pt|px|em|rem|%)?/i.exec(style);
  if (!m) return undefined;
  const value = parseFloat(m[1]);
  const unit = (m[2] || 'pt').toLowerCase();
  // Normalize to px for rough comparison: 1pt ≈ 1.333px, 1em≈16px default
  if (unit === 'pt') return value * 1.333;
  if (unit === 'px') return value;
  if (unit === 'em' || unit === 'rem') return value * 16;
  if (unit === '%') return (value / 100) * 16;
  return value;
}

function normalizeTagName(name: string): string {
  if (name === 'b') return 'strong';
  if (name === 'i') return 'em';
  return name;
}

export default function rehypeGoogleStylesToSemantic() {
  return (tree: Root) => {
    visit(tree as unknown as Parent, 'element', (node: Element, index: number | undefined, parent: Parent | undefined) => {
      if (!parent || typeof index !== 'number' || !parent.children) return;

      // Normalize <b>/<i>
      node.tagName = normalizeTagName(node.tagName);

      // Heuristic: Some Google Docs encode variables like "C̅train" as a letter with a combining overline (U+0305)
      // followed by plain letters; wrap the immediate next sibling letters in <sub>...
      const OVERLINE = '\u0305';
      const isTextNode = (c: Content): c is Text => c.type === 'text';
      const nodeHasOverline = Array.isArray(node.children) && (node.children as Content[]).some((c) => isTextNode(c) && typeof (c as Text).value === 'string' && (c as Text).value!.includes(OVERLINE));
      if (nodeHasOverline && parent && Array.isArray((parent as unknown as Parent).children)) {
        const parentWithChildren = parent as unknown as Parent & { children: Content[] };
        const next = parentWithChildren.children[index + 1] as Content | undefined;
        if (next) {
          const getText = (n: Content | undefined): string => {
            if (!n) return '';
            if (n.type === 'text') return String((n as Text).value || '');
            if (n.type === 'element') return ((n as Element).children as Content[]).map((child) => getText(child)).join('');
            return '';
          };
          const token = getText(next);
          if (token && /^[A-Za-z0-9_]+$/.test(token) && token.length <= 16) {
            parentWithChildren.children[index + 1] = { type: 'element', tagName: 'sub', properties: {}, children: [next] } as unknown as Content;
          }
        }
      }

      const tag = node.tagName;
      const props = (node.properties || {}) as Record<string, unknown> & { style?: string };
      const style: string | undefined = typeof props.style === 'string' ? props.style : undefined;

      const isInlineSpan = tag === 'span' || tag === 'font';
      const isBold = tag === 'strong' || hasBold(style);
      const isItalic = tag === 'em' || hasItalic(style);
      const isSupTag = tag === 'sup';
      const isSubTag = tag === 'sub';
      const isSup = isSupTag || hasSup(style);
      let isSub = isSubTag || hasSub(style);

      if (!(isInlineSpan || tag === 'strong' || tag === 'em' || tag === 'b' || tag === 'i' || isSupTag || isSubTag)) {
        return;
      }

      // Avoid double wrapping for already semantic strong/em when no sup/sub is needed
      if ((tag === 'strong' || tag === 'em') && !isSup && !isSub) {
        return;
      }

      // Heuristic: detect likely subscript by significantly smaller font-size than parent
      if (!isSup && !isSub && isInlineSpan && style) {
        const childSize = parseFontSize(style);
        const parentStyle = (parent as unknown as Element).properties && typeof (parent as unknown as Element).properties!.style === 'string' ? ((parent as unknown as Element).properties!.style as string) : undefined;
        const parentSize = parseFontSize(parentStyle);
        if (childSize && parentSize && childSize <= parentSize * 0.8) {
          isSub = true;
        }
      }

      // If nothing to do, skip
      if (!isBold && !isItalic && !isSup && !isSub) {
        return;
      }

      // Build replacement nodes
      const originalChildren = node.children || [];
      let children = originalChildren;

      // Remove interpreted pieces from style
      if (props.style) {
        const newStyle = String(props.style)
          .split(';')
          .map((part) => part.trim())
          .filter((part) =>
            part &&
            !/^font-weight\s*:/.test(part) &&
            !/^font-style\s*:/.test(part) &&
            !/^vertical-align\s*:/.test(part) &&
            !/^baseline-shift\s*:/.test(part)
          )
          .join('; ');
        if (newStyle) {
          props.style = newStyle;
        } else {
          delete props.style;
        }
      }

      // Apply italic and bold (inner wrappers) – ensure we wrap all children, not collapse to first element
      if (isItalic) {
        children = [{ type: 'element', tagName: 'em', properties: {}, children }] as Element[];
      }
      if (isBold) {
        children = [{ type: 'element', tagName: 'strong', properties: {}, children }] as Element[];
      }

      // Apply sup/sub (outer wrapper). If node already sup/sub, reuse that tag once
      let replacement: Element = { type: 'element', tagName: 'span', properties: props, children } as Element;
      if (isSup || isSub) {
        const outerTag = isSup ? 'sup' : 'sub';
        replacement = { type: 'element', tagName: outerTag, properties: {}, children } as Element;
      } else if (isBold || isItalic) {
        // When only bold/italic, keep the constructed wrapper with all children, not just first child
        replacement = { type: 'element', tagName: 'span', properties: props, children } as Element;
      }

      parent.children[index] = replacement;
    });
  };
}


