import { createSlugger } from '@/utils/slugger';

export type TocItem = {
  id: string;
  title: string;
  level: number;
};

function stripInlineMarkdown(text: string): string {
  return text
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_~]/g, '')
    .replace(/<[^>]+>/g, '')
    .trim();
}

export function extractTocFromMarkdown(
  markdown: string,
  options?: { minLevel?: number; maxLevel?: number }
): TocItem[] {
  const minLevel = options?.minLevel ?? 2;
  const maxLevel = options?.maxLevel ?? 3;
  const slugger = createSlugger();

  const items: TocItem[] = [];
  const lines = markdown.split('\n');

  for (const rawLine of lines) {
    const line = rawLine.trim();

    // Skip fenced code blocks quickly (best-effort)
    if (line.startsWith('```')) {
      continue;
    }

    const match = /^(#{1,6})\s+(.+?)\s*$/.exec(line);
    if (!match) {
      continue;
    }

    const level = match[1].length;
    if (level < minLevel || level > maxLevel) {
      continue;
    }

    const title = stripInlineMarkdown(match[2].replace(/\s+#+\s*$/, ''));
    const id = slugger(title);
    if (!id) {
      continue;
    }

    items.push({ id, title, level });
  }

  return items;
}


