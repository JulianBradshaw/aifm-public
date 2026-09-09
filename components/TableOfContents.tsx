'use client';

import { useEffect, useMemo, useState } from 'react';
import type { TocItem } from '@/utils/extractTocFromMarkdown';
import { createSlugger } from '@/utils/slugger';

type TableOfContentsProps = {
  items?: TocItem[];
  headingSelector?: string;
  contentSelector?: string;
  minLevel?: number;
  maxLevel?: number;
  topOffsetPx?: number;
};

function indentClassForLevel(level: number): string {
  if (level <= 2) {
    return '';
  }
  if (level === 3) {
    return 'pl-3';
  }
  return 'pl-5';
}

export default function TableOfContents({
  items,
  headingSelector = 'h2, h3',
  contentSelector = '.prose',
  minLevel = 2,
  maxLevel = 3,
  topOffsetPx = 96,
}: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string | null>(null);
  const [derivedItems, setDerivedItems] = useState<TocItem[]>([]);

  useEffect(() => {
    if (items && items.length > 0) {
      setDerivedItems([]);
      return;
    }

    const root = document.querySelector(contentSelector);
    if (!root) {
      setDerivedItems([]);
      return;
    }

    const slugger = createSlugger();
    const usedIds = new Set<string>();

    const headings = Array.from(root.querySelectorAll(headingSelector)).filter(
      (node): node is HTMLElement => node instanceof HTMLElement
    );

    const nextItems: TocItem[] = [];
    for (const heading of headings) {
      const level = Number(heading.tagName.replace('H', ''));
      if (!Number.isFinite(level) || level < minLevel || level > maxLevel) {
        continue;
      }

      const title = (heading.textContent ?? '').trim();
      if (!title) {
        continue;
      }

      let id = heading.id?.trim();
      if (!id) {
        const generated = slugger(title);
        if (!generated) {
          continue;
        }
        id = generated;
        heading.id = id;
      }

      if (usedIds.has(id)) {
        let i = 2;
        let candidate = `${id}-${i}`;
        while (usedIds.has(candidate)) {
          i += 1;
          candidate = `${id}-${i}`;
        }
        id = candidate;
        heading.id = id;
      }

      usedIds.add(id);
      heading.style.scrollMarginTop ||= '6rem';
      nextItems.push({ id, title, level });
    }

    setDerivedItems(nextItems);
  }, [contentSelector, headingSelector, items]);

  const tocItems = items && items.length > 0 ? items : derivedItems;
  const ids = useMemo(() => tocItems.map((item) => item.id), [tocItems]);
  const idSet = useMemo(() => new Set(ids), [ids]);

  useEffect(() => {
    if (ids.length === 0) {
      return;
    }

    const elements = ids
      .map((id) => document.getElementById(id))
      .filter((element): element is HTMLElement => element instanceof HTMLElement);

    if (elements.length === 0) {
      return;
    }

    const observed = new Set<HTMLElement>();
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .map((entry) => entry.target)
          .filter((node): node is HTMLElement => node instanceof HTMLElement)
          .filter((node) => idSet.has(node.id));

        if (visible.length === 0) {
          return;
        }

        visible.sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);
        setActiveId(visible[0].id);
      },
      {
        root: null,
        rootMargin: '-20% 0px -72% 0px',
        threshold: [0, 1],
      }
    );

    for (const element of elements) {
      observer.observe(element);
      observed.add(element);
    }

    return () => {
      for (const element of observed) {
        observer.unobserve(element);
      }
      observer.disconnect();
    };
  }, [headingSelector, idSet, ids.length]);

  if (tocItems.length === 0) {
    return null;
  }

  return (
    <nav
      aria-label="Table of contents"
      className="hidden xl:block xl:fixed xl:left-3 xl:z-20 xl:w-44"
      style={{ top: `${topOffsetPx}px` }}
    >
      <div className="border-l border-neutral-300 pl-3">

        <ol className="max-h-[calc(100vh-7rem)] overflow-auto scrollbar-none pr-2">
          {tocItems.map((item) => {
            const isActive = item.id === activeId;
            return (
              <li key={item.id} className={indentClassForLevel(item.level)}>
                <a
                  href={`#${item.id}`}
                  onClick={(event) => {
                    event.preventDefault();
                    const target = document.getElementById(item.id);
                    if (!target) {
                      return;
                    }
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    setActiveId(item.id);
                    window.history.replaceState(null, '', `#${item.id}`);
                  }}
                  className={[
                    'group relative block py-1 text-[12px] leading-4 transition-colors',
                    'text-neutral-600 hover:text-neutral-900',
                    isActive ? 'text-neutral-950 font-medium' : '',
                  ].join(' ')}
                >
                  <span
                    className={[
                      'absolute left-[-10px] top-[10px] h-1 w-1 rounded-full transition-opacity',
                      isActive ? 'opacity-100 bg-[var(--accent-color)]' : 'opacity-0 group-hover:opacity-50 bg-neutral-400',
                    ].join(' ')}
                  />
                  <span className="line-clamp-2">{item.title}</span>
                </a>
              </li>
            );
          })}
        </ol>
      </div>
    </nav>
  );
}


