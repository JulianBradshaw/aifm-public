import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkBreaks from 'remark-breaks';
import rehypeRaw from 'rehype-raw';
import rehypeKatex from 'rehype-katex';
import { slugifyHeading } from '@/utils/slugger';

/**
 * When exporting a google doc, external links get rewritten to `https://www.google.com/url?q=[original_url]...`
 * This function strips the redirect and returns the original URL.
 */
function stripGoogleUrlRedirect(url: string): string {
  try {
    const urlObj = new URL(url);
    const pathname = urlObj.pathname;
    if (pathname !== '/url') {
      return url;
    }
    const urlParam = urlObj.searchParams.get('q');
    if (urlParam) {
      return urlParam;
    }
    return url;
  } catch (err) {
    console.error(`Error stripping Google URL redirect from ${url}: ${err}`);
    return url;
  }
}

// Heading ids must be assigned inside the rehype pipeline, NOT during React render.
// The pipeline runs once per processed document with a fresh counter, so the ids are
// deterministic and identical between the server render and client hydration. The old
// approach (a stateful slugger closure called from the h1–h6 component renderers) broke
// under React's dev double-render: the second pass saw an already-incremented counter
// and emitted `-2`-suffixed ids on every heading, causing a hydration mismatch per
// heading. Slug format (slugifyHeading + bare first occurrence, `-N` for repeats) is
// unchanged, and matches extractTocFromMarkdown / TableOfContents.
type HastNode = {
  type: string;
  tagName?: string;
  value?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
};

function hastText(node: HastNode): string {
  if (node.type === 'text') {
    return node.value ?? '';
  }
  return (node.children ?? []).map(hastText).join('');
}

function rehypeHeadingIds() {
  return (tree: HastNode) => {
    const counts = new Map<string, number>();
    const walk = (node: HastNode) => {
      if (node.type === 'element' && node.tagName && /^h[1-6]$/.test(node.tagName)) {
        node.properties = node.properties ?? {};
        if (!node.properties.id) {
          const base = slugifyHeading(hastText(node).trim());
          if (base) {
            const next = (counts.get(base) ?? 0) + 1;
            counts.set(base, next);
            node.properties.id = next === 1 ? base : `${base}-${next}`;
          }
        }
      }
      (node.children ?? []).forEach(walk);
    };
    walk(tree);
  };
}

export default function MarkdownRenderer({ markdown }: { markdown: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkBreaks, [remarkMath, { singleDollarTextMath: false }]]}
      rehypePlugins={[rehypeRaw, rehypeHeadingIds, rehypeKatex]}
      skipHtml={false}
      components={{
        p: ({ children }) => <p className="mb-4">{children}</p>,
        h1: ({ id, children }) => (
          <h1 id={id} className="text-lg font-semibold font-mono mb-4 mt-14 first:mt-0 scroll-mt-24">
            {children}
          </h1>
        ),
        h2: ({ id, children }) => (
          <h2 id={id} className=" font-semibold font-mono mb-4 mt-14 first:mt-0 scroll-mt-24">
            {children}
          </h2>
        ),
        h3: ({ id, children }) => (
          <h3 id={id} className="font-semibold font-mono mb-4 mt-12 first:mt-0 scroll-mt-24">
            {children}
          </h3>
        ),
        h4: ({ id, children }) => (
          <h4 id={id} className="text-base font-semibold font-mono mb-4 mt-10 first:mt-0 scroll-mt-24">
            {children}
          </h4>
        ),
        h5: ({ id, children }) => (
          <h5 id={id} className="text-sm font-semibold font-mono mb-4 mt-8 first:mt-0 scroll-mt-24">
            {children}
          </h5>
        ),
        h6: ({ id, children }) => (
          <h6 id={id} className="text-xs font-semibold font-mono mb-4 mt-8 first:mt-0 scroll-mt-24">
            {children}
          </h6>
        ),
        ul: ({ children }) => <ul className="list-disc mb-4">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-inside mb-4">{children}</ol>,
        li: ({ children }) => <li className="mb-2 ml-4">{children}</li>,
        a: ({ href, children, className, id }) => {
          // Internal anchor links. Preserve `id` so in-text markers (e.g. the covert
          // supplement's `<a id="covert-fnref-N">` footnote refs) keep their anchor id —
          // the sidenote layout looks them up by id to align each note to its marker.
          if (href?.startsWith('#')) {
            return <a id={id} href={href} className="text-accent hover:text-accent/80">{children}</a>;
          }

          // Footnote references
          if (className === 'footnote-ref') {
            return <a href={href} className="text-blue-400 hover:text-blue-300 underline text-sm">{children}</a>;
          }

          const cleanHref = href && stripGoogleUrlRedirect(href);

          // Same-page anchor links to this site (e.g. https://aifuturesmodel.com/#section)
          const sitePatterns = [
            'https://aifuturesmodel.com/',
            'https://www.aifuturesmodel.com/',
            'http://aifuturesmodel.com/',
            'http://www.aifuturesmodel.com/',
          ];
          const matchedPattern = sitePatterns.find(pattern => cleanHref?.startsWith(pattern + '#'));
          if (matchedPattern) {
            const anchor = cleanHref!.replace(matchedPattern, '');
            return <a href={anchor} className="text-accent hover:text-accent/80">{children}</a>;
          }

          // External links
          return <a href={cleanHref} target="_blank" rel="noopener noreferrer" className="text-accent hover:text-accent/80">{children}</a>;
        },
        img: ({ src, alt }) => src ? <img src={src} alt={alt} className="mb-4 mx-auto mix-blend-darken" /> : null,
        blockquote: ({ children }) => <blockquote className="border-l-4 border-gray-300 pl-4 mb-4">{children}</blockquote>,
        code: ({ children }) => <code className="bg-gray-100 px-2 py-1 rounded">{children}</code>,
        pre: ({ children }) => <pre className="bg-gray-100 px-4 py-2 rounded">{children}</pre>,
        table: ({ children }) => <table className="border-collapse border border-gray-300 mb-4">{children}</table>,
        // The tables we have don't have headers and we don't want them otherwise hanging around looking weird
        thead: ({ children }) => <thead className="hidden">{children}</thead>,
        tbody: ({ children }) => <tbody>{children}</tbody>,
        tr: ({ children }) => <tr className="border-b border-gray-300">{children}</tr>,
        th: ({ children }) => <th className="border border-gray-300 px-4 py-2 text-left font-bold">{children}</th>,
        td: ({ children }) => <td className="border border-gray-300 px-4 py-2">{children}</td>,
        br: () => <br />,
      }}
    >
      {markdown}
    </ReactMarkdown>
  );
}


