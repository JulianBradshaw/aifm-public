import { Suspense } from 'react';
import { cacheLife, cacheTag } from 'next/cache';
import MarkdownRenderer from '@/components/MarkdownRenderer';
import { googleDocToMarkdown } from '@/utils/googleDocToMarkdown';
import TableOfContents from '@/components/TableOfContents';

async function CachedDocContent({ id }: { id: string }) {
  'use cache';
  cacheLife('hours');
  cacheTag(`gdoc-${id}`);
  
  const markdown = await googleDocToMarkdown(id);
  return (
    <div className="relative">
      <TableOfContents />
      <div className="prose mx-auto p-6 max-w-3xl">
        <MarkdownRenderer markdown={markdown} />
      </div>
      <hr />
      <div>
        <p>Renderer sanity test:</p>
        <MarkdownRenderer markdown={"x<sup>2</sup> + H<sub>2</sub>O"} />
      </div>
    </div>
  );
}

async function DocPageContent({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <CachedDocContent id={id} />;
}

export default function Page({ params }: { params: Promise<{ id: string }> }) {
  return (
    <Suspense fallback={
      <div className="flex min-h-screen items-center justify-center text-sm text-slate-500">
        Loading document...
      </div>
    }>
      <DocPageContent params={params} />
    </Suspense>
  );
}
