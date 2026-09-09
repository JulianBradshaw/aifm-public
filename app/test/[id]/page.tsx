import { Suspense } from "react";
import { headers } from "next/headers";
import { googleDocToMarkdown } from "@/utils/googleDocToMarkdown";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import { HeaderContent } from "@/components/HeaderContent";

async function TestPageContent({ params }: { params: Promise<{ id: string }> }) {
  // Access headers first to opt into dynamic rendering before any Math.random() calls
  await headers();

  const { id } = await params;
  const markdown = await googleDocToMarkdown(id, { noCache: true });

  return (
    <main className="mt-10 prose mx-auto max-w-5xl px-6">
      <MarkdownRenderer markdown={markdown} />
    </main>
  );
}

export default function TestPage({ params }: { params: Promise<{ id: string }> }) {
  return <>
    <HeaderContent variant="inline" className="pt-6 pb-4 px-6" />
    <Suspense fallback={
      <div className="flex min-h-screen items-center justify-center text-sm text-slate-500">
        Loading document...
      </div>
    }>
      <TestPageContent params={params} />
    </Suspense>
  </>
}
