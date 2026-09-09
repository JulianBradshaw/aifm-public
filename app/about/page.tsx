import { cacheLife } from "next/cache";
import Link from "next/link";
import { googleDocToMarkdown } from "@/utils/googleDocToMarkdown";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import { HeaderContent } from "@/components/HeaderContent";
import TableOfContents from "@/components/TableOfContents";

const googleDocId = "1G2oxT1dIwHNggEwXYU_R4VYMANAxJRNb_KZPyRxWTIA";

async function CachedBodyMarkdown() {
  'use cache';
  cacheLife('hours');
  
  return await googleDocToMarkdown(googleDocId);
}

export default async function AboutPage() {
  'use cache';
  cacheLife('hours');

  const markdown = await CachedBodyMarkdown();

  return <>
    <HeaderContent variant="inline" className="pt-6 pb-4 px-6" />
    <div className="relative">
      <TableOfContents />
      <main className="mt-10 prose mx-auto max-w-3xl px-6">
        <MarkdownRenderer markdown={markdown} />
      </main>
    </div>
  </>
}
