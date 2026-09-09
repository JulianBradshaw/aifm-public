import { cacheLife, cacheTag } from 'next/cache';
import { getCachedGDocMarkdown } from '@/lib/cachedGDocMarkdown';
import MarkdownRenderer from './MarkdownRenderer';

// Separate cached component for the Google Doc content
// This allows it to load independently from the main chart
async function CachedGoogleDocContent() {
  'use cache';
  cacheLife('hours');
  cacheTag('gdoc-content');
  
  const gDocMarkdown = await getCachedGDocMarkdown();
  
  return <MarkdownRenderer markdown={gDocMarkdown} />;
}

export default async function GoogleDocSection() {
  return <CachedGoogleDocContent />;
}

