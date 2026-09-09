import 'server-only';
import { cacheLife, cacheTag } from 'next/cache';
import { googleDocToMarkdown } from '@/utils/googleDocToMarkdown';

const GOOGLE_DOC_ID = '1aMgKau-Wmq2dCMEIHDanDqUnAI4_2R12yZw2WJ58_-M';

// Cached function using Next.js 16 'use cache' directive
export async function getCachedGDocMarkdown(): Promise<string> {
  'use cache';
  cacheLife('hours');
  cacheTag('gdoc-markdown');
  
  console.log('[getCachedGDocMarkdown] Cache miss - fetching Google Doc');
  return googleDocToMarkdown(GOOGLE_DOC_ID);
}

