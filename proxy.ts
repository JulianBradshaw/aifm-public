import { NextRequest, NextResponse } from 'next/server';
import { isLegacyShareLink, LEGACY_HOST, SHARE_LINK_PATHS } from '@/utils/legacyLinkRouting';

// Embed-only hosts.
//
// The plan-a Vercel project sits behind Vercel deployment protection (SSO),
// which also blocks iframing: the login interstitial sends X-Frame-Options:
// DENY. So the ai-2030 Plan A site can't frame the takeoff explorer from the
// protected domain. Instead, the SAME project carries a second domain (below)
// listed under Vercel's Deployment Protection → Exceptions, so that hostname
// alone is publicly reachable — and this proxy limits requests on it to
// exactly the explorer embed (/plan-a) and the API routes it calls, so
// nothing else on the site becomes public through the exempted domain. Every
// other path there redirects to the embed.
//
// On every other hostname (the protected plan-a domain, local dev, preview
// URLs) this proxy is a no-op. EMBED_ONLY_DEPLOYMENT=1 forces embed-only
// mode regardless of hostname — for local testing, or a dedicated embed-only
// project if one is ever wanted.
const EMBED_ONLY_HOSTS = new Set(['embed.aifuturesmodel.com']);

const EMBED_ALLOWED_PREFIXES = [
  '/plan-a', // the explorer page (the ai-2030 site frames /plan-a?embed=1)
  '/api/plan-a', // simulation + defaults (also matches /api/plan-a/defaults)
  '/api/compute', // default input series
  '/api/parameter-config', // slider ranges/labels
  '/_next', // JS/CSS chunks, next/font files
  '/fonts', // et-book/FiraMath @font-face files (public/fonts, referenced from globals.css)
  '/favicon.ico',
];

export function proxy(req: NextRequest) {
  const host = (req.headers.get('host') ?? '').split(':')[0].toLowerCase();
  const embedOnly = EMBED_ONLY_HOSTS.has(host) || process.env.EMBED_ONLY_DEPLOYMENT === '1';
  if (embedOnly) {
    const { pathname } = req.nextUrl;
    const allowed = EMBED_ALLOWED_PREFIXES.some(
      (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
    );
    if (allowed) return NextResponse.next();
    return NextResponse.redirect(new URL('/plan-a?embed=1', req.url));
  }

  // Legacy share-link routing (see utils/legacyLinkRouting.ts): Dec-2025-era
  // parameter links — no `v` marker, but a `base` or an old codec key — are
  // served by the frozen Dec-2025 deployment, which interprets them exactly
  // as production did when they were minted. 302 (not 308) so the mapping
  // stays revisable. Skipped in dev so local work on old presets isn't
  // bounced off-machine.
  if (process.env.NODE_ENV !== 'development') {
    const { pathname, search, searchParams } = req.nextUrl;
    if (SHARE_LINK_PATHS.has(pathname) && isLegacyShareLink(searchParams)) {
      return NextResponse.redirect(`${LEGACY_HOST}${pathname}${search}`, 302);
    }
  }

  return NextResponse.next();
}
