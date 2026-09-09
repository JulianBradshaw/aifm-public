export function slugifyHeading(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/&/g, ' and ')
    .replace(/['"]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

export function createSlugger() {
  const counts = new Map<string, number>();

  return (headingText: string): string | undefined => {
    const base = slugifyHeading(headingText);
    if (!base) {
      return undefined;
    }

    const next = (counts.get(base) ?? 0) + 1;
    counts.set(base, next);
    return next === 1 ? base : `${base}-${next}`;
  };
}


