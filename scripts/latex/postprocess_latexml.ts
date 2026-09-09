#!/usr/bin/env tsx

import * as fs from 'fs';
import * as path from 'path';
import { parseHTML } from 'linkedom';

// Find the workspace root (where public/ is located)
// Go up from scripts/latex to the workspace root
const scriptDir = path.dirname(path.dirname(__filename));
const workspaceRoot = path.resolve(scriptDir, '..');

function postprocessHtml(inputFile: string): number {
  // Read the LaTeXML HTML file
  const content = fs.readFileSync(inputFile, 'utf-8');

  // Extract footnotes and their content
  const footnotes: Record<string, string> = {};

  // Pattern to match LaTeXML footnote structure (using [\s\S] instead of . with s flag for ES2017 compat)
  const footnotePattern = /<span id="footnote(\d+)" class="ltx_note ltx_role_footnote">[\s\S]*?<span class="ltx_note_content">[\s\S]*?<span class="ltx_tag ltx_tag_note">\d+<\/span>([\s\S]*?)<\/span><\/span><\/span>/g;

  const matches = [...content.matchAll(footnotePattern)];

  // Process each footnote
  let processedContent = content;
  for (const match of matches) {
    const num = match[1];
    const text = match[2].trim();

    // Create footnote link
    const footnoteLink = `<sup><a href="#fn${num}" id="fnref${num}">${num}</a></sup>`;

    // Store footnote content
    footnotes[num] = `<li id="fn${num}"><p>${text} <a href="#fnref${num}">↩</a></p></li>`;

    // Replace the entire inline footnote with just the link
    const inlinePattern = new RegExp(`<span id="footnote${num}" class="ltx_note ltx_role_footnote">[\\s\\S]*?<\\/span><\\/span><\\/span>`, 'g');
    processedContent = processedContent.replace(inlinePattern, footnoteLink);
  }

  // Remove any existing footnotes section
  processedContent = processedContent.replace(/<div class="footnotes">[\s\S]*?<\/div>/g, '');

  // Remove LaTeXML footer
  processedContent = processedContent.replace(/<footer class="ltx_page_footer">[\s\S]*?<\/footer>/g, '');

  processedContent = transformExpandables(processedContent);

  // Fix multiple non-breaking spaces after section numbers (e.g., "1.&#160;&#160;&#160;" → "1. ")
  processedContent = processedContent.replace(/([\d.]+)\.(&#160;)+/g, '$1. ');

  // Handle images: copy to public and update paths
  processedContent = handleImages(processedContent, inputFile);

  // Add footnotes section at the end if we have footnotes
  if (Object.keys(footnotes).length > 0) {
    let footnotesSection = '<div class="footnotes"><h2>Footnotes</h2><ol>';
    // Sort footnotes by number
    const sortedNums = Object.keys(footnotes).sort((a, b) => parseInt(a) - parseInt(b));
    for (const num of sortedNums) {
      footnotesSection += footnotes[num];
    }
    footnotesSection += '</ol></div>';

    // Insert before closing body tag
    processedContent = processedContent.replace('</body>', footnotesSection + '\n</body>');
  }

  // Extract only the body contents (without the <body> tag itself)
  processedContent = extractBodyContents(processedContent);

  // Write the fixed HTML back to the same file
  fs.writeFileSync(inputFile, processedContent, 'utf-8');

  const inputFileName = path.basename(inputFile);
  const publicHtmlFile = path.join(workspaceRoot, 'public', inputFileName);
  fs.writeFileSync(publicHtmlFile, processedContent, 'utf-8');

  return Object.keys(footnotes).length;
}

function extractBodyContents(content: string): string {
  // Parse the HTML using linkedom
  const { document } = parseHTML(content);
  
  // Get the body element
  const body = document.body;
  
  if (!body) {
    console.warn('Warning: No <body> tag found in HTML, returning content as-is');
    return content;
  }
  
  // Return the innerHTML of the body (contents without the body tag itself)
  return body.innerHTML.trim();
}

function handleImages(content: string, htmlFilePath: string): string {
  const htmlDir = path.dirname(htmlFilePath);
  const publicDir = path.join(workspaceRoot, 'public', 'latex');
  
  // Ensure public/latex directory exists
  if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir, { recursive: true });
  }
  
  // Parse the HTML using linkedom
  const { document } = parseHTML(content);
  
  const imagesCopied: string[] = [];
  
  // Find all img elements
  const images = document.querySelectorAll('img');
  
  for (const img of images) {
    const originalSrc = img.getAttribute('src');
    
    if (!originalSrc) {
      continue;
    }
    
    // Skip if it's already an absolute URL or starts with /
    if (originalSrc.startsWith('http://') || 
        originalSrc.startsWith('https://') || 
        originalSrc.startsWith('/')) {
      continue;
    }
    
    // Resolve the image path relative to the HTML file
    const imageSourcePath = path.resolve(htmlDir, originalSrc);
    
    // Check if the image file exists
    if (!fs.existsSync(imageSourcePath)) {
      console.warn(`Warning: Image not found: ${imageSourcePath}`);
      continue;
    }
    
    // Get just the filename for the destination
    const imageFilename = path.basename(imageSourcePath);
    const imageDestPath = path.join(publicDir, imageFilename);
    
    // Copy the image to public/latex/
    fs.copyFileSync(imageSourcePath, imageDestPath);
    imagesCopied.push(imageFilename);
    
    // Update the src attribute to point to /latex/filename
    img.setAttribute('src', `latex/${imageFilename}`);
  }
  
  if (imagesCopied.length > 0) {
    console.log(`Copied ${imagesCopied.length} image(s) to public/latex/: ${imagesCopied.join(', ')}`);
  }
  
  // Return the modified HTML
  return document.toString();
}

function slugify(text: string): string {
  // Extract text content only (strip HTML tags)
  const textOnly = text.replace(/<[^>]*>/g, '');
  return textOnly
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '') // Remove non-word chars except spaces and hyphens
    .replace(/\s+/g, '-')     // Replace spaces with hyphens
    .replace(/-+/g, '-')      // Collapse multiple hyphens
    .replace(/^-|-$/g, '');   // Trim leading/trailing hyphens
}

function transformExpandables(content: string): string {
  const { document } = parseHTML(content);
  const expandableSpans = Array.from(document.querySelectorAll('span')).filter(
    (span) => span.textContent && span.textContent.includes('Expandable:')
  );

  for (const span of expandableSpans) {
    if (!document.contains(span)) {
      continue;
    }
    if (!span.parentNode) {
      continue;
    }

    const summaryHtml = extractExpandableSummary(span);
    if (!summaryHtml) {
      continue;
    }

    const endSpan = findMatchingEndSpan(span);
    if (!endSpan) {
      continue;
    }
    if (!endSpan.parentNode) {
      continue;
    }

    const bodyFragment = extractExpandableBody(document, span, endSpan);
    const details = document.createElement('details');
    const summary = document.createElement('summary');

    // Generate an id from the summary text
    const summaryId = slugify(summaryHtml);
    if (summaryId) {
      summary.setAttribute('id', summaryId);
    }

    summary.innerHTML = summaryHtml;
    details.appendChild(summary);
    if (bodyFragment.childNodes.length > 0) {
      details.appendChild(bodyFragment);
    }

    const startBlock = span.closest('div.ltx_para') ?? span.parentElement;
    const endBlock = endSpan.closest('div.ltx_para') ?? endSpan.parentElement;
    if (!startBlock || !endBlock || !startBlock.parentNode) {
      continue;
    }

    replaceBlocksWithDetails(details, startBlock, endBlock);
  }

  return document.toString();
}

function extractExpandableSummary(span: Element): string | null {
  const innerHtml = span.innerHTML;
  const markerIndex = innerHtml.indexOf('Expandable:');
  if (markerIndex === -1) {
    return null;
  }
  const summaryHtml = innerHtml
    .slice(markerIndex + 'Expandable:'.length)
    .trim();
  return summaryHtml || null;
}

function extractExpandableBody(
  document: Document,
  startSpan: Element,
  endSpan: Element
): DocumentFragment {
  const startBlock = startSpan.closest('div.ltx_para') ?? startSpan;
  const endBlock = endSpan.closest('div.ltx_para') ?? endSpan;
  const fragment = document.createDocumentFragment();

  if (startBlock === endBlock) {
    const clone = cloneBlockWithoutMarkers(document, startBlock, {
      removeSummary: true,
      removeEnd: true,
    });
    if (clone) {
      fragment.appendChild(clone);
    }
    return fragment;
  }

  const startClone = cloneBlockWithoutMarkers(document, startBlock, {
    removeSummary: true,
    removeEnd: false,
  });
  if (startClone) {
    fragment.appendChild(startClone);
  }

  let sibling: Node | null = startBlock.nextSibling;
  while (sibling && sibling !== endBlock) {
    const clone = cloneNodeIfMeaningful(document, sibling);
    if (clone) {
      fragment.appendChild(clone);
    }
    sibling = sibling.nextSibling;
  }

  const endClone = cloneBlockWithoutMarkers(document, endBlock, {
    removeSummary: false,
    removeEnd: true,
  });
  if (endClone) {
    fragment.appendChild(endClone);
  }

  return fragment;
}

function cloneBlockWithoutMarkers(
  document: Document,
  block: Element,
  options: { removeSummary: boolean; removeEnd: boolean }
): Element | null {
  const clone = block.cloneNode(true) as Element;
  if (options.removeSummary) {
    removeSpanByLabel(clone, 'Expandable:');
  }
  if (options.removeEnd) {
    removeSpanByLabel(clone, 'End of expandable');
  }
  cleanupEmptyParagraphs(clone);
  return hasMeaningfulContent(clone) ? clone : null;
}

function removeSpanByLabel(root: Element, label: string): void {
  const spans = Array.from(root.querySelectorAll('span'));
  const target = spans.find((span) => span.textContent?.includes(label));
  if (target) {
    target.remove();
  }
}

function cleanupEmptyParagraphs(root: Element): void {
  const paragraphs = Array.from(root.querySelectorAll('p'));
  for (const paragraph of paragraphs) {
    const text = paragraph.textContent?.trim() ?? '';
    if (!text && paragraph.children.length === 0) {
      paragraph.remove();
    }
  }
}

function hasMeaningfulContent(node: Element): boolean {
  const text = node.textContent?.replace(/\s+/g, '') ?? '';
  if (text.length > 0) {
    return true;
  }
  return Boolean(
    node.querySelector(
      'img,figure,math,table,details,ol,ul,li,code,pre,blockquote'
    )
  );
}

function cloneNodeIfMeaningful(document: Document, node: Node): Node | null {
  if (node.nodeType === node.TEXT_NODE) {
    if (node.textContent?.trim()) {
      return document.createTextNode(node.textContent);
    }
    return null;
  }
  if (node.nodeType === node.ELEMENT_NODE) {
    const clone = (node as Element).cloneNode(true) as Element;
    return hasMeaningfulContent(clone) ? clone : null;
  }
  return null;
}

function replaceBlocksWithDetails(
  details: Element,
  startBlock: Element,
  endBlock: Element
): void {
  const parent = startBlock.parentNode;
  if (!parent) {
    return;
  }
  parent.insertBefore(details, startBlock);
  let node: Node | null = startBlock;
  while (node) {
    const next: Node | null = node.nextSibling;
    parent.removeChild(node);
    if (node === endBlock) {
      break;
    }
    node = next;
  }
}

function findMatchingEndSpan(startSpan: Element): Element | null {
  let depth = 0;
  let current: Node | null = startSpan;
  while ((current = nextNode(current)) !== null) {
    if (
      current.nodeType === current.ELEMENT_NODE &&
      (current as Element).tagName.toLowerCase() === 'span'
    ) {
      const text = (current.textContent || '').trim();
      if (text.startsWith('Expandable:')) {
        depth += 1;
      } else if (text.startsWith('End of expandable')) {
        if (depth === 0) {
          return current as Element;
        }
        depth -= 1;
      }
    }
  }
  return null;
}

function nextNode(node: Node | null): Node | null {
  if (!node) {
    return null;
  }
  if (node.firstChild) {
    return node.firstChild;
  }
  let current: Node | null = node;
  while (current) {
    if (current.nextSibling) {
      return current.nextSibling;
    }
    current = current.parentNode;
  }
  return null;
}

if (require.main === module) {
  if (process.argv.length !== 3) {
    console.log('Usage: tsx postprocess_latexml.ts <html_file>');
    process.exit(1);
  }

  const htmlFile = process.argv[2];
  const footnoteCount = postprocessHtml(htmlFile);
  console.log(`Post-processed HTML (${footnoteCount} footnotes normalized)`);
}

