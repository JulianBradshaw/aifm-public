import { googleDocToMarkdown } from '@/utils/googleDocToMarkdown';

async function main() {
  const id = process.argv[2] || '1_Fe34EcaYP5xLXtcfydZYH9-8-0zpSB2_DilV-0CwQM';
  const md = await googleDocToMarkdown(id);
  console.log('MD length:', md.length);
  console.log(md);
  console.log('Has <sub>:', /<sub>/i.test(md));
  console.log('Has <sup>:', /<sup>/i.test(md));
  const idx = md.toLowerCase().indexOf('train');
  if (idx >= 0) {
    console.log('Context around "train":', md.slice(Math.max(0, idx - 40), idx + 40));
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});


