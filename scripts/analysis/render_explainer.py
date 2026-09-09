#!/usr/bin/env python3
"""Render the plain-English explainer to HTML.

Math is pre-rendered to bare MathML with the repo's vendored KaTeX, so the page
needs no stylesheet, no webfonts and no JavaScript. Figures are referenced
relative to this directory by default; pass --inline to embed them as data URIs
for a single self-contained file (that version is ~0.5 MB and noticeably slower
for the browser to paint).

    python3 render_explainer.py [--inline]
"""
import base64, json, pathlib, re, subprocess, sys, markdown

HERE = pathlib.Path(__file__).parent
REPO = HERE.parents[1]

MATH_DISPLAY = re.compile(r'\$\$(.+?)\$\$', re.S)
MATH_INLINE = re.compile(r'(?<![\$\\])\$([^\$\n]+?)\$(?!\$)')


def tex_to_mathml(items):
    """Pre-render LaTeX to bare MathML with the repo's vendored KaTeX.

    output:'mathml' needs no stylesheet and no webfonts -- Chrome typesets MathML
    natively -- so the page stays a single self-contained file.
    """
    if not items:
        return []
    script = """
    const katex = require(process.argv[1]);
    let raw = '';
    process.stdin.on('data', d => raw += d);
    process.stdin.on('end', () => {
      const items = JSON.parse(raw);
      process.stdout.write(JSON.stringify(items.map(
        ([tex, disp]) => katex.renderToString(tex, {
          output: 'mathml', displayMode: disp, throwOnError: false, strict: false})
      )));
    });
    """
    out = subprocess.run(['node', '-e', script, str(REPO / 'node_modules' / 'katex')],
                         input=json.dumps(items), capture_output=True, text=True, check=True)
    # KaTeX wraps MathML in <span class="katex">; keep the span for styling hooks
    return json.loads(out.stdout)
SRC = HERE / 'uplift_unsolvable_plain_english.md'
OUT = HERE / 'uplift_unsolvable_plain_english.html'

CSS = """
:root{--bg:#fffff8;--fg:#1a1a1a;--muted:#5a5a52;--rule:#d8d8cc;--accent:#1f4e79}
@media (prefers-color-scheme:dark){:root{--bg:#16160f;--fg:#eceadf;--muted:#a3a396;--rule:#3a3a30;--accent:#7fb3e0}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:3rem 1.5rem 6rem;
  font:16.5px/1.65 Charter,Georgia,"Iowan Old Style",serif}
main{max-width:47rem;margin:0 auto}
h1{font-size:2.1rem;line-height:1.2;margin:0 0 .4rem;letter-spacing:-.01em}
h2{font-size:1.42rem;margin:3rem 0 .8rem;padding-bottom:.35rem;border-bottom:1px solid var(--rule);scroll-margin-top:1.5rem}
h3{margin:2.1rem 0 .5rem;color:var(--muted);text-transform:uppercase;letter-spacing:.055em;
  font-size:.85rem;font-weight:700;scroll-margin-top:1.5rem}
p,li{margin:.85rem 0}
em{color:var(--muted)}
a{color:var(--accent)}
hr{border:0;border-top:1px solid var(--rule);margin:2.6rem 0}
img{max-width:100%;height:auto;aspect-ratio:auto;display:block;margin:1.4rem auto;border:1px solid var(--rule);border-radius:3px}
blockquote{margin:1.6rem 0;padding:.9rem 1.3rem;border-left:3px solid var(--accent);
  background:color-mix(in srgb,var(--accent) 7%,transparent);border-radius:0 4px 4px 0}
blockquote p:first-child{margin-top:0}blockquote p:last-child{margin-bottom:0}
blockquote h3{margin-top:0;color:var(--accent);text-transform:none;letter-spacing:0;font-size:1.05rem}
code{font:.88em/1.4 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  background:color-mix(in srgb,var(--fg) 8%,transparent);padding:.12em .38em;border-radius:3px}
a code{color:inherit}
table{border-collapse:collapse;width:100%;margin:1.5rem 0;font-size:.94em;display:block;overflow-x:auto}
th,td{border:1px solid var(--rule);padding:.55rem .75rem;text-align:left;vertical-align:top}
th{background:color-mix(in srgb,var(--fg) 6%,transparent);font-weight:600}
strong{font-weight:640}
sup,sub{font-size:.72em;line-height:0}
.formula{margin:1.8rem 0;padding:1.35rem 1.6rem;border:1px solid var(--rule);border-radius:5px;
  background:color-mix(in srgb,var(--fg) 3.5%,transparent)}
.formula p{margin:.4rem 0}
.formula p.lbl{font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  font-weight:700;margin-top:1.15rem}
.formula p.lbl:first-child{margin-top:0}
.formula p.eq{font-size:1.05rem;color:var(--accent);padding:.1rem 0 .3rem 1.1rem}
.formula p.katex-line{padding-left:0}
math{font-size:1.14em;color:var(--accent)}
math[display="block"]{display:block;margin:.75rem 0 1.05rem 1.6rem;text-align:left}
p math[display="block"]{margin-top:.35rem}
.katex{display:inline}
span.katex math[display="block"]{margin-left:1.6rem}
:not(.formula) > p math{color:inherit;font-size:1.05em}
.formula code{background:none;padding:0;font-size:.95em;color:inherit}
.headerlink{display:none}
"""


def main():
    text = SRC.read_text()

    # Pull math out before markdown touches it (backslashes, underscores, asterisks)
    maths, tokens = [], []

    def stash(m, display):
        tokens.append(f'@@MATH{len(maths)}@@')
        maths.append([m.group(1).strip(), display])
        return tokens[-1]

    text = MATH_DISPLAY.sub(lambda m: stash(m, True), text)
    text = MATH_INLINE.sub(lambda m: stash(m, False), text)

    html = markdown.markdown(text,
                             extensions=['tables', 'attr_list', 'sane_lists', 'md_in_html', 'toc'])

    if '--inline' in sys.argv:
        def embed(m):
            f = HERE / m.group(2)
            if not f.exists():
                return m.group(0)
            return f'{m.group(1)}data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}"'

        html = re.sub(r'(<img [^>]*src=")([^"]+)"', embed, html)

    for i, rendered in enumerate(tex_to_mathml(maths)):
        html = html.replace(f'@@MATH{i}@@', rendered)
    html = html.replace('href="uplift_unsolvable_region.md"',
                        'href="uplift_unsolvable_region.md" target="_blank"')
    # tag the formula block: bold-only lines are labels, the rest are equations
    def tag(m):
        body = m.group(1)
        body = re.sub(r'<p>(<strong>.*?</strong>[^<]*)</p>', r'<p class="lbl">\1</p>', body)
        body = re.sub(r'<p>(?!<strong)', '<p class="eq">', body)
        body = body.replace('<p class="eq"><span class="katex">',
                            '<p class="eq katex-line"><span class="katex">')
        return f'<div class="formula">{body}</div>'
    html = re.sub(r'<div class="formula">(.*?)</div>', tag, html, flags=re.S)

    OUT.write_text('<!doctype html><html><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width,initial-scale=1">'
                   '<title>Why some rollouts die with "Uplift Schedule Unsolvable"</title>'
                   f'<style>{CSS}</style></head><body><main>{html}</main></body></html>')
    print('wrote', OUT, OUT.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
