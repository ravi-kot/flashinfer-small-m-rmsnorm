from pathlib import Path
import re
from markdown_it import MarkdownIt

root=Path(__file__).parent
readme=(root/'README.md').read_text(encoding='utf-8')
content=readme[readme.index('## 1. The question'):]
ids=['question','runtime-path','change','kernel-results','correctness','model','reproduce','full-decode','future-work','conclusion']
for n,anchor in enumerate(ids,1):
    content = re.sub(rf'^(## {n}\. )',f'<a id="{anchor}"></a>\n\\1',content,count=1,flags=re.MULTILINE)
md=MarkdownIt('commonmark',{'html':True, 'linkify': True}).enable('table')
body=md.render(content)
body=body.replace('<table>','<div class="table-scroll"><table>').replace('</table>','</table></div>')
body=re.sub(r'<a href="(?!#)([^"]+)"',r'<a href="\1" target="_blank" rel="noopener noreferrer"',body)
html_doc='''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="description" content="A FlashInfer CuTe fused-add RMSNorm experiment with up to 1.60x isolated speedup on an RTX 5070 Laptop GPU and validation inside Qwen3.5-2B." />
<meta name="theme-color" content="#faf9f6" />
<title>One Row per Block — FlashInfer RMSNorm · Ravi Kothari</title>
<style>
:root { --ink:#1e2b34; --muted:#60717a; --soft:#8b9699; --border:#e6e8e3; --teal:#126c60; --page:#faf9f6; --paper:#fff; --shade:#f5f7f4; }
*{box-sizing:border-box} html{scroll-behavior:smooth} body{margin:0;background:var(--page);color:var(--ink);font:16px/1.75 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--teal);text-decoration-thickness:1px;text-underline-offset:3px}a:hover{color:#0b5148}
.wrap{max-width:1260px;margin:auto;padding:0 34px}.topbar{height:74px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border)}
.topbar .brand{font-family:Georgia,'Times New Roman',serif;font-size:19px;font-weight:700;letter-spacing:-.3px;color:var(--ink);text-decoration:none}.topbar .tag{color:var(--muted);font-size:12px;letter-spacing:.13em;text-transform:uppercase}
header.hero{padding:75px 0 30px;max-width:1010px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.19em;text-transform:uppercase;color:var(--teal)}
h1{font:700 clamp(2.65rem,5.4vw,4.5rem)/1.09 Georgia,'Times New Roman',serif;letter-spacing:-.045em;margin:21px 0 21px;max-width:930px}
.deck{font-size:clamp(1.03rem,1.6vw,1.29rem);line-height:1.65;color:var(--muted);max-width:780px;margin-bottom:24px}
.meta{font-size:13px;color:var(--soft);letter-spacing:.02em}.meta strong{color:var(--ink);font-weight:600}
.stat-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:34px 0 28px}.stat{border:1px solid var(--border);border-radius:12px;padding:22px 19px;background:var(--paper);min-height:138px}.stat .val{font-size:clamp(1.65rem,2.7vw,2.25rem);font-weight:750;line-height:1.15;letter-spacing:-.055em;color:var(--teal)}.stat .label{font-size:13px;margin-top:12px;line-height:1.5;color:var(--muted)}
.summary{max-width:1010px;background:#eef4f0;border:1px solid #dbe8e0;border-radius:12px;padding:24px 28px;line-height:1.7;color:#284438}.summary strong{color:#143d33}
.figure{max-width:1010px;margin:27px 0 49px;overflow:hidden;border-radius:16px;border:1px solid var(--border);background:var(--paper)}.figure img{display:block;width:100%;height:auto}.figcaption{padding:3px 25px 20px;font-size:12px;color:var(--muted)}
.content-grid{display:grid;grid-template-columns:minmax(0,780px) 235px;gap:65px;padding-bottom:90px;align-items:start}.article{min-width:0;font-size:15.5px}.article h2{font:700 2rem/1.2 Georgia,'Times New Roman',serif;letter-spacing:-.025em;margin:65px 0 19px;scroll-margin-top:24px}.article h3{font:700 1.24rem/1.3 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:39px 0 12px}.article p{margin:0 0 19px}.article ul,.article ol{padding-left:24px}.article li{margin-bottom:8px}.article hr{border:0;border-top:1px solid var(--border);margin:53px 0}
.article blockquote{border-left:3px solid var(--teal);background:#eff4ef;padding:14px 20px;margin:25px 0;border-radius:0 8px 8px 0}.article blockquote p{margin:0}
.table-scroll{overflow-x:auto;margin:19px 0 25px;border:1px solid var(--border);border-radius:9px;background:#fff}.article table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.5;min-width:600px}.article th{text-align:left;background:#f5f7f4;color:#3a4b56;white-space:nowrap;font-weight:650;border-bottom:1px solid var(--border)}.article th,.article td{padding:13px 13px;border-bottom:1px solid #eef0ec}.article tbody tr:last-child td{border-bottom:0}.article tbody tr:hover{background:#f9fbf9}.article td strong{color:var(--teal)}
code{font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;color:#26584d;background:#eef3ef;border-radius:4px;padding:2px 5px}pre{overflow-x:auto;border:1px solid #e4e8e5;background:#f3f6f3;border-radius:9px;padding:19px;margin:23px 0}pre code{background:none;padding:0;color:#34474f;line-height:1.7}
.toc{position:sticky;top:26px;margin-top:55px;border-left:1px solid var(--border);padding-left:20px}.toc .toc-head{color:var(--soft);text-transform:uppercase;letter-spacing:.17em;font-size:11px;font-weight:700;margin-bottom:17px}.toc a{font-size:13px;line-height:1.4;display:block;margin:0 0 12px;text-decoration:none;color:#657680}.toc a:hover{color:var(--teal)}
.footer{border-top:1px solid var(--border);padding:24px 0 52px;color:var(--muted);font-size:13px}
@media(max-width:940px){.content-grid{grid-template-columns:1fr}.toc{display:none}.stat-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.wrap{padding:0 22px}header.hero{padding-top:50px}.article h2{margin-top:50px}}
@media(max-width:530px){html,body{width:100%;max-width:100%;overflow-x:hidden}.wrap{width:100vw;max-width:100vw;overflow-x:hidden}.topbar,header.hero,.stat-grid,.summary,.figure,.content-grid{width:calc(100vw - 44px);max-width:calc(100vw - 44px)}.content-grid,.article,header.hero,.summary,.figure{min-width:0}.stat-grid{grid-template-columns:1fr;gap:9px}.stat{padding:16px 14px;min-height:0}.stat .val{font-size:1.75rem}.topbar{height:62px}.topbar .brand{font-size:16px}.topbar .tag{display:none}.article table{font-size:12px}.summary{padding:18px}.hero h1{font-size:2.15rem;line-height:1.08}.deck{font-size:1rem}h1,.deck,.meta,.summary,.article p{overflow-wrap:anywhere}}
@media print{body{background:#fff}.toc{display:none}.content-grid{display:block}.topbar{height:auto}.stat,.summary{break-inside:avoid}.article h2{break-after:avoid}.article .table-scroll{overflow:visible}}
</style>
</head>
<body>
<div class="wrap"><div class="topbar"><a class="brand" href="index.html">Ravi Kothari <span style="font-weight:400;color:#8b9699">/ Research Notes</span></a><span class="tag">GPU performance · 2026</span></div>
<header class="hero"><h1>One Row per Block:<br/>Up to 1.60x Faster Fused RMSNorm</h1>
<div class="deck">A small CUDA block-mapping change, measured from one kernel to real Qwen3.5-2B decoding in SGLang.</div>
<div class="meta"><strong>Ravi Kothari</strong> &nbsp;·&nbsp; October 21, 2026 &nbsp;·&nbsp; RTX 5070 Laptop (SM120) &nbsp;·&nbsp; BF16</div></header>
<div class="stat-grid"><div class="stat"><div class="val">1.60x</div><div class="label">Isolated speedup at M=8<br/>from 1.625 to 1.018 µs</div></div><div class="stat"><div class="val">1.25x</div><div class="label">Norm critical-path time in Qwen3.5-2B<br/>at batch 8, guarded 4 ≤ M ≤ 64</div></div><div class="stat"><div class="val">117 / 117</div><div class="label">Kernel equivalence checks passed<br/>plus extended M, H and dtype matrix; model outputs matched</div></div></div>
<div class="summary"><strong>Finding.</strong> At small M, the default kernel puts several rows in a very small grid. Giving each row its own block exposes more GPU parallelism. A guarded 4 ≤ M ≤ 64 dispatch cuts the model’s norm critical-path time by 12% to 20% at batch 4 to 32, while smaller shapes keep the installed kernel. Outputs matched in every tested case.</div>
<div class="figure"><img src="assets/latency-small-m.svg" alt="Graph showing faster one-row-per-block fused RMSNorm timings versus the default CuTe implementation at M=2–36"/><div class="figcaption">Isolated, L2-warm CUDA Graph experiment. The separate M=2304 no-gain case is included in the full table below.</div></div>
<div class="content-grid"><article class="article">'''+body+'''</article><aside class="toc" aria-label="On this page"><div class="toc-head">In this article</div>
<a href="#question">01 / The question</a><a href="#runtime-path">02 / Runtime path</a><a href="#change">03 / The change</a><a href="#kernel-results">04 / Kernel results</a><a href="#correctness">05 / Correctness</a><a href="#model">06 / Model test</a><a href="#reproduce">07 / Reproduce</a><a href="#full-decode">08 / Integration check</a><a href="#future-work">09 / Future work</a><a href="#conclusion">10 / Findings and limits</a>
</aside></div><footer class="footer">Measured on an RTX 5070 Laptop GPU · Kernel, profiler, and model evidence included · Built as a static page for GitHub Pages</footer></div>
</body></html>'''
(root/'index.html').write_text(html_doc,encoding='utf-8')
print('HTML size:',len(html_doc),'tables:',html_doc.count('<table>'),'links:',html_doc.count('<a '))
