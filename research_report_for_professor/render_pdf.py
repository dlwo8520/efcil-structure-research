"""Render the report, including LaTeX math and all nine evidence figures.

Dependencies are isolated in .render_deps; never touch the cds environment.
No network, model inference, or training is performed by this script.
"""
from __future__ import annotations
import hashlib
import io
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.render_deps'))

def main():
    import markdown
    from weasyprint import HTML
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.mathtext import math_to_image
    from matplotlib.font_manager import FontProperties
    from xml.etree import ElementTree as ET
    mathdir=ROOT/'figures/math';mathdir.mkdir(exist_ok=True)
    source=(ROOT/'EFCIL_research_report.md').read_text()
    formulas=[]
    def render_tex(tex,display):
        tex=tex.strip().replace(r'\tfrac',r'\frac').replace(r'\mathbf1',r'\mathbf{1}')
        tex=tex.replace(r'\overline\ell',r'\overline{\ell}')
        # Mathtext requires braces even where TeX accepts one-character arguments.
        tex=re.sub(r'\\frac\s*([A-Za-z0-9])',r'\\frac{\1}',tex)
        tex=re.sub(r'\\frac(\{(?:[^{}]|\{[^{}]*\})*\})\s*([A-Za-z0-9])',r'\\frac\1{\2}',tex)
        tex=re.sub(r'\\le\b',r'\\leq',tex)
        tex=re.sub(r'\\ge\b',r'\\geq',tex)
        tex=re.sub(r'\\(mathcal|mathbf|mathbb)\s+([A-Za-z0-9])',r'\\\1{\2}',tex)
        key=hashlib.sha256((str(display)+tex).encode()).hexdigest()[:16]
        target=mathdir/(key+'.svg')
        buf=io.BytesIO()
        math_to_image('$'+tex+'$',buf,prop=FontProperties(size=12 if display else 10),format='svg',dpi=170,color='#182632')
        value=buf.getvalue();target.write_bytes(value)
        xml=ET.fromstring(value);width=float(xml.attrib['width'].replace('pt',''));height=float(xml.attrib['height'].replace('pt',''))
        formulas.append(dict(tex=tex,display=display,source=str(target.relative_to(ROOT)),width_pt=width,height_pt=height))
        if display and width>470 and r'\qquad' in tex:
            parts=tex.split(r'\qquad')
            return '\n'.join(render_tex(part.rstrip(',;'),True) for part in parts if part.strip())
        if display:
            return f'<div class="equation"><img src="{target.relative_to(ROOT)}" style="width:{width:.1f}pt" /></div>'
        return f'<img class="inline-math" src="{target.relative_to(ROOT)}" style="width:{width:.1f}pt;height:{height:.1f}pt" />'
    source=re.sub(r'^\$\$\s*\n(.*?)\n\$\$\s*$',lambda m:render_tex(m.group(1),True),source,flags=re.M|re.S)
    source=re.sub(r'(?<!\$)\$([^\n$]+?)\$(?!\$)',lambda m:render_tex(m.group(1),False),source)
    body=markdown.markdown(source,extensions=['tables','fenced_code','sane_lists','toc'])
    # Keep summary within the first two pages; method starts on a fresh page.
    body=body.replace('<h2 id="1-', '<h2 class="section-start" id="1-',1)
    css='''
    @page {size:A4; margin:16mm 15mm 17mm;
      @bottom-left {content:"EFCIL · 교수님 논의용 · 2026-10-12";font:8pt "Noto Sans CJK KR";color:#6b737a;}
      @bottom-right {content:counter(page);font:8pt sans-serif;color:#6b737a;}}
    body {font-family:"Noto Sans CJK KR",sans-serif;font-size:10pt;line-height:1.52;color:#182632;}
    h1 {font-size:24pt;color:#245b7d;margin:0 0 7mm;line-height:1.2;}
    h2 {font-size:16pt;line-height:1.35;margin:8mm 0 3mm;break-after:avoid;}
    h3 {font-size:12pt;line-height:1.35;margin:5mm 0 2mm;break-after:avoid;}
    p {margin:0 0 3mm;orphans:3;widows:3;word-break:normal;}
    strong {font-weight:700;} a {color:#245b7d;text-decoration:none;}
    code {font-family:"DejaVu Sans Mono",monospace;font-size:8.4pt;overflow-wrap:anywhere;}
    ul,ol {padding-left:5mm;margin:2mm 0 4mm;}li {margin:1.4mm 0;}
    table {width:100%;border-collapse:collapse;font-size:8.3pt;line-height:1.4;margin:4mm 0;}
    th {background:#e6eff3;text-align:left;font-weight:700;}th,td {padding:2mm;border-bottom:.3pt solid #becad1;overflow-wrap:anywhere;}
    tr {break-inside:avoid;}thead {display:table-header-group;}
    img {max-width:100%;height:auto;}p:has(>img) {break-inside:avoid;margin-top:3mm;margin-bottom:2mm;}
    p:has(>img)+p {font-size:8.6pt;line-height:1.4;color:#44525c;break-before:avoid;}
    .equation {text-align:center;margin:3mm 0;break-inside:avoid;line-height:1.1;}
    .equation img {max-width:100%;height:auto;}
    .inline-math {vertical-align:-.22em;max-width:100%;}
    .section-start {break-before:page;}
    hr {border:0;border-top:.5pt solid #c7d3d9;margin:5mm 0;}
    '''
    html='<!doctype html><html lang="ko"><meta charset="utf-8"><title>EFCIL 전체 연구 보고서</title><style>'+css+'</style><body>'+body+'</body></html>'
    (ROOT/'EFCIL_research_report.html').write_text(html)
    output=ROOT/'EFCIL_research_report.pdf'
    HTML(string=html,base_url=str(ROOT)).write_pdf(output)
    (ROOT/'appendix/pdf_render_manifest.json').write_text(json.dumps(dict(
        renderer='WeasyPrint 66.0; Python-Markdown 3.9; matplotlib mathtext',
        source_sha256=hashlib.sha256((ROOT/'EFCIL_research_report.md').read_bytes()).hexdigest(),
        pdf_bytes=output.stat().st_size,formulas_rendered=len(formulas),
        font='Noto Sans CJK KR',formulas=formulas),ensure_ascii=False,indent=2)+'\n')
    print('Rendered:',output,'bytes=',output.stat().st_size,'math expressions=',len(formulas))

if __name__=='__main__':main()
