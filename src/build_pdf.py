"""paper/paper.md -> paper/paper.html -> paper/paper.pdf (headless Chrome)."""
import subprocess, re
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
md = (ROOT / "paper" / "paper.md").read_text(encoding="utf-8")
# Python-Markdown needs a blank line before a list or a new bold-led paragraph; add one where missing.
lines, fixed = md.split("\n"), []
for i, ln in enumerate(lines):
    prev = fixed[-1] if fixed else ""
    starts_block = re.match(r"^(\s*[-*] |\s*\d+\. |\*\*)", ln)
    prev_is_item = re.match(r"^(\s*[-*] |\s*\d+\. |\s{2,}\S)", prev)
    if starts_block and prev.strip() and not prev_is_item and not prev.startswith("|"):
        fixed.append("")
    fixed.append(ln)
md = "\n".join(fixed)
html_body = markdown.markdown(md, extensions=["tables", "attr_list", "sane_lists", "md_in_html"])
# keep the first paragraph after the title (author line) as a subtitle block
css = """
@page { size: A4; margin: 20mm 19mm 20mm 19mm; }
body { font-family: 'Charter', 'Georgia', 'Times New Roman', serif; font-size: 10.4pt; line-height: 1.42; color: #111; }
h1 { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 17pt; line-height: 1.2; margin: 0 0 6pt; text-align: center; }
h1 + p { text-align: center; font-size: 9.5pt; color: #444; margin-bottom: 10pt; }
h2 { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 12pt; margin: 14pt 0 4pt; border-bottom: 0.6pt solid #bbb; padding-bottom: 2pt; }
h3 { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 10.6pt; margin: 10pt 0 3pt; }
p { margin: 0 0 6pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 6pt 16pt; padding: 0; } li { margin-bottom: 2pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 6pt 0 8pt; page-break-inside: avoid; }
th, td { border-top: 0.5pt solid #ccc; border-bottom: 0.5pt solid #ccc; padding: 2.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #f1f1ef; font-family: 'Helvetica Neue', Arial, sans-serif; }
code { font-size: 8.8pt; background: #f3f3f1; padding: 0 2pt; border-radius: 2pt; }
img { display: block; margin: 6pt auto 2pt; max-width: 100%; }
figure, .fig { page-break-inside: avoid; margin: 6pt 0 10pt; }
.fig p { text-align: left; }
em.cap, .fig em { font-size: 8.8pt; color: #333; }
.abstract { margin: 0 6% 10pt; font-size: 9.6pt; }
h2#abstract { text-align: center; border: none; font-size: 10.5pt; }
.pb { page-break-before: always; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><title>Agents Change Names, Not Words</title><style>{css}</style></head><body>{html_body}</body></html>"
html = html.replace('<h2 id="appendix-a', '<h2 class="pb" id="appendix-a', 1)
out_html = ROOT / "paper" / "paper.html"
out_html.write_text(html, encoding="utf-8")
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
pdf = ROOT / "paper" / "paper.pdf"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                out_html.resolve().as_uri()], check=True, capture_output=True)
print("wrote", pdf, pdf.stat().st_size // 1024, "KB")
