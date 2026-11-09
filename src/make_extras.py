"""Submission extras: one project image combining the key figures, and a PowerPoint copy of the slide deck.
figs/project_image.png = title band + fig_overview + fig_steer_layers + fig_coordtalk (all already in figs/).
slides/slides.pptx    = one full-slide picture per page of slides/slides.pdf (16:9).
"""
from pathlib import Path
import fitz
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figs"
W = 2000
parts = [Image.open(FIG / f).convert("RGB") for f in ["fig_overview.png", "fig_steer_layers.png", "fig_coordtalk.png"]]
parts = [im.resize((W, int(im.height * W / im.width)), Image.LANCZOS) for im in parts]
try:
    f1 = ImageFont.truetype("DejaVuSans-Bold.ttf", 46); f2 = ImageFont.truetype("DejaVuSans.ttf", 30)
except OSError:
    import matplotlib; fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    f1 = ImageFont.truetype(str(fd / "DejaVuSans-Bold.ttf"), 46); f2 = ImageFont.truetype(str(fd / "DejaVuSans.ttf"), 30)
head = Image.new("RGB", (W, 170), "white"); d = ImageDraw.Draw(head)
d.text((60, 35), "Agents Change Names, Not Words", font=f1, fill="#0b0b0b")
d.text((60, 100), "What Oversight Can See in a Real AI Swarm  |  Dhruva P Gowda (TARA)  |  Swarm Alignment Hackathon 2026", font=f2, fill="#52514e")
gap = 30
H = head.height + sum(p.height for p in parts) + gap * len(parts)
out = Image.new("RGB", (W, H), "white"); y = 0
out.paste(head, (0, y)); y += head.height
for p in parts:
    y += gap; out.paste(p, (0, y)); y += p.height
out.save(FIG / "project_image.png", optimize=True)
print("project_image.png", out.size, f"{(FIG / 'project_image.png').stat().st_size / 1e6:.2f} MB")

doc = fitz.open(ROOT / "slides" / "slides.pdf")
prs = Presentation(); prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
cp = prs.core_properties; cp.author = cp.last_modified_by = "Dhruva P Gowda"
cp.title = "Agents Change Names, Not Words: What Oversight Can See in a Real AI Swarm"; cp.comments = ""
tmp = ROOT / "slides" / "_png"; tmp.mkdir(exist_ok=True)
for i, page in enumerate(doc):
    png = tmp / f"s{i:02d}.png"; page.get_pixmap(dpi=200).save(png)
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    sl.shapes.add_picture(str(png), 0, 0, width=prs.slide_width, height=prs.slide_height)
prs.save(ROOT / "slides" / "slides.pptx")
for f in tmp.iterdir():
    f.unlink()
tmp.rmdir()
print("slides.pptx", len(doc), "slides", f"{(ROOT / 'slides' / 'slides.pptx').stat().st_size / 1e6:.2f} MB")
