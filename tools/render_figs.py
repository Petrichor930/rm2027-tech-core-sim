import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import fitz

RULES = r"D:\26engineer\PinyCaster\.docs\UTF-8__RoboMaster 2027 机甲大师超级对抗赛比赛规则手册V1.0.0（20261015）-1.pdf"
doc = fitz.open(RULES)

# Find PDF pages whose text mentions the target figure captions
targets = ["图 5-6", "图 5-7", "图 5-8"]
for i in range(doc.page_count):
    t = doc[i].get_text()
    for kw in targets:
        if kw in t:
            print(f"pdf page index {i} (1-based {i+1}) contains '{kw}'")

# Render pages 77-82 (1-based) at 150 dpi
for p in range(77, 83):
    page = doc[p - 1]
    pix = page.get_pixmap(dpi=150)
    out = f"figs_page_{p:03d}.png"
    pix.save(out)
    print("saved", out, pix.width, "x", pix.height)
doc.close()
